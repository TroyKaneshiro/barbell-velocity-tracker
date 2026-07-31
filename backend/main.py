"""
RPE Estimator – FastAPI backend
"""

import os
import shutil
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path

import cv2
import numpy as np

from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from tracker import BarTracker, _get_yolo_net
from velocity import calculate_velocity
from rpe_tables import velocity_to_rpe, projected_1rm
from plates import PLATE_DIAMETERS, DEFAULT_PLATE, get_diameter

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="RPE Estimator", version="0.1.0")

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
DEBUG_DIR    = FRONTEND_DIR / "debug"
DEBUG_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
app.mount("/debug",  StaticFiles(directory=str(DEBUG_DIR)),    name="debug")


@app.on_event("startup")
async def _warm_yolo_model():
    try:
        _get_yolo_net()
        print("[main] YOLO plate model loaded")
    except FileNotFoundError as exc:
        print(f"[main] {exc} — falling back to Hough detection until weights are added")


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": f"Internal error: {exc}"})


@app.get("/", response_class=HTMLResponse)
async def index():
    return (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Analysis endpoint
# ---------------------------------------------------------------------------

VALID_LIFTS = {"squat", "bench", "deadlift"}


@app.get("/plates")
async def list_plates():
    return {"plates": list(PLATE_DIAMETERS.keys()), "default": DEFAULT_PLATE}


@app.post("/detect-plate")
async def detect_plate(
    video: UploadFile = File(...),
    click_x: float = Form(...),
    click_y: float = Form(...),
):
    """
    Run plate detection on the first frame near the user's click.
    Returns normalised circle coordinates so the frontend can overlay them.
    """
    suffix   = Path(video.filename or "video.mp4").suffix or ".mp4"
    tmp_path = tempfile.mktemp(suffix=suffix)
    try:
        with open(tmp_path, "wb") as fh:
            shutil.copyfileobj(video.file, fh)

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(422, "Cannot open video")
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise HTTPException(422, "Cannot read first frame")

        h, w = frame.shape[:2]
        tracker = BarTracker()
        circle  = tracker._detect_plate(frame, click_x, click_y, w, h)

        if circle is None:
            return {"found": False}

        cx, cy, r = circle
        return {
            "found":  True,
            "cx_norm": cx / w,
            "cy_norm": cy / h,
            "r_norm":  r  / min(w, h),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Detection error: {exc}")
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass


@app.post("/analyze")
async def analyze(
    video: UploadFile = File(...),
    lift_type: str = Form(...),
    plate: str = Form(DEFAULT_PLATE),
    click_x: float = Form(...),
    click_y: float = Form(...),
    plate_r_norm: Optional[float] = Form(default=None),
    bar_weight: Optional[float] = Form(default=None),
):
    lift_type = lift_type.lower().strip()
    if lift_type not in VALID_LIFTS:
        raise HTTPException(400, f"lift_type must be one of: {', '.join(VALID_LIFTS)}")

    try:
        plate_diameter_m = get_diameter(plate)
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    suffix   = Path(video.filename or "video.mp4").suffix or ".mp4"
    tmp_path = tempfile.mktemp(suffix=suffix)

    stem           = uuid.uuid4().hex
    debug_avi      = str(DEBUG_DIR / f"debug_{stem}.avi")
    debug_filename = f"debug_{stem}.avi"  # updated to .mp4 after conversion

    try:
        with open(tmp_path, "wb") as fh:
            shutil.copyfileobj(video.file, fh)

        # ── Track ────────────────────────────────────────────────────
        tracking = BarTracker(lift_type=lift_type).process_video(
            tmp_path, debug_video_path=debug_avi, click_x=click_x, click_y=click_y,
            plate_r_norm=plate_r_norm,
        )

        # ── Velocity ─────────────────────────────────────────────────
        vel = calculate_velocity(tracking, plate_diameter_m=plate_diameter_m, lift_type=lift_type)

        # ── RPE lookup ───────────────────────────────────────────────
        mcv        = vel["mean_concentric_velocity"]
        rpe_result = velocity_to_rpe(lift_type, mcv)
        rm_result  = projected_1rm(lift_type, mcv, bar_weight) if bar_weight else None

        # ── Debug video: annotate + convert in background ─────────────
        # The mp4 filename is deterministic so the client can poll/load it
        # once it appears; results are returned immediately without waiting.
        mp4_filename = debug_filename.replace(".avi", ".mp4")

        def _process_debug():
            _annotate_debug_video(debug_avi, vel)
            _convert_to_mp4(DEBUG_DIR, debug_filename)
            _prune_debug_videos(DEBUG_DIR, keep=5, latest=mp4_filename)

        threading.Thread(target=_process_debug, daemon=True).start()

        return {
            "lift_type":                 lift_type,
            "rpe":                       rpe_result["rpe"],
            "rpe_description":           rpe_result["description"],
            "rpe_note":                  rpe_result.get("note"),
            "projected_1rm":             rm_result["projected_1rm"] if rm_result else None,
            "percent_1rm":               rm_result["percent_1rm"]   if rm_result else None,
            "rm_note":                   rm_result["note"]           if rm_result else None,
            "mean_concentric_velocity":  mcv,
            "peak_concentric_velocity":  vel["peak_concentric_velocity"],
            "debug_video_url":           f"/debug/{mp4_filename}",
            # chart data
            "time":                      vel["time"],
            "velocity":                  vel["velocity"],
            "position_m":                vel["position_m"],
            "eccentric_start":           vel.get("eccentric_start", 0),
            "concentric_start":          vel["concentric_start"],
            "concentric_end":            vel["concentric_end"],
            # calibration
            "calibration": {
                "fps":               vel["fps"],
                "plate":             plate,
                "plate_diameter_m":  plate_diameter_m,
                "plate_diameter_px": vel["plate_diameter_px"],
                "m_per_px":          vel["m_per_px"],
            },
        }

    except ValueError as exc:
        raise HTTPException(422, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"Processing error: {exc}")
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _annotate_debug_video(video_path: str, vel: dict) -> None:
    """Re-encode debug video adding per-frame velocity value and phase label."""
    velocities      = vel["velocity"]
    eccentric_start = vel["eccentric_start"]
    concentric_start= vel["concentric_start"]
    concentric_end  = vel["concentric_end"]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tmp_path = video_path + ".annot.avi"
    fourcc   = cv2.VideoWriter_fourcc(*"MJPG")
    writer   = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        cap.release()
        return

    PHASE_COLORS = {
        "SETUP":      (160, 160, 160),
        "ECCENTRIC":  (200, 130,  60),
        "CONCENTRIC": ( 50, 200, 100),
        "LOCKOUT":    (160, 160, 160),
    }

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        v = velocities[frame_idx] if frame_idx < len(velocities) else 0.0

        if frame_idx < eccentric_start:
            phase = "SETUP"
        elif frame_idx < concentric_start:
            phase = "ECCENTRIC"
        elif frame_idx < concentric_end:
            phase = "CONCENTRIC"
        else:
            phase = "LOCKOUT"

        color = PHASE_COLORS[phase]
        font  = cv2.FONT_HERSHEY_SIMPLEX

        # Semi-transparent box, top-right
        box_w, box_h, margin = 280, 74, 12
        box_x1 = w - box_w - margin
        box_y1 = margin
        box_x2 = w - margin
        box_y2 = margin + box_h

        overlay = frame.copy()
        cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        vel_text = f"v = {v:+.3f} m/s"
        (vel_w, _), _ = cv2.getTextSize(vel_text, font, 0.9, 2)
        cv2.putText(frame, vel_text, (box_x2 - vel_w - 14, box_y1 + 34), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        (phase_w, _), _ = cv2.getTextSize(phase, font, 0.75, 2)
        cv2.putText(frame, phase, (box_x2 - phase_w - 14, box_y1 + 60), font, 0.75, color, 2, cv2.LINE_AA)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    os.replace(tmp_path, video_path)


def _convert_to_mp4(directory: Path, avi_filename: str) -> str:
    """Convert MJPEG .avi → H.264 .mp4 via ffmpeg. Returns served filename.

    ffmpeg writes to a temp path and we rename into place only once it's
    fully written — the client polls the final filename, and +faststart
    requires ffmpeg to rewrite the file after encoding to relocate the moov
    atom, so a partially-written file at the final path would look "ready"
    (HEAD 200) to the poller while still being an unplayable container.
    """
    avi_path     = directory / avi_filename
    mp4_filename = avi_filename.replace(".avi", ".mp4")
    mp4_path     = directory / mp4_filename
    tmp_path     = directory / f"{mp4_filename}.tmp"

    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(avi_path),
             "-vcodec", "libx264", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", "-f", "mp4", str(tmp_path)],
            capture_output=True, timeout=120,
        )
        if result.returncode == 0 and tmp_path.exists():
            os.replace(tmp_path, mp4_path)
            avi_path.unlink(missing_ok=True)
            print(f"[main] debug video converted: {mp4_filename}")
            return mp4_filename
        tmp_path.unlink(missing_ok=True)
        print(f"[main] ffmpeg failed (rc={result.returncode}): {result.stderr.decode(errors='replace')[-500:]}")
        return avi_filename
    except (FileNotFoundError, subprocess.TimeoutExpired):
        tmp_path.unlink(missing_ok=True)
        print("[main] ffmpeg not found, serving avi")
        return avi_filename


def _prune_debug_videos(directory: Path, keep: int, latest: str) -> None:
    files = sorted(
        [f for f in directory.glob("debug_*") if f.name != latest],
        key=lambda f: f.stat().st_mtime,
    )
    for f in files[: max(0, len(files) - keep + 1)]:
        try:
            f.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
