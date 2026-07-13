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
            "burst_start":               vel["burst_start"],
            "burst_end":                 vel["burst_end"],
            "burst_threshold":           vel["burst_threshold"],
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
    burst_start     = vel["burst_start"]
    burst_end       = vel["burst_end"]

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
        "CONCENTRIC": ( 60,  60, 210),
        "BURST":      ( 50, 200, 100),
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
        elif burst_start <= frame_idx < burst_end:
            phase = "BURST"
        elif frame_idx < concentric_end:
            phase = "CONCENTRIC"
        else:
            phase = "LOCKOUT"

        color = PHASE_COLORS[phase]
        font  = cv2.FONT_HERSHEY_SIMPLEX

        # Semi-transparent bar at bottom
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 56), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        cv2.putText(frame, f"v = {v:+.3f} m/s", (10, h - 32), font, 0.65, (220, 220, 220), 2)
        cv2.putText(frame, phase,                (10, h -  8), font, 0.55, color,           2)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    os.replace(tmp_path, video_path)


def _convert_to_mp4(directory: Path, avi_filename: str) -> str:
    """Convert MJPEG .avi → H.264 .mp4 via ffmpeg. Returns served filename."""
    avi_path     = directory / avi_filename
    mp4_filename = avi_filename.replace(".avi", ".mp4")
    mp4_path     = directory / mp4_filename

    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(avi_path),
             "-vcodec", "libx264", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(mp4_path)],
            capture_output=True, timeout=120,
        )
        if result.returncode == 0 and mp4_path.exists():
            avi_path.unlink(missing_ok=True)
            print(f"[main] debug video converted: {mp4_filename}")
            return mp4_filename
        print(f"[main] ffmpeg failed (rc={result.returncode}), serving avi")
        return avi_filename
    except (FileNotFoundError, subprocess.TimeoutExpired):
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
