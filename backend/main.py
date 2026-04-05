"""
RPE Estimator – FastAPI backend
"""

import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from tracker import BarTracker
from velocity import calculate_velocity
from rpe_tables import velocity_to_rpe
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


@app.post("/analyze")
async def analyze(
    video: UploadFile = File(...),
    lift_type: str = Form(...),
    plate: str = Form(DEFAULT_PLATE),
    click_x: float = Form(...),
    click_y: float = Form(...),
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
            tmp_path, debug_video_path=debug_avi, click_x=click_x, click_y=click_y
        )

        # ── Convert debug video to browser-playable mp4 ──────────────
        debug_filename = _convert_to_mp4(DEBUG_DIR, debug_filename)

        # ── Velocity ─────────────────────────────────────────────────
        vel = calculate_velocity(tracking, plate_diameter_m=plate_diameter_m)

        # ── RPE lookup ───────────────────────────────────────────────
        mcv        = vel["mean_concentric_velocity"]
        rpe_result = velocity_to_rpe(lift_type, mcv)

        _prune_debug_videos(DEBUG_DIR, keep=5, latest=debug_filename)

        return {
            "lift_type":                 lift_type,
            "rpe":                       rpe_result["rpe"],
            "rpe_description":           rpe_result["description"],
            "rpe_note":                  rpe_result.get("note"),
            "mean_concentric_velocity":  mcv,
            "peak_concentric_velocity":  vel["peak_concentric_velocity"],
            "debug_video_url":           f"/debug/{debug_filename}",
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
