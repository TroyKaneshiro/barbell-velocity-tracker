"""
RPE Estimator – FastAPI backend
"""

import os
import shutil
import tempfile
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
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    """Ensure all unhandled errors return JSON so the frontend can parse them."""
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
    """Return available plate options for the frontend selector."""
    return {"plates": list(PLATE_DIAMETERS.keys()), "default": DEFAULT_PLATE}


@app.post("/analyze")
async def analyze(
    video: UploadFile = File(...),
    lift_type: str = Form(...),
    plate: str = Form(DEFAULT_PLATE),
):
    lift_type = lift_type.lower().strip()
    if lift_type not in VALID_LIFTS:
        raise HTTPException(400, f"lift_type must be one of: {', '.join(VALID_LIFTS)}")

    try:
        plate_diameter_m = get_diameter(plate)
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    # ── Save upload to a temp file ────────────────────────────────────
    suffix   = Path(video.filename or "video.mp4").suffix or ".mp4"
    tmp_path = tempfile.mktemp(suffix=suffix)

    try:
        with open(tmp_path, "wb") as fh:
            shutil.copyfileobj(video.file, fh)

        # ── Track ────────────────────────────────────────────────────
        tracking = BarTracker().process_video(tmp_path)

        # ── Velocity ─────────────────────────────────────────────────
        vel = calculate_velocity(tracking, plate_diameter_m=plate_diameter_m)

        # ── RPE lookup ───────────────────────────────────────────────
        mcv        = vel["mean_concentric_velocity"]
        rpe_result = velocity_to_rpe(lift_type, mcv)

        return {
            "lift_type":                 lift_type,
            "rpe":                       rpe_result["rpe"],
            "rpe_description":           rpe_result["description"],
            "rpe_note":                  rpe_result.get("note"),
            "mean_concentric_velocity":  mcv,
            "peak_concentric_velocity":  vel["peak_concentric_velocity"],
            # chart data
            "time":                      vel["time"],
            "velocity":                  vel["velocity"],
            "position_m":                vel["position_m"],
            "concentric_start":          vel["concentric_start"],
            "concentric_end":            vel["concentric_end"],
            # debug / calibration
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
            pass  # Windows may still hold the handle briefly; file will be cleaned up on next OS temp purge


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
