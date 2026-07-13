# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

RPE Tracker is a velocity-based training (VBT) analysis tool that processes barbell lift videos to estimate RPE (Rate of Perceived Exertion), Mean Concentric Velocity (MCV), and projected 1RM. It uses computer vision (OpenCV CSRT tracking + YOLO plate detection, with a Hough circle fallback) with biomechanics lookup tables based on Helms et al. 2017.

## Running the App

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Then open `http://localhost:8000`. No build step required — frontend is static HTML/CSS/JS served by FastAPI.

## Installing Dependencies

```bash
pip install -r backend/requirements.txt
# ffmpeg is optional (for MP4 debug video conversion — AVI fallback works without it)
```

## No Automated Tests

This is a single-developer research tool with no test suite. Validate changes manually via the UI: upload a side-on barbell video, click the plate to detect it, and run analysis.

## Architecture

The app is a Python FastAPI backend + vanilla JS frontend. All CV and biomechanics logic lives in the backend; the frontend is a stateless form UI.

### Request Pipeline

```
POST /detect-plate  →  YOLO plate detection on first video frame (Hough fallback)
POST /analyze       →  Full pipeline:
    tracker.process_video()     — CSRT tracking frame-by-frame (at 50% scale)
    calculate_velocity()        — Position → velocity (Savitzky-Golay smoothed)
    _find_rep_phases()          — Phase detection (squat/bench) or
    _find_rep_phases_deadlift() — (deadlift-specific)
    velocity_to_rpe()           — Helms regression lookup
    projected_1rm()             — 1RM estimate
    Background thread: annotate AVI → convert MP4, prune old debug videos
```

Results return immediately; the frontend polls `/debug/{filename}.mp4` until the conversion finishes.

### Module Responsibilities

| File | Role |
|------|------|
| [backend/main.py](backend/main.py) | FastAPI app, `/analyze` and `/detect-plate` endpoints, temp file cleanup |
| [backend/tracker.py](backend/tracker.py) | CSRT tracking, YOLO plate detection (Hough fallback), debug video rendering |
| [backend/velocity.py](backend/velocity.py) | Velocity calculation, phase detection, MCV burst window |
| [backend/rpe_tables.py](backend/rpe_tables.py) | Helms et al. 2017 regression tables, RPE↔%1RM interpolation |
| [backend/plates.py](backend/plates.py) | Plate diameter reference table (lb/kg sizes) for pixel calibration |
| [frontend/app.js](frontend/app.js) | Event handlers, canvas overlay, API calls, Chart.js rendering |

### Key Distinctions

**Deadlift vs. squat/bench**: Different phase-detection functions. Deadlift starts at floor (global position minimum before lockout); squat/bench start at the eccentric top. See `_find_rep_phases_deadlift()` vs `_find_rep_phases()` in [backend/velocity.py](backend/velocity.py).

**Pixel calibration**: The outer plate diameter is the reference object. `m_per_px = plate_diameter_m / plate_diameter_px`. Every velocity calculation depends on this.

**Plate detection**: `_detect_plate()` tries YOLO (`_detect_plate_yolo()`, ONNX model via `cv2.dnn`) first, falling back to the localised Hough search (`_find_circle_near_click()`) if the model weights are missing or detection fails. The YOLO model is loaded once per process via a module-level singleton (`_get_yolo_net()`) rather than per-request. Weights are expected at `backend/yolo_plate.onnx` (overridable via the `YOLO_MODEL_PATH` env var) and are gitignored — the app runs fine without them, just using Hough only.

**Tracking robustness**: CSRT runs at 50% scale (`TRACK_SCALE = 0.50`), re-initialises after 5 consecutive failures or jumps > `2.0 × plate_r` per frame.

**MCV burst window**: MCV is computed only over frames where velocity ≥ 20% of peak (`BURST_FRACTION = 0.20`), matching GymAware/PUSH device methodology.

### Frontend State (app.js globals)

- `plateClick` — `{normX, normY}` normalized (0–1) coordinates of the user's left-click
- `detectedCircle` — `{cx_norm, cy_norm, r_norm}` from the `/detect-plate` response (YOLO or Hough)
- `manualEdge` — Right-click point for manual radius override
- `analyzeAbortCtrl` — AbortController for cancelling in-flight `/analyze` requests

### Naming Conventions

- `_norm` suffix — normalised 0–1 coordinates
- `_s` suffix — scaled frame coordinates (50% downscale)
- `_m` suffix — metres; `_px` suffix — pixels
- `_idx` — frame index

## Key Constants

Defined in [backend/tracker.py](backend/tracker.py) and [backend/velocity.py](backend/velocity.py):

```python
TRACK_SCALE = 0.50          # Video downscale for CSRT tracking
ROI_FACTOR = 3.5            # ROI half-width = plate_r × ROI_FACTOR
MAX_CSRT_FAILURES = 5       # Re-init threshold
MAX_FRAME_JUMP = 2.0        # Max drift per frame (× plate_r)
BURST_FRACTION = 0.20       # MCV window lower bound
MIN_PHASE_FRAMES = 4        # Min consecutive frames to confirm a phase

YOLO_MODEL_PATH = "backend/yolo_plate.onnx"  # overridable via YOLO_MODEL_PATH env var
YOLO_INPUT_SIZE = (640, 640)
YOLO_CONF_THRESH = 0.30     # min detection confidence
YOLO_NMS_THRESH = 0.40      # non-max suppression IoU threshold
```

## Research Basis

See [METHODOLOGY.md](METHODOLOGY.md) for the Helms et al. 2017 regression equations, %1RM↔RPE tables, population notes, and calibration approach. This context matters when modifying [backend/rpe_tables.py](backend/rpe_tables.py).
