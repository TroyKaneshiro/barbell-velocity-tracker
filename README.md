# RPE Estimator

A velocity-based training (VBT) tool that analyses a video of a barbell lift and estimates **RPE**, **mean concentric velocity**, and **projected 1RM** using lift-specific regression models from peer-reviewed research.

---

## How It Works

1. Upload a side-on video of a squat, bench press, or deadlift
2. Select the lift type, outer plate weight, and optionally the total bar weight
3. Left-click the plate in the preview frame to seed tracking; right-click the outer rim to manually override the detected circle if needed
4. Click **Analyze** — results appear in a few seconds

The backend tracks the plate frame-by-frame, converts pixel displacement to metres using the known plate diameter as a calibration reference, computes bar velocity, detects the concentric phase, and maps mean concentric velocity (MCV) to RPE via the Helms et al. 2017 regression.

---

## Results

| Field | Description |
|---|---|
| RPE | Estimated rate of perceived exertion (6.0–10.0, ±0.5) |
| Mean concentric velocity | MCV in m/s — the primary input to RPE lookup |
| Peak concentric velocity | Highest velocity during the concentric phase |
| Projected 1RM | Estimated max (requires bar weight input) |
| % of 1RM | Intensity of the set |
| Velocity chart | Frame-by-frame velocity with eccentric / concentric / burst regions highlighted |
| Debug video | Annotated playback showing the tracker and per-frame velocity + phase label |

---

## Setup

### Requirements

- Python 3.10+
- ffmpeg (optional — used to convert debug video to MP4; falls back to AVI if unavailable)

### Install

```bash
pip install -r backend/requirements.txt
```

### Run

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Then open `http://localhost:8000` in a browser.

---

## Plate Detection

On first frame, a Hough circle transform searches the region around the user's click to find the plate boundary. This circle sets the pixel-to-metre calibration and the initial CSRT tracker bounding box.

If the detected circle lands on an inner bezel instead of the outer rim:
- **Right-click** on the outer edge of the plate — this sets the radius manually and overrides Hough

---

## RPE & 1RM Model

All three lifts use regression equations from **Helms et al. 2017** (J Strength Cond Res 31(2), [PubMed 27243918](https://pubmed.ncbi.nlm.nih.gov/27243918/)) — a study of competitive powerlifters.

| Lift | Equation | r | R² |
|---|---|---|---|
| Squat | %1RM = −0.449 × MCV + 1.096 | −0.91 | 0.83 |
| Bench | %1RM = −0.600 × MCV + 1.051 | −0.90 | 0.81 |
| Deadlift | %1RM = −0.600 × MCV + 1.076 | −0.92 | 0.85 |

Projected 1RM = bar weight / %1RM

**Caveats:**
- Tables are validated for trained/competitive lifters performing single reps
- Individual velocity variation of ±0.05–0.10 m/s → RPE uncertainty of ~±0.5
- Most reliable at RPE ≥ 8; less precise at lower intensities
- Multi-rep sets: the last rep is used (closest to failure)

See [METHODOLOGY.md](METHODOLOGY.md) for full details on the model, calibration approach, and population considerations.

---

## Project Structure

```
RPEApp/
├── backend/
│   ├── main.py          # FastAPI app and API endpoints
│   ├── tracker.py       # Hough circle detection + CSRT plate tracker
│   ├── velocity.py      # Velocity calculation, phase detection, MCV
│   ├── rpe_tables.py    # RPE lookup tables and projected 1RM function
│   ├── plates.py        # Supported plate sizes and diameters
│   └── requirements.txt
├── frontend/
│   ├── index.html       # UI
│   ├── app.js           # Upload, plate click UI, chart rendering, polling
│   ├── style.css        # Styles
│   └── debug/           # Generated debug videos (runtime)
└── rpe_velocity_research.txt   # Research notes and source data
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Serve frontend |
| GET | `/plates` | List supported plates |
| POST | `/detect-plate` | Hough detection on first frame near click |
| POST | `/analyze` | Full analysis — returns RPE, velocity, chart data, debug video URL |
