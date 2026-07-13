# Plate Detector Training

Trains the single-class ("plate") YOLOv8n model used by `backend/tracker.py`
(`_detect_plate_yolo`). This is a one-off/occasional local training workflow,
kept separate from the app's serving dependencies — nothing here is imported
by `backend/`, and none of it runs in production.

## 1. Setup

Use a separate virtual environment from the backend (this pulls in torch,
which the server doesn't need):

```bash
cd training
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Confirm CUDA is visible to torch (should print your GTX 4060):

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

If that fails, reinstall torch with a CUDA-enabled wheel per
https://pytorch.org/get-started/locally/ before continuing — `pip install
ultralytics` alone doesn't guarantee the right CUDA build on every system.

## 2. Extract frames from existing lift videos

```bash
python extract_frames.py "C:\path\to\squat1.mp4" --fps 2 --out dataset/raw
python extract_frames.py "C:\path\to\bench1.mp4" --fps 2 --out dataset/raw
```

Run this across several videos covering different plate colors, lighting,
and gyms — not just one clip. Frame 0 of a single video isn't enough
variety; see the "what to capture" list below.

## 3. Label

Upload `dataset/raw/*.jpg` to [Roboflow](https://roboflow.com) (free tier).

- Single class: `plate`
- Draw a tight axis-aligned box around the visible **outer rim** of the
  plate (this is what `_detect_plate_yolo` converts to a radius via
  `max(box_w, box_h) / 2` — a loose box directly inflates the pixel→metre
  calibration error).
- Target **200-400 labeled images minimum**. More if variety is limited.
- Split 80/20 or 85/15 train/val (Roboflow does this for you on export).

**What to capture**: plate colors (bumper red/blue/yellow, iron
black/chrome), lighting (home vs. commercial gym, natural vs. fluorescent),
partial occlusion (hands, collars, chalk), a few motion-blurred frames.

Export the Roboflow project in **"YOLOv8" format** (zip download or direct
`roboflow` package pull), then unzip it so the contents land in
`dataset/images/{train,val}/` and `dataset/labels/{train,val}/` — matching
the layout `plate.yaml` expects. See `dataset/README.md` for the exact
structure.

## 4. Train

```bash
python train.py
```

Defaults: `yolov8n.pt` base checkpoint, 100 epochs, batch 16, imgsz 640,
`device=0` (your 4060). Override with flags, e.g. `python train.py
--epochs 150 --batch 8` if you hit a VRAM ceiling. Output goes to
`runs/plate/weights/best.pt` (gitignored — large binary, not checked in).

## 5. Export to the backend

```bash
python export.py
```

Exports `runs/plate/weights/best.pt` to ONNX and copies it to
`backend/yolo_plate.onnx` — the path `backend/tracker.py` already looks for
by default (see `YOLO_MODEL_PATH` in `backend/tracker.py`).

## 6. Verify

Follow the verification steps in `CLAUDE.md` / the plate-detection plan:
a standalone frame check, then the `/detect-plate` endpoint via the UI
(green overlay circle should tightly wrap the outer rim, and the server
log should say `[tracker] plate detected by YOLO: ...` rather than falling
back to Hough), then a full upload → analyze pass.
