# Plate Detector Training

Trains the YOLO11n model used by `backend/tracker.py` (`_detect_plate_yolo`).
This is a one-off/occasional local training workflow, kept separate from the
app's serving dependencies — nothing here is imported by `backend/`, and
none of it runs in production.

**Current dataset**: a premade Roboflow dataset (project `barbell-i86fn-8voka`),
cleaned down to a single `plate` class in `dataset/train/` (863 images) and
`dataset/valid/` (154 images). The original export had 3 classes — `0`
(barbell sleeve end-cap), `Bar`, and `Barbell` (the latter two both boxed
the round weight plate, a redundant naming collision from the source data).
`Bar` and `0` were dropped entirely (verified visually — `0` boxes the
sleeve cap, not the plate), and the 110 images that had no plate box left
after dropping them were removed too (they were isolated product-style
photos of a bare bar, not lift footage). `Barbell` was kept and remapped to
class `0` = `plate`. `data.yaml` already points at the cleaned set, so no
frame extraction or labeling is needed to start training — skip straight
to Step 2.

Because the dataset is now purely round plate boxes, `_detect_plate_yolo`'s
`max(box_w, box_h) / 2` square→circle conversion is the correct geometry
here — no bypass logic needed.

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

## 2. Train

```bash
python train.py
```

Defaults: `yolo11n.pt` base checkpoint, 100 epochs, batch 16, imgsz 640,
`device=0` (your 4060). Override with flags, e.g. `python train.py
--epochs 150 --batch 8` if you hit a VRAM ceiling. Output goes to
`runs/plate/weights/best.pt` (gitignored — large binary, not checked in).

## 3. Export to the backend

```bash
python export.py
```

Exports `runs/plate/weights/best.pt` to ONNX and copies it to
`backend/yolo_plate.onnx` — the path `backend/tracker.py` already looks for
by default (see `YOLO_MODEL_PATH` in `backend/tracker.py`).

## 4. Verify

Follow the verification steps in `CLAUDE.md` / the plate-detection plan:
a standalone frame check, then the `/detect-plate` endpoint via the UI
(green overlay circle should tightly wrap the outer rim, and the server
log should say `[tracker] plate detected by YOLO: ...` rather than falling
back to Hough), then a full upload → analyze pass.

## Extending the dataset (not needed to run Steps 1-4 above)

Not used to build the current dataset — it's a premade Roboflow set. Use
this only if you later want to add your own footage on top of it (e.g. to
cover gyms/plates/angles it doesn't already have, or to relabel toward the
plate specifically per the note above).

```bash
python extract_frames.py "C:\path\to\squat1.mp4" --fps 2 --out dataset/raw
python extract_frames.py "C:\path\to\bench1.mp4" --fps 2 --out dataset/raw
```

Run across several videos covering different plate colors, lighting, and
gyms — not just one clip. Then upload the frames to the same Roboflow
project (`troys-workspace-4zp5s/barbell-i86fn-8voka`, see `data.yaml`) and
label:

- Single class: `plate`. Draw a tight axis-aligned box around the visible
  **outer rim** of the plate — a loose box directly inflates the
  pixel→metre calibration error downstream. Don't reintroduce the old
  `Bar`/`0` classes.
- Target **200-400 labeled images minimum**. More if variety is limited.
- Split 80/20 or 85/15 train/val (Roboflow does this for you on export).

Export the Roboflow project in **"YOLO11" format**, then unzip it so the
contents land in `dataset/train/{images,labels}/` and
`dataset/valid/{images,labels}/` — matching what `data.yaml` already
expects (re-exporting overwrites in place; if Roboflow re-adds other
classes from the source project, re-run the same class-drop cleanup before
training). See `dataset/README.md` for the exact structure.
