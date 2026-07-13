# Dataset layout

This directory is gitignored (except this file) — labeled images and
annotations are large and local-only. Populate it via the Roboflow export
described in `../README.md`, so it ends up matching:

```
dataset/
  raw/                  # frames from extract_frames.py, pre-labeling
  images/
    train/
    val/
  labels/
    train/
    val/
```

`images/*` and `labels/*` use Ultralytics YOLO format: one `.jpg`/`.png`
per image in `images/`, one same-named `.txt` per image in `labels/`
(`class_id x_center y_center width height`, all normalized 0-1). A
Roboflow "YOLOv8" export unzips directly into this shape — just merge its
`train/images`, `train/labels`, `valid/images`, `valid/labels` folders
into the paths above (rename `valid` → `val` to match `plate.yaml`).
