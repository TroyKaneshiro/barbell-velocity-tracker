# Dataset layout

This directory is gitignored (except this file) — labeled images and
annotations are large and local-only. Currently populated by a Roboflow
"YOLO11" export (project `barbell-i86fn-8voka`, see `../data.yaml`):

```
dataset/
  raw/                  # frames from extract_frames.py, pre-labeling
  train/
    images/
    labels/
  valid/
    images/
    labels/
```

`images/` and `labels/` use Ultralytics YOLO format: one `.jpg`/`.png` per
image, one same-named `.txt` per image in the matching `labels/` folder
(`class_id x_center y_center width height`, all normalized 0-1). A
Roboflow "YOLO11" export unzips directly into this shape — just drop its
`train/` and `valid/` folders in here as-is (no renaming needed; `../data.yaml`
already points at `train/images` and `valid/images`).

Single class (`0` = `plate`) only. The original export had two extra
classes (a barbell sleeve end-cap, and a duplicate "Bar" label pointing at
the same plate object as a third class) — those were stripped out along
with the images that only had those objects and no plate box. If you
re-export from Roboflow and the source project still has those classes,
re-run the same drop-and-remap cleanup before training — don't just point
`data.yaml` at a fresh unfiltered export.
