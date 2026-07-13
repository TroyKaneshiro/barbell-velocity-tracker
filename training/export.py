"""
Export a trained checkpoint to ONNX and copy it into the backend.

Usage:
    python export.py [--weights runs/plate/weights/best.pt]

Writes to backend/yolo_plate.onnx, the default path backend/tracker.py
looks for (see YOLO_MODEL_PATH).
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).parent
BACKEND_TARGET = HERE.parent / "backend" / "yolo_plate.onnx"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(HERE / "runs" / "plate" / "weights" / "best.pt"))
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"Checkpoint not found: {weights} — run train.py first")

    model = YOLO(str(weights))
    exported = model.export(format="onnx", opset=12, imgsz=640, simplify=True)

    shutil.copy(exported, BACKEND_TARGET)
    print(f"Copied {exported} -> {BACKEND_TARGET}")


if __name__ == "__main__":
    main()
