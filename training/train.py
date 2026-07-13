"""
Train the plate-detection YOLOv8n model.

Usage:
    python train.py [--epochs 100] [--batch 16] [--imgsz 640]

Expects labeled data in dataset/images/{train,val} and
dataset/labels/{train,val} (Ultralytics YOLO format) — see README.md for
the labeling workflow. Output lands in runs/plate/weights/best.pt.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--model", default="yolov8n.pt", help="Base checkpoint to fine-tune from")
    parser.add_argument("--device", default="0", help="CUDA device index, or 'cpu'")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=str(HERE / "plate.yaml"),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=str(HERE / "runs"),
        name="plate",
    )


if __name__ == "__main__":
    main()
