"""
Train the plate-detection YOLO11n model.

Usage:
    python train.py [--data data.yaml] [--epochs 100] [--batch 16] [--imgsz 640] [--name plate]

Expects labeled data in dataset/train/{images,labels} and
dataset/valid/{images,labels} (Ultralytics YOLO format, as exported by
Roboflow — see data.yaml). Output lands in runs/<name>/weights/best.pt.
"""
import argparse
from pathlib import Path
from ultralytics import YOLO

HERE = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data.yaml", help="Dataset config, relative to this file")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--model", default="yolo11n.pt", help="Base checkpoint to fine-tune from")
    parser.add_argument("--device", default="0", help="CUDA device index, or 'cpu'")
    parser.add_argument("--name", default="plate", help="Run name, under runs/<name>/")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=str(HERE / args.data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=str(HERE / "runs"),
        name=args.name,
    )


if __name__ == "__main__":
    main()
