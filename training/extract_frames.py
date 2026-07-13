"""
Extract frames from a lift video for labeling.

Usage:
    python extract_frames.py path/to/video.mp4 [--fps 2] [--out dataset/raw]

Run this over several source videos (different plates/lighting/gyms) to
build labeling variety — one video alone won't give the detector enough
to generalize from.
"""
import argparse
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Path to a source lift video")
    parser.add_argument("--fps", type=float, default=2.0, help="Frames to extract per second")
    parser.add_argument("--out", default="dataset/raw", help="Output directory for extracted frames")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, round(video_fps / args.fps))

    stem = Path(args.video).stem
    idx, saved = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            out_path = out_dir / f"{stem}_{saved:04d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1
        idx += 1
    cap.release()

    print(f"Saved {saved} frames to {out_dir}")


if __name__ == "__main__":
    main()
