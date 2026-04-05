"""
Barbell plate tracker.

Detection: Hough circle transform run on every frame.
           Frames where detection fails are filled by linear interpolation
           from neighbouring successful detections.

Calibration: The detected plate diameter in pixels is compared against the
             user-supplied real-world plate diameter to give m/px.

This replaces the CSRT-based approach which drifted during dynamic movement.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrackingResult:
    frame_positions: list[Optional[tuple[float, float]]]  # (x_px, y_px) per frame
    fps: float
    plate_diameter_px: float   # median over all successful detections
    total_frames: int
    width: int
    height: int


class BarTracker:
    MAX_DETECTION_FRAMES = 60  # used only for the initial calibration scan

    def _detect_plate(
        self,
        frame: np.ndarray,
        hint_r: Optional[float] = None,
    ) -> Optional[tuple[float, float, float]]:
        """
        Run Hough circle detection on a single frame.

        hint_r: expected radius from a previous detection. When supplied the
                radius search window is tightened, improving per-frame speed
                and reducing false positives during tracking.

        Returns (cx, cy, radius) in pixels, or None.
        """
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        h, w    = gray.shape
        short   = min(w, h)

        if hint_r is not None:
            min_r = max(5, int(hint_r * 0.75))
            max_r = int(hint_r * 1.25)
        else:
            min_r = max(12, int(short * 0.05))
            max_r = int(short * 0.45)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=min_r * 2,
            param1=50,
            param2=28,
            minRadius=min_r,
            maxRadius=max_r,
        )

        if circles is None:
            return None

        circles = circles[0]
        # Pick the largest circle — most likely the outermost plate
        best = max(circles, key=lambda c: c[2])
        return (float(best[0]), float(best[1]), float(best[2]))

    def _calibrate(self, cap: cv2.VideoCapture) -> Optional[float]:
        """
        Scan the first MAX_DETECTION_FRAMES frames to find a reliable
        plate radius for the hint used during full tracking.

        Returns median radius in pixels, or None if nothing found.
        """
        radii = []
        for _ in range(self.MAX_DETECTION_FRAMES):
            ret, frame = cap.read()
            if not ret:
                break
            result = self._detect_plate(frame)
            if result is not None:
                radii.append(result[2])
            if len(radii) >= 10:
                break

        if not radii:
            return None
        return float(np.median(radii))

    def process_video(self, video_path: str) -> TrackingResult:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        try:
            return self._process(cap)
        finally:
            cap.release()

    def _process(self, cap: cv2.VideoCapture) -> TrackingResult:
        fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # ── Phase 1: calibrate radius hint ───────────────────────────
        hint_r = self._calibrate(cap)
        if hint_r is None:
            raise ValueError(
                "Could not detect a barbell plate in the first "
                f"{self.MAX_DETECTION_FRAMES} frames. "
                "Ensure the plate is clearly visible, well-lit, and the "
                "camera has a side-on view."
            )

        print(f"[tracker] calibrated radius hint: {hint_r:.1f} px")

        # ── Phase 2: detect on every frame ───────────────────────────
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        raw: list[Optional[tuple[float, float, float]]] = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result = self._detect_plate(frame, hint_r=hint_r)
            raw.append(result)

        if not raw:
            raise ValueError("No frames could be read from the video.")

        n_detected = sum(1 for r in raw if r is not None)
        print(f"[tracker] detected plate in {n_detected}/{len(raw)} frames")

        if n_detected < 5:
            raise ValueError(
                f"Plate detected in only {n_detected} frames — not enough for "
                "velocity analysis. Try better lighting or a clearer side-on view."
            )

        # ── Phase 3: reject implausible jumps ────────────────────────
        # A real barbell plate cannot teleport. If a detection jumps more
        # than max_jump_px from the previous valid detection it is a false
        # positive (different circle) and should be treated as a miss.
        # Allow ~15% of frame height per frame as the maximum plausible move.
        max_jump_px = height * 0.15

        cleaned: list[Optional[tuple[float, float, float]]] = []
        last_valid: Optional[tuple[float, float, float]] = None

        for r in raw:
            if r is None:
                cleaned.append(None)
                continue
            if last_valid is None:
                cleaned.append(r)
                last_valid = r
                continue
            dx = r[0] - last_valid[0]
            dy = r[1] - last_valid[1]
            dist = (dx**2 + dy**2) ** 0.5
            if dist > max_jump_px:
                cleaned.append(None)   # treat as missed — will be interpolated
            else:
                cleaned.append(r)
                last_valid = r

        raw = cleaned
        n_after = sum(1 for r in raw if r is not None)
        print(f"[tracker] after jump filter: {n_after}/{len(raw)} frames kept")

        # ── Phase 4: interpolate missing frames ───────────────────────
        # Build arrays of x, y, r with NaN for missed frames
        xs = np.array([r[0] if r else np.nan for r in raw])
        ys = np.array([r[1] if r else np.nan for r in raw])
        rs = np.array([r[2] if r else np.nan for r in raw])

        idx = np.arange(len(raw))
        valid = ~np.isnan(xs)

        if valid.sum() < 2:
            raise ValueError("Too few detections to interpolate.")

        xs_interp = np.interp(idx, idx[valid], xs[valid])
        ys_interp = np.interp(idx, idx[valid], ys[valid])

        # Use median radius from valid frames for calibration
        plate_diameter_px = float(np.nanmedian(rs)) * 2

        positions: list[Optional[tuple[float, float]]] = [
            (float(xs_interp[i]), float(ys_interp[i]))
            for i in range(len(raw))
        ]

        print(f"[tracker] plate_diameter_px={plate_diameter_px:.1f}")

        return TrackingResult(
            frame_positions=positions,
            fps=fps,
            plate_diameter_px=plate_diameter_px,
            total_frames=len(raw),
            width=width,
            height=height,
        )
