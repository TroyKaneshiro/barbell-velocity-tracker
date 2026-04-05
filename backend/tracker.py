"""
Barbell plate tracker — user-click seeded CSRT tracking.

Pipeline
  1. The user clicks on the weight plate in the first video frame.
     Normalised click coordinates (0-1) are passed in.
  2. A localised Hough search around the click finds the exact plate circle.
     If Hough finds nothing, a default box sized to the click region is used.
  3. CSRT tracker is initialised on that bounding box and runs frame-by-frame.
  4. On CSRT failure the tracker is re-initialised from the last good bbox.

Debug video: MJPEG .avi converted to H.264 .mp4 by main.py.
  Shows: green box = CSRT bbox, crosshair = plate centre, confidence label.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# ── CSRT re-init on failure ───────────────────────────────────────────────────
MAX_CSRT_FAILURES  = 5     # consecutive failures before re-init
MAX_FRAME_JUMP     = 2.0   # bbox centre may not move more than this × plate_r per frame
# ── Localised Hough around click ─────────────────────────────────────────────
CLICK_SEARCH_FACTOR = 0.20  # search region half-size = this × min(frame_w, frame_h)
MIN_EDGE_RATIO      = 0.18


@dataclass
class TrackingResult:
    frame_positions: list[Optional[tuple[float, float]]]
    fps: float
    plate_diameter_px: float
    total_frames: int
    width: int
    height: int
    debug_video_path: Optional[str] = field(default=None)


class BarTracker:

    def __init__(self, lift_type: str = "squat"):
        self.lift_type = lift_type.lower()

    # ── Edge strength ─────────────────────────────────────────────────────────

    def _edge_strength(
        self,
        edges: np.ndarray,
        cx_c: float, cy_c: float, r_c: float,
        n_samples: int = 36,
    ) -> float:
        h, w   = edges.shape
        angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
        xs = np.round(cx_c + r_c * np.cos(angles)).astype(int)
        ys = np.round(cy_c + r_c * np.sin(angles)).astype(int)
        mask = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        if mask.sum() == 0:
            return 0.0
        return float(edges[ys[mask], xs[mask]].astype(bool).mean())

    # ── Localised Hough around click ──────────────────────────────────────────

    def _find_circle_near_click(
        self,
        frame: np.ndarray,
        click_x: float,
        click_y: float,
        frame_w: int,
        frame_h: int,
    ) -> Optional[tuple[float, float, float]]:
        """
        Run a tight Hough search in a region centred on the user's click.
        Returns full-frame (cx, cy, r) for the best circle found, or None.
        """
        half   = int(min(frame_w, frame_h) * CLICK_SEARCH_FACTOR)
        cx_px  = int(click_x * frame_w)
        cy_px  = int(click_y * frame_h)

        rx = max(0, cx_px - half)
        ry = max(0, cy_px - half)
        rw = min(frame_w - rx, half * 2)
        rh = min(frame_h - ry, half * 2)

        crop    = frame[ry:ry + rh, rx:rx + rw]
        gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        edges   = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 1), 30, 90)

        short = min(rw, rh)
        min_r = max(6, int(short * 0.10))
        max_r = int(short * 0.90)

        raw = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1,
            minDist=min_r, param1=50, param2=20,
            minRadius=min_r, maxRadius=max_r,
        )
        if raw is None:
            return None

        candidates = []
        for c in raw[0]:
            cx_c, cy_c, r_c = float(c[0]), float(c[1]), float(c[2])
            if self._edge_strength(edges, cx_c, cy_c, r_c) < MIN_EDGE_RATIO:
                continue
            # Distance from click in crop coordinates
            dist = ((cx_c - (cx_px - rx)) ** 2 + (cy_c - (cy_px - ry)) ** 2) ** 0.5
            candidates.append((cx_c + rx, cy_c + ry, r_c, dist))

        if not candidates:
            return None

        # Pick the largest circle — the outer rim is always the biggest candidate.
        # Proximity to click is used only to break ties within 10% of the max radius,
        # so a slightly off-centre click still returns the outer edge, not an inner bezel.
        max_r = max(c[2] for c in candidates)
        best  = min(candidates, key=lambda c: (c[2] < max_r * 0.90, c[3]))
        return best[0], best[1], best[2]

    # ── Writer helper ─────────────────────────────────────────────────────────

    def _make_writer(self, path: str, fps: float, w: int, h: int):
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        return writer if writer.isOpened() else None

    # ── Public entry point ────────────────────────────────────────────────────

    def process_video(
        self,
        video_path: str,
        debug_video_path: Optional[str] = None,
        click_x: Optional[float] = None,
        click_y: Optional[float] = None,
        plate_r_norm: Optional[float] = None,
    ) -> TrackingResult:
        if click_x is None or click_y is None:
            raise ValueError("Click coordinates are required. Please click on the weight plate.")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        try:
            return self._process(cap, debug_video_path, click_x, click_y, plate_r_norm)
        finally:
            cap.release()

    # ── Main processing ───────────────────────────────────────────────────────

    def _process(
        self,
        cap: cv2.VideoCapture,
        debug_video_path: Optional[str],
        click_x: float,
        click_y: float,
        plate_r_norm: Optional[float] = None,
    ) -> TrackingResult:
        fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # ── Read first frame and find plate circle near click ─────────────
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, first_frame = cap.read()
        if not ret:
            raise ValueError("Could not read first frame.")

        cx0 = click_x * width
        cy0 = click_y * height

        if plate_r_norm is not None:
            # User manually defined the radius via right-click — use it directly
            plate_r = plate_r_norm * min(width, height)
            print(f"[tracker] manual circle: cx={cx0:.1f} cy={cy0:.1f} r={plate_r:.1f}")
        else:
            circle = self._find_circle_near_click(first_frame, click_x, click_y, width, height)
            if circle is not None:
                cx0, cy0, plate_r = circle
                print(f"[tracker] plate found near click: cx={cx0:.1f} cy={cy0:.1f} r={plate_r:.1f}")
            else:
                plate_r = min(width, height) * CLICK_SEARCH_FACTOR * 0.5
                print(f"[tracker] no circle near click — using click point directly: "
                      f"cx={cx0:.1f} cy={cy0:.1f} r={plate_r:.1f}")

        # CSRT bounding box — slightly larger than the plate for texture context
        box_half = int(plate_r * 1.3)
        init_bbox = (
            max(0, int(cx0) - box_half),
            max(0, int(cy0) - box_half),
            min(width  - max(0, int(cx0) - box_half), box_half * 2),
            min(height - max(0, int(cy0) - box_half), box_half * 2),
        )

        tracker = cv2.TrackerCSRT_create()
        tracker.init(first_frame, init_bbox)
        print(f"[tracker] CSRT initialised with bbox {init_bbox}")

        dbg_writer = None
        if debug_video_path:
            dbg_writer = self._make_writer(debug_video_path, fps, width, height)

        positions:       list[Optional[tuple[float, float]]] = []
        last_good_bbox   = init_bbox
        consecutive_fail = 0

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx == 0:
                # First frame — seeded from click
                positions.append((cx0, cy0))
                consecutive_fail = 0
            else:
                ok, bbox = tracker.update(frame)

                if ok:
                    bx, by, bw, bh = [int(v) for v in bbox]
                    cx_new = bx + bw / 2.0
                    cy_new = by + bh / 2.0

                    # Reject if bbox centre jumped too far — CSRT drifted to wrong target
                    prev = positions[-1] if positions else None
                    if prev is not None:
                        jump = ((cx_new - prev[0]) ** 2 + (cy_new - prev[1]) ** 2) ** 0.5
                        if jump > MAX_FRAME_JUMP * plate_r:
                            ok = False

                if ok:
                    positions.append((cx_new, cy_new))
                    last_good_bbox   = (bx, by, bw, bh)
                    consecutive_fail = 0
                else:
                    consecutive_fail += 1
                    positions.append(None)

                    # Re-init CSRT from the original seed bbox so it snaps back to the plate
                    if consecutive_fail >= MAX_CSRT_FAILURES:
                        tracker = cv2.TrackerCSRT_create()
                        tracker.init(frame, init_bbox)
                        consecutive_fail = 0
                        print(f"[tracker] CSRT re-init at frame {frame_idx}")

            # ── Debug overlay ─────────────────────────────────────────────
            if dbg_writer is not None:
                dbg = frame.copy()
                p   = positions[-1]

                if p is not None:
                    bx, by, bw, bh = last_good_bbox
                    cv2.rectangle(dbg, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                    cv2.circle(dbg,
                               (int(p[0]), int(p[1])), max(1, int(plate_r)),
                               (0, 255, 0), 2)
                    cv2.drawMarker(dbg, (int(p[0]), int(p[1])),
                                   (0, 255, 255), cv2.MARKER_CROSS, 14, 2)
                else:
                    cv2.putText(dbg, "LOST", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

                if frame_idx == 0:
                    # Draw click point
                    cv2.drawMarker(dbg,
                                   (int(click_x * width), int(click_y * height)),
                                   (0, 220, 255), cv2.MARKER_TILTED_CROSS, 20, 2)
                    cv2.putText(dbg, "SEED", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

                cv2.putText(dbg, f"f{frame_idx}", (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
                dbg_writer.write(dbg)

        if dbg_writer is not None:
            dbg_writer.release()

        # ── Interpolate missing positions ──────────────────────────────────
        xs      = np.array([p[0] if p else np.nan for p in positions])
        ys      = np.array([p[1] if p else np.nan for p in positions])
        idx_arr = np.arange(len(positions))
        valid   = ~np.isnan(xs)

        if valid.sum() < 5:
            raise ValueError(
                f"Only {int(valid.sum())} frames tracked. "
                "Ensure the full body and plate are clearly visible."
            )

        xs = np.interp(idx_arr, idx_arr[valid], xs[valid])
        ys = np.interp(idx_arr, idx_arr[valid], ys[valid])

        plate_diameter_px = plate_r * 2
        tracked = int(valid.sum())
        print(f"[tracker] tracked={tracked}/{len(positions)} "
              f"({100 * tracked // len(positions)}%)  "
              f"plate_diameter_px={plate_diameter_px:.1f}")

        return TrackingResult(
            frame_positions=[(float(xs[i]), float(ys[i]))
                             for i in range(len(positions))],
            fps=fps,
            plate_diameter_px=plate_diameter_px,
            total_frames=len(positions),
            width=width,
            height=height,
            debug_video_path=debug_video_path,
        )
