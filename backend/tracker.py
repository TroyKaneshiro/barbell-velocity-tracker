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
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── CSRT re-init on failure ───────────────────────────────────────────────────
MAX_CSRT_FAILURES  = 5     # consecutive failures before re-init
MAX_FRAME_JUMP     = 2.0   # bbox centre may not move more than this × plate_r per frame
# ── Localised Hough around click ─────────────────────────────────────────────
CLICK_SEARCH_FACTOR = 0.20  # search region half-size = this × min(frame_w, frame_h)
MIN_EDGE_RATIO      = 0.18
# ── YOLO detection ───────────────────────────────────────────────────────────
YOLO_MODEL_PATH   = Path(os.environ.get('YOLO_MODEL_PATH', Path(__file__).parent / 'yolo_plate.onnx'))
YOLO_INPUT_SIZE   = (640, 640)
YOLO_CONF_THRESH  = 0.30
YOLO_NMS_THRESH   = 0.40
# ── Speed: downscale + ROI tracking ──────────────────────────────────────────
TRACK_SCALE = 0.50   # resize each frame to this fraction before CSRT (~4× fewer pixels at 1080p)
ROI_FACTOR  = 3.5    # ROI half-width = plate_r_scaled × ROI_FACTOR

_yolo_net = None
_yolo_net_missing = False


def _get_yolo_net():
    """Load the YOLO ONNX model once per process and cache it at module scope."""
    global _yolo_net, _yolo_net_missing
    if _yolo_net is not None:
        return _yolo_net
    if _yolo_net_missing:
        raise FileNotFoundError(f"YOLO model not found at {YOLO_MODEL_PATH}")

    if not YOLO_MODEL_PATH.exists():
        _yolo_net_missing = True
        raise FileNotFoundError(
            f"YOLO model not found at {YOLO_MODEL_PATH}. "
            "Set YOLO_MODEL_PATH or place a model at this path."
        )

    net = cv2.dnn.readNet(str(YOLO_MODEL_PATH))
    if hasattr(cv2.dnn, 'DNN_BACKEND_OPENCV'):
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    if hasattr(cv2.dnn, 'DNN_TARGET_CPU'):
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    _yolo_net = net
    return _yolo_net


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
        max_r = int(short * 0.52)

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

        # Discard circles whose centre is too far from the click — the user clicked the
        # plate centre, so a valid detection should be near that point.
        candidates = [c for c in candidates if c[3] <= half * 0.50]

        if not candidates:
            return None

        # Pick the largest circle — the outer rim is always the biggest candidate.
        # Proximity to click is used only to break ties within 10% of the max radius,
        # so a slightly off-centre click still returns the outer edge, not an inner bezel.
        max_r = max(c[2] for c in candidates)
        best  = min(candidates, key=lambda c: (c[2] < max_r * 0.90, c[3]))
        return best[0], best[1], best[2]

    def _letterbox(self, frame: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        input_w, input_h = YOLO_INPUT_SIZE
        h, w = frame.shape[:2]
        scale = min(input_w / w, input_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))
        letterbox = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
        dx = (input_w - new_w) // 2
        dy = (input_h - new_h) // 2
        letterbox[dy:dy + new_h, dx:dx + new_w] = resized
        return letterbox, scale, dx, dy

    def _detect_plate_yolo(
        self,
        frame: np.ndarray,
        click_x: float,
        click_y: float,
        frame_w: int,
        frame_h: int,
    ) -> Optional[tuple[float, float, float]]:
        """Detect the plate with a YOLO model and convert the box to a circle."""
        net = _get_yolo_net()
        letterbox, ratio, dx, dy = self._letterbox(frame)
        blob = cv2.dnn.blobFromImage(letterbox, 1/255.0, YOLO_INPUT_SIZE, swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward()

        if isinstance(outputs, list):
            outputs = np.vstack([o.reshape(-1, o.shape[-1]) for o in outputs])
        elif outputs.ndim == 3 and outputs.shape[0] == 1:
            outputs = outputs[0]

        # Ultralytics YOLO11 ONNX exports are channels-first: (4+nc, num_anchors),
        # e.g. (5, 8400) for a single class. Transpose to (num_anchors, 4+nc) so
        # each row is one detection. 4+nc is always far smaller than the anchor count.
        if outputs.ndim == 2 and outputs.shape[0] < outputs.shape[1] and outputs.shape[0] < 10:
            outputs = outputs.T

        if outputs.ndim != 2 or outputs.shape[1] < 5:
            return None

        boxes = []
        scores = []

        for row in outputs.reshape(-1, outputs.shape[-1]):
            confidence = float(row[4])
            if confidence < YOLO_CONF_THRESH:
                continue

            x_c, y_c, w_c, h_c = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            if x_c <= 1 and y_c <= 1 and w_c <= 1 and h_c <= 1:
                x_c *= YOLO_INPUT_SIZE[0]
                y_c *= YOLO_INPUT_SIZE[1]
                w_c *= YOLO_INPUT_SIZE[0]
                h_c *= YOLO_INPUT_SIZE[1]

            if row.shape[0] > 5:
                class_scores = row[5:]
                class_id = int(np.argmax(class_scores))
                class_score = float(class_scores[class_id])
                confidence *= class_score
            else:
                class_id = 0

            if confidence < YOLO_CONF_THRESH:
                continue

            x1 = x_c - w_c / 2.0
            y1 = y_c - h_c / 2.0
            boxes.append([x1, y1, w_c, h_c])
            scores.append(confidence)

        if not boxes:
            return None

        indices = cv2.dnn.NMSBoxes(boxes, scores, YOLO_CONF_THRESH, YOLO_NMS_THRESH)
        if len(indices) == 0:
            return None

        indices = indices.flatten() if isinstance(indices, np.ndarray) else [int(i[0]) for i in indices]
        best_idx = indices[0]

        if click_x is not None and click_y is not None:
            click_px = click_x * frame_w
            click_py = click_y * frame_h
            best_score = float('-inf')
            best_idx = None
            for idx in indices:
                x1, y1, w_c, h_c = boxes[idx]
                cx = x1 + w_c / 2.0
                cy = y1 + h_c / 2.0
                cx_orig = (cx - dx) / ratio
                cy_orig = (cy - dy) / ratio
                dist = ((cx_orig - click_px) ** 2 + (cy_orig - click_py) ** 2) ** 0.5
                score = scores[idx] - dist * 0.001
                if best_idx is None or score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx is None:
                best_idx = indices[0]

        x1, y1, w_c, h_c = boxes[best_idx]
        cx = x1 + w_c / 2.0
        cy = y1 + h_c / 2.0
        cx = (cx - dx) / ratio
        cy = (cy - dy) / ratio
        w_orig = w_c / ratio
        h_orig = h_c / ratio

        cx = float(np.clip(cx, 0, frame_w - 1))
        cy = float(np.clip(cy, 0, frame_h - 1))
        r = float(max(w_orig, h_orig) / 2.0)
        return cx, cy, r

    def _detect_plate(
        self,
        frame: np.ndarray,
        click_x: float,
        click_y: float,
        frame_w: int,
        frame_h: int,
    ) -> Optional[tuple[float, float, float]]:
        """Try YOLO detection first, and fall back to Hough if necessary."""
        try:
            circle = self._detect_plate_yolo(frame, click_x, click_y, frame_w, frame_h)
            if circle is not None:
                print(f"[tracker] plate detected by YOLO: cx={circle[0]:.1f} cy={circle[1]:.1f} r={circle[2]:.1f}")
                return circle
        except FileNotFoundError as exc:
            print(f"[tracker] YOLO model missing: {exc}")
        except Exception as exc:
            print(f"[tracker] YOLO detection failed: {exc}")

        return self._find_circle_near_click(frame, click_x, click_y, frame_w, frame_h)

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
            circle = self._detect_plate(first_frame, click_x, click_y, width, height)
            if circle is not None:
                cx0, cy0, plate_r = circle
                print(f"[tracker] plate detected near click: cx={cx0:.1f} cy={cy0:.1f} r={plate_r:.1f}")
            else:
                plate_r = min(width, height) * CLICK_SEARCH_FACTOR * 0.5
                print(f"[tracker] no plate detection — using click point directly: "
                      f"cx={cx0:.1f} cy={cy0:.1f} r={plate_r:.1f}")

        # ── Scaled tracking dimensions ────────────────────────────────────────
        # CSRT runs on a downscaled + cropped region; coordinates are converted
        # back to original pixels before being stored.
        sw = max(int(width  * TRACK_SCALE), 320)   # never shrink below 320 px wide
        sh = max(int(height * TRACK_SCALE), 240)
        sx = sw / width    # actual horizontal scale (may differ from TRACK_SCALE if clamped)
        sy = sh / height

        # Plate centre and radius in scaled coords
        cx0s      = cx0    * sx
        cy0s      = cy0    * sy
        plate_r_s = plate_r * (sx + sy) / 2.0

        # ROI half-width (in scaled pixels) — region fed to CSRT each frame
        roi_half  = max(int(plate_r_s * ROI_FACTOR), 60)
        box_half_s = int(plate_r_s * 1.3)

        def _roi(small_fr: np.ndarray, cx_s: float, cy_s: float):
            """Crop a fixed-size window around (cx_s, cy_s) in the scaled frame.
            Returns (crop, origin_x, origin_y) in scaled-frame coords."""
            fh, fw = small_fr.shape[:2]
            rx  = max(0, int(cx_s) - roi_half)
            ry  = max(0, int(cy_s) - roi_half)
            rx2 = min(fw, rx + roi_half * 2)
            ry2 = min(fh, ry + roi_half * 2)
            return small_fr[ry:ry2, rx:rx2], rx, ry

        # Initialise CSRT on a scaled+cropped first frame
        small_first          = cv2.resize(first_frame, (sw, sh))
        init_crop, irx, iry  = _roi(small_first, cx0s, cy0s)
        init_bbox_s = (
            max(0, int(cx0s - irx) - box_half_s),
            max(0, int(cy0s - iry) - box_half_s),
            box_half_s * 2,
            box_half_s * 2,
        )

        tracker = cv2.TrackerCSRT_create()
        tracker.init(init_crop, init_bbox_s)
        print(f"[tracker] CSRT initialised on {init_crop.shape[1]}×{init_crop.shape[0]} crop "
              f"(frame scaled {width}×{height} → {sw}×{sh})  bbox={init_bbox_s}")

        # init_bbox in original coords — used only to seed last_good_bbox for debug overlay
        box_half = int(plate_r * 1.3)
        init_bbox_orig = (
            max(0, int(cx0) - box_half),
            max(0, int(cy0) - box_half),
            min(width  - max(0, int(cx0) - box_half), box_half * 2),
            min(height - max(0, int(cy0) - box_half), box_half * 2),
        )

        dbg_writer = None
        if debug_video_path:
            dbg_writer = self._make_writer(debug_video_path, fps, width, height)

        positions:       list[Optional[tuple[float, float]]] = []
        last_good_bbox   = init_bbox_orig
        last_cx_s        = cx0s   # running scaled-coord estimate of plate centre
        last_cy_s        = cy0s
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
                small            = cv2.resize(frame, (sw, sh))
                crop, rx, ry     = _roi(small, last_cx_s, last_cy_s)
                ok, bbox_c       = tracker.update(crop)

                if ok:
                    bx_c, by_c, bw_c, bh_c = [int(v) for v in bbox_c]
                    # Unproject: crop → scaled frame → original frame
                    cx_s  = bx_c + bw_c / 2.0 + rx
                    cy_s  = by_c + bh_c / 2.0 + ry
                    cx_new = cx_s / sx
                    cy_new = cy_s / sy

                    # Reject if centre jumped too far — CSRT drifted to wrong target
                    prev = positions[-1] if positions else None
                    if prev is not None:
                        jump = ((cx_new - prev[0]) ** 2 + (cy_new - prev[1]) ** 2) ** 0.5
                        if jump > MAX_FRAME_JUMP * plate_r:
                            ok = False

                if ok:
                    positions.append((cx_new, cy_new))
                    last_cx_s        = cx_s
                    last_cy_s        = cy_s
                    last_good_bbox   = (
                        int(cx_new) - box_half,
                        int(cy_new) - box_half,
                        box_half * 2,
                        box_half * 2,
                    )
                    consecutive_fail = 0
                else:
                    consecutive_fail += 1
                    positions.append(None)

                    # Re-init CSRT at the original seed position so it snaps back to the plate
                    if consecutive_fail >= MAX_CSRT_FAILURES:
                        reinit_crop, _, _ = _roi(small, cx0s, cy0s)
                        tracker = cv2.TrackerCSRT_create()
                        tracker.init(reinit_crop, init_bbox_s)
                        last_cx_s        = cx0s
                        last_cy_s        = cy0s
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
