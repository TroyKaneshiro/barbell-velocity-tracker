"""
Velocity calculator.

Pipeline:
  1. Extract valid (non-None) y-positions from tracking data.
  2. Convert pixel displacement → metres using plate-diameter calibration.
  3. Smooth positions with a Savitzky-Golay filter.
  4. Differentiate to get velocity (positive = bar moving up).
  5. Find the start of the eccentric phase — first sustained downward
     acceleration. Everything before this is ignored (static setup).
  6. Find the concentric burst — the region around the peak upward velocity
     that occurs after the eccentric phase ends.
  7. Compute mean concentric velocity (MCV) and peak concentric velocity.
"""

import numpy as np
from scipy.signal import savgol_filter

from tracker import TrackingResult

_DEFAULT_PLATE_DIAMETER_M = 0.450  # fallback only; callers should pass explicitly

# Velocity must exceed this (m/s) to count as intentional movement
ECCENTRIC_THRESHOLD_M_S = 0.02   # downward
CONCENTRIC_THRESHOLD_M_S = 0.02  # upward

# Must be moving in the same direction for this many consecutive frames
# before it counts as the start of a phase
MIN_PHASE_FRAMES = 4

# Concentric burst: include frames where velocity >= this fraction of peak
BURST_FRACTION = 0.20


def _make_odd(n: int, minimum: int = 3) -> int:
    n = max(n, minimum)
    return n if n % 2 == 1 else n - 1


def _smooth(arr: np.ndarray, polyorder: int = 3) -> np.ndarray:
    window = _make_odd(min(len(arr) - 1, 15), minimum=polyorder + 2)
    if window <= polyorder or len(arr) < window:
        return arr.copy()
    return savgol_filter(arr, window_length=window, polyorder=polyorder)


def _find_rep_phases(position: np.ndarray, velocity: np.ndarray) -> tuple[int, int, int]:
    """
    Use position to find rep phases robustly.

    The bottom of the rep is the global position minimum — the natural
    boundary between eccentric (descent) and concentric (ascent).

    Returns (eccentric_start, bottom_idx, concentric_end).
    """
    n = len(position)
    bottom_idx = int(np.argmin(position))

    # Eccentric start: walk left from bottom until we find MIN_PHASE_FRAMES
    # consecutive frames that are essentially static (bar still in setup).
    eccentric_start = 0
    consecutive = 0
    for i in range(bottom_idx - 1, -1, -1):
        if abs(velocity[i]) < ECCENTRIC_THRESHOLD_M_S:
            consecutive += 1
            if consecutive >= MIN_PHASE_FRAMES:
                eccentric_start = i
                break
        else:
            consecutive = 0

    # Concentric end: the bar reaches its highest point after the bottom.
    # Using position argmax is more robust than a velocity threshold, which
    # can fail when tracking noise keeps velocity above the threshold after
    # the bar has stopped moving at lockout.
    post_bottom    = position[bottom_idx:]
    lockout_local  = int(np.argmax(post_bottom))
    concentric_end = bottom_idx + lockout_local
    if concentric_end <= bottom_idx:
        concentric_end = n

    return int(eccentric_start), int(bottom_idx), int(concentric_end)


def calculate_velocity(tracking: TrackingResult, plate_diameter_m: float = _DEFAULT_PLATE_DIAMETER_M) -> dict:
    """
    Calculate bar velocity and identify the concentric phase.

    Args:
        tracking: result from BarTracker.process_video()
        plate_diameter_m: real-world diameter of the outer plate in metres,
                          used for pixel-to-metre calibration.

    Returns a dict ready to be serialised as JSON:
      time                    : list[float]  – seconds
      velocity                : list[float]  – m/s (positive = upward)
      position_m              : list[float]  – bar height relative to start (m)
      concentric_start        : int          – index into the above arrays
      concentric_end          : int
      mean_concentric_velocity: float
      peak_concentric_velocity: float
      fps                     : float
      plate_diameter_px       : float
      m_per_px                : float
    """
    positions = tracking.frame_positions
    fps       = tracking.fps

    # ── 1. Gather valid frames ────────────────────────────────────────
    valid_indices: list[int]   = []
    y_px:          list[float] = []

    for i, pos in enumerate(positions):
        if pos is not None:
            valid_indices.append(i)
            # Invert Y: image coords increase downward, we want up = positive
            y_px.append(-pos[1])

    if len(y_px) < 15:
        raise ValueError(
            f"Only {len(y_px)} tracked frames — not enough data to compute "
            "velocity. Try a cleaner side-view shot."
        )

    # ── 2. Calibrate pixels → metres ─────────────────────────────────
    m_per_px = plate_diameter_m / tracking.plate_diameter_px
    y_m      = np.array(y_px) * m_per_px
    y_m     -= y_m[0]
    time_s   = np.array(valid_indices) / fps

    print(f"[velocity] fps={fps:.1f}  tracked_frames={len(y_px)}"
          f"  plate_px={tracking.plate_diameter_px:.1f}  m_per_px={m_per_px:.5f}"
          f"  plate_diameter_m={plate_diameter_m}")

    # ── 3. Smooth positions ───────────────────────────────────────────
    y_smooth = _smooth(y_m, polyorder=3)

    # ── 4. Differentiate → velocity ───────────────────────────────────
    dt       = 1.0 / fps
    velocity = np.gradient(y_smooth, dt)
    velocity = _smooth(velocity, polyorder=2)

    # ── 5. Find phases using position minimum as the eccentric/concentric boundary
    eccentric_start, bottom_idx, best_end = _find_rep_phases(y_smooth, velocity)
    best_start = bottom_idx  # concentric begins at the bottom

    # ── 6. Compute MCV strictly over the concentric phase ─────────────
    conc_vel = velocity[best_start:best_end]
    conc_pos = conc_vel[conc_vel > 0]
    if len(conc_pos) == 0:
        conc_pos = np.abs(conc_vel)

    peak = float(np.max(conc_pos))
    burst_threshold = BURST_FRACTION * peak

    # Trim near-zero frames at reversal and lockout that drag down the mean.
    # Only average frames where velocity >= BURST_FRACTION * peak — this
    # matches the effective "bar is actually moving" window and aligns MCV
    # with research measurements from dedicated VBT devices.
    in_burst = (velocity[best_start:best_end] >= burst_threshold)
    burst_local = np.where(in_burst)[0]
    if len(burst_local) > 0:
        burst_start = int(best_start + burst_local[0])
        burst_end   = int(best_start + burst_local[-1]) + 1
        burst       = velocity[burst_start:burst_end]
        burst       = burst[burst > 0]
    else:
        burst_start = best_start
        burst_end   = best_end
        burst       = conc_pos

    if len(burst) == 0:
        burst = conc_pos
    mcv = float(np.mean(burst))

    print(f"[velocity] eccentric_start={eccentric_start}  bottom={bottom_idx}"
          f"  concentric=[{best_start}:{best_end}]  burst=[{burst_start}:{burst_end}]"
          f"  MCV={mcv:.4f} m/s  peak={peak:.4f} m/s"
          f"  max_raw_vel={float(np.max(np.abs(velocity))):.4f} m/s")

    return {
        "time":                     time_s.tolist(),
        "velocity":                 velocity.tolist(),
        "position_m":               y_smooth.tolist(),
        "eccentric_start":          int(eccentric_start),
        "concentric_start":         int(best_start),
        "concentric_end":           int(best_end),
        "burst_start":              int(burst_start),
        "burst_end":                int(burst_end),
        "burst_threshold":          round(burst_threshold, 4),
        "mean_concentric_velocity": round(mcv,  3),
        "peak_concentric_velocity": round(peak, 3),
        "fps":                      fps,
        "plate_diameter_px":        tracking.plate_diameter_px,
        "m_per_px":                 round(m_per_px, 6),
    }
