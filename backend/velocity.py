"""
Velocity calculator.

Pipeline:
  1. Extract valid (non-None) y-positions from tracking data.
  2. Convert pixel displacement → metres using plate-diameter calibration.
  3. Smooth positions with a Savitzky-Golay filter.
  4. Differentiate to get velocity (positive = bar moving up).
  5. Find the start of the eccentric phase — first sustained downward
     acceleration. Everything before this is ignored (static setup).
  6. Find the concentric phase — from the bottom to the first lockout that
     occurs after the eccentric phase ends.
  7. Compute mean concentric velocity (MCV) — averaged over the whole
     concentric phase, matching Helms et al. 2017 / GymAware ACV — and peak
     concentric velocity.
"""

import numpy as np
from scipy.signal import savgol_filter, find_peaks

from tracker import TrackingResult

_DEFAULT_PLATE_DIAMETER_M = 0.450  # fallback only; callers should pass explicitly

# Velocity must exceed this (m/s) to count as intentional movement
ECCENTRIC_THRESHOLD_M_S = 0.02   # downward
CONCENTRIC_THRESHOLD_M_S = 0.02  # upward

# Must be moving in the same direction for this many consecutive frames
# before it counts as the start of a phase
MIN_PHASE_FRAMES = 4


def _make_odd(n: int, minimum: int = 3) -> int:
    n = max(n, minimum)
    return n if n % 2 == 1 else n - 1


def _smooth(arr: np.ndarray, polyorder: int = 3) -> np.ndarray:
    window = _make_odd(min(len(arr) - 1, 15), minimum=polyorder + 2)
    if window <= polyorder or len(arr) < window:
        return arr.copy()
    return savgol_filter(arr, window_length=window, polyorder=polyorder)


def _find_rep_phases_deadlift(
    position: np.ndarray, velocity: np.ndarray
) -> tuple[int, int, int, int]:
    """
    Find phases for a deadlift.

    Unlike squat/bench, the bar starts on the floor and the concentric phase
    is the initial upward pull.  Position trace: low → high (→ low between reps).

    We find local MAXIMA (lockouts) and target the last one, then locate the
    floor position immediately before that lockout as the concentric start.

    Returns (setup_end, floor_idx, lockout_idx, num_reps).
    """
    n         = len(position)
    pos_range = float(position.max() - position.min())
    min_prom  = max(pos_range * 0.30, 0.02)
    min_dist  = max(10, n // 20)

    # Find all local maxima (lockouts) with sufficient prominence.
    tops, _ = find_peaks(position, prominence=min_prom, distance=min_dist)

    # Keep only tops in the upper 60% of the position range so small
    # oscillations near the floor (e.g. liftoff wiggles) are rejected.
    height_threshold = position.min() + pos_range * 0.60
    tops = tops[position[tops] >= height_threshold]

    if len(tops) > 0:
        num_reps = len(tops)
        top_idx  = int(tops[-1])
        print(f"[velocity] deadlift: detected {num_reps} rep(s) — using last lockout at idx {top_idx}")
    else:
        num_reps = 1
        top_idx  = int(np.argmax(position))
        print(f"[velocity] deadlift: no distinct reps — using global maximum at idx {top_idx}")

    # floor_idx: the global minimum in the segment before the last lockout.
    # This is where the bar was resting on the floor at the start of the last pull.
    floor_idx = int(np.argmin(position[: top_idx + 1]))

    # setup_end: walk left from floor_idx to find MIN_PHASE_FRAMES consecutive
    # static frames — the lifter setting up, bracing, etc.
    setup_end   = 0
    consecutive = 0
    for i in range(floor_idx - 1, -1, -1):
        if abs(velocity[i]) < ECCENTRIC_THRESHOLD_M_S:
            consecutive += 1
            if consecutive >= MIN_PHASE_FRAMES:
                setup_end = i
                break
        else:
            consecutive = 0

    lockout_idx = top_idx
    if lockout_idx <= floor_idx:
        lockout_idx = n

    return int(setup_end), int(floor_idx), int(lockout_idx), int(num_reps)


def _find_rep_phases(
    position: np.ndarray, velocity: np.ndarray
) -> tuple[int, int, int, int]:
    """
    Use position to find rep phases robustly, targeting the LAST rep.

    For multi-rep sets, the last rep is closest to failure and most relevant
    for RPE estimation. We find all local minima (rep bottoms) with sufficient
    prominence and use the last one.

    Returns (eccentric_start, bottom_idx, concentric_end, num_reps).
    """
    n = len(position)

    # Find all local minima (rep bottoms).
    # Use a higher prominence threshold (30% of range) so small pre-lift
    # bracing oscillations (~1–3 cm) are rejected while actual rep bottoms
    # (typically 40–80 cm below start) are kept.
    pos_range   = float(position.max() - position.min())
    min_prom    = max(pos_range * 0.30, 0.02)
    min_dist    = max(10, n // 20)

    bottoms, _ = find_peaks(-position, prominence=min_prom, distance=min_dist)

    # Additionally filter to bottoms that are in the lower 40% of the position
    # range — this eliminates any residual bracing dips that passed the
    # prominence check but are still far above the true squat depth.
    depth_threshold = position.min() + pos_range * 0.40
    bottoms = bottoms[position[bottoms] <= depth_threshold]

    if len(bottoms) > 0:
        num_reps   = len(bottoms)
        bottom_idx = int(bottoms[-1])   # last qualifying rep
        print(f"[velocity] detected {num_reps} rep(s) — using last bottom at idx {bottom_idx}")
    else:
        num_reps   = 1
        bottom_idx = int(np.argmin(position))   # fallback: global min
        print(f"[velocity] no distinct reps detected — using global minimum at idx {bottom_idx}")

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

    # Concentric end: the FIRST local maximum after the bottom.
    # argmax finds the global peak, which on heavy squat sets is often an
    # oscillation peak above initial lockout — not the moment the bar first
    # stopped ascending. find_peaks with a small prominence threshold finds
    # the first genuine lockout position instead.
    post_bottom = position[bottom_idx:]
    prom_lockout = max(pos_range * 0.05, 0.005)   # small — just reject noise
    peaks_local, _ = find_peaks(post_bottom, prominence=prom_lockout)

    if len(peaks_local) > 0:
        lockout_local = int(peaks_local[0])        # first peak = initial lockout
    else:
        lockout_local = int(np.argmax(post_bottom))  # fallback

    concentric_end = bottom_idx + lockout_local
    if concentric_end <= bottom_idx:
        concentric_end = n

    return int(eccentric_start), int(bottom_idx), int(concentric_end), int(num_reps)


def calculate_velocity(
    tracking: TrackingResult,
    plate_diameter_m: float = _DEFAULT_PLATE_DIAMETER_M,
    lift_type: str = "squat",
) -> dict:
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

    # ── 5. Find phases (dispatch on lift type) ───────────────────────────
    if lift_type == "deadlift":
        eccentric_start, bottom_idx, best_end, num_reps = _find_rep_phases_deadlift(y_smooth, velocity)
    else:
        eccentric_start, bottom_idx, best_end, num_reps = _find_rep_phases(y_smooth, velocity)
    best_start = bottom_idx  # concentric begins at the bottom

    # ── 5b. Trim concentric_end with a velocity-based cutoff ──────────
    # The position peak gives an upper bound, but after the velocity peak the bar
    # decelerates into lockout and then velocity stays near zero. Find the first
    # sustained drop below CONCENTRIC_THRESHOLD_M_S that occurs AFTER the velocity
    # peak and use that as the true end of the concentric phase.
    peak_vel_idx = best_start + int(np.argmax(velocity[best_start:best_end]))
    consecutive_low = 0
    for i in range(peak_vel_idx, best_end):
        if velocity[i] < CONCENTRIC_THRESHOLD_M_S:
            consecutive_low += 1
            if consecutive_low >= MIN_PHASE_FRAMES:
                best_end = max(best_start + 1, i - MIN_PHASE_FRAMES + 1)
                break
        else:
            consecutive_low = 0
    print(f"[velocity] concentric_end trimmed to {best_end} (peak_vel_idx={peak_vel_idx})")

    # ── 6. Compute MCV over the full concentric phase ─────────────────
    # MCV (ACV in Helms et al. 2017, recorded on a GymAware PowerTool) is the
    # mean velocity across the entire concentric phase — GymAware's headline
    # "Average Velocity" metric is a plain whole-ROM mean, not a windowed
    # sub-selection. An earlier version of this code averaged only a "burst
    # window" (frames >= 20% of peak velocity) to trim reversal noise at the
    # bottom, matching a misremembered/unverified claim about GymAware/PUSH
    # methodology — that inflated MCV relative to what the regression table
    # was actually calibrated against, systematically under-reporting RPE.
    conc_vel = velocity[best_start:best_end]
    conc_pos = conc_vel[conc_vel > 0]
    if len(conc_pos) == 0:
        conc_pos = np.abs(conc_vel)

    peak = float(np.max(conc_pos))
    mcv  = float(np.mean(conc_pos))

    print(f"[velocity] eccentric_start={eccentric_start}  bottom={bottom_idx}"
          f"  concentric=[{best_start}:{best_end}]"
          f"  MCV={mcv:.4f} m/s  peak={peak:.4f} m/s"
          f"  max_raw_vel={float(np.max(np.abs(velocity))):.4f} m/s")

    return {
        "time":                     time_s.tolist(),
        "velocity":                 velocity.tolist(),
        "position_m":               y_smooth.tolist(),
        "eccentric_start":          int(eccentric_start),
        "concentric_start":         int(best_start),
        "concentric_end":           int(best_end),
        "mean_concentric_velocity": round(mcv,  3),
        "peak_concentric_velocity": round(peak, 3),
        "fps":                      fps,
        "plate_diameter_px":        tracking.plate_diameter_px,
        "m_per_px":                 round(m_per_px, 6),
    }
