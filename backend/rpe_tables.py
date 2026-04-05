"""
VBT-to-RPE lookup tables derived from lift-specific peer-reviewed research.

Each lift uses its own dedicated load-velocity study rather than a single
shared regression, because the velocity profiles differ substantially between
squat, bench press, and deadlift.

Sources & methodology
─────────────────────
Squat:
  Helms et al. 2017 (J Strength Cond Res 31(2), PubMed 27243918) — competitive powerlifters.
  Regression: %1RM = −0.449 × MCV + 1.096  (r=−0.91, R²=0.83). MVT at 1RM ≈ 0.21 m/s.

Bench:
  Helms et al. 2017 (J Strength Cond Res 31(2), PubMed 27243918) — competitive powerlifters.
  Regression: %1RM = −0.600 × MCV + 1.051  (r=−0.90, R²=0.81). MVT at 1RM ≈ 0.09 m/s.

Deadlift:
  Helms et al. 2017 (J Strength Cond Res 31(2), PubMed 27243918) — competitive powerlifters.
  Regression: %1RM = −0.600 × MCV + 1.076  (r=−0.92, R²=0.85). MVT at 1RM ≈ 0.13 m/s.

%1RM → RPE mapping (standard VBT convention):
  100% = RPE 10, 97.5% = RPE 9.5, 95% = RPE 9, 92.5% = RPE 8.5,
  90% = RPE 8, 87.5% = RPE 7.5, 85% = RPE 7, 82.5% = RPE 6.5, 80% = RPE 6

Important caveats
─────────────────
- These tables apply to trained/intermediate lifters performing single reps.
- Competitive powerlifters have lower MVTs (squat ~0.23, bench ~0.10,
  deadlift ~0.14 m/s) — their RPE will be slightly underestimated by these tables.
- Individual variation of ±0.05–0.10 m/s means RPE estimates carry ~±0.5 RPE
  uncertainty at best.
- The velocity-RPE relationship is most reliable near failure (RPE ≥ 8) and
  becomes progressively less precise at lower intensities (RPE 6–7).
"""

import numpy as np

# Regression coefficients from Helms et al. 2017 for each lift.
# Equation: %1RM = slope × MCV + intercept
# Inverted for 1RM projection: 1RM = weight / (slope × MCV + intercept)
REGRESSION: dict[str, tuple[float, float]] = {
    "squat":    (-0.449, 1.096),
    "bench":    (-0.600, 1.051),
    "deadlift": (-0.600, 1.076),
}

# Format: [(velocity_m_s, rpe), ...] sorted ascending by velocity
# All velocities are Mean Concentric Velocity (MCV / MPV) in m/s.
VBT_TABLES: dict[str, list[tuple[float, float]]] = {

    # ── Squat (Helms et al. 2017 regression) ──────────────────────────────
    # Regression: %1RM = −0.449 × MCV + 1.096  (r=−0.91, R²=0.83)
    # Sample: competitive powerlifters. Source: J Strength Cond Res 31(2).
    # Velocities derived by inverting: MCV = (1.096 − %1RM) / 0.449
    "squat": [
        (0.21, 10.0),   # 100% 1RM
        (0.27,  9.5),   # 97.5%
        (0.32,  9.0),   # 95%
        (0.38,  8.5),   # 92.5%
        (0.44,  8.0),   # 90%
        (0.49,  7.5),   # 87.5%
        (0.55,  7.0),   # 85%
        (0.60,  6.5),   # 82.5%
        (0.66,  6.0),   # 80%
    ],

    # ── Bench Press (Helms et al. 2017 regression) ────────────────────────
    # Regression: %1RM = −0.600 × MCV + 1.051  (r=−0.90, R²=0.81)
    # Sample: competitive powerlifters. Source: J Strength Cond Res 31(2).
    # Velocities derived by inverting: MCV = (1.051 − %1RM) / 0.600
    "bench": [
        (0.09, 10.0),   # 100% 1RM
        (0.13,  9.5),   # 97.5%
        (0.17,  9.0),   # 95%
        (0.21,  8.5),   # 92.5%
        (0.25,  8.0),   # 90%
        (0.29,  7.5),   # 87.5%
        (0.34,  7.0),   # 85%
        (0.38,  6.5),   # 82.5%
        (0.42,  6.0),   # 80%
    ],

    # ── Deadlift (Helms et al. 2017 regression) ───────────────────────────
    # Regression: %1RM = −0.600 × MCV + 1.076  (r=−0.92, R²=0.85)
    # Sample: competitive powerlifters. Source: J Strength Cond Res 31(2).
    # Velocities derived by inverting: MCV = (1.076 − %1RM) / 0.600
    "deadlift": [
        (0.13, 10.0),   # 100% 1RM
        (0.17,  9.5),   # 97.5%
        (0.21,  9.0),   # 95%
        (0.25,  8.5),   # 92.5%
        (0.29,  8.0),   # 90%
        (0.34,  7.5),   # 87.5%
        (0.38,  7.0),   # 85%
        (0.42,  6.5),   # 82.5%
        (0.46,  6.0),   # 80%
    ],
}

RPE_DESCRIPTIONS = {
    10.0: "Max effort — no reps in reserve",
    9.5:  "Could maybe do 1 more rep",
    9.0:  "1 rep in reserve",
    8.5:  "1–2 reps in reserve",
    8.0:  "2 reps in reserve",
    7.5:  "2–3 reps in reserve",
    7.0:  "3 reps in reserve",
    6.5:  "3–4 reps in reserve",
    6.0:  "4 reps in reserve",
    5.0:  "5+ reps in reserve",
    4.0:  "Very light — 6+ reps in reserve",
    3.0:  "Warm-up weight",
}


def velocity_to_rpe(lift_type: str, mean_concentric_velocity: float) -> dict:
    """
    Interpolate RPE from mean concentric velocity using lift-specific VBT table.

    Returns:
        {
            "rpe": float (rounded to nearest 0.5),
            "description": str,
            "note": str | None   # set when velocity is outside table range
        }
    """
    lift_type = lift_type.lower()
    if lift_type not in VBT_TABLES:
        raise ValueError(f"Unknown lift type '{lift_type}'. Use: squat, bench, deadlift.")

    table      = VBT_TABLES[lift_type]
    velocities = [row[0] for row in table]
    rpes       = [row[1] for row in table]
    mcv        = mean_concentric_velocity

    note = None
    if mcv <= velocities[0]:
        rpe_raw = rpes[0]
        note = "Velocity at or below minimum threshold — RPE capped at 10"
    elif mcv >= velocities[-1]:
        rpe_raw = rpes[-1]
        note = "Velocity above table range — load is very light"
    else:
        rpe_raw = float(np.interp(mcv, velocities, rpes))

    # Round to nearest 0.5
    rpe = round(rpe_raw * 2) / 2

    description = RPE_DESCRIPTIONS.get(rpe, f"{rpe} RPE")

    return {"rpe": rpe, "description": description, "note": note}


def projected_1rm(lift_type: str, mean_concentric_velocity: float, weight: float) -> dict:
    """
    Estimate 1RM from mean concentric velocity and the weight on the bar.

    Uses the Helms et al. 2017 regression: %1RM = slope × MCV + intercept
    Rearranged: 1RM = weight / %1RM

    Args:
        lift_type: "squat", "bench", or "deadlift"
        mean_concentric_velocity: MCV in m/s
        weight: weight lifted in any unit (kg or lb) — 1RM returned in same unit

    Returns:
        {
            "projected_1rm": float,
            "percent_1rm":   float,   # 0–1 scale
            "note": str | None
        }
    """
    lift_type = lift_type.lower()
    if lift_type not in REGRESSION:
        raise ValueError(f"Unknown lift type '{lift_type}'. Use: squat, bench, deadlift.")

    slope, intercept = REGRESSION[lift_type]
    pct = slope * mean_concentric_velocity + intercept

    note = None
    if pct <= 0:
        return {"projected_1rm": None, "percent_1rm": None,
                "note": "Velocity too high to estimate 1RM from this regression"}
    if pct > 1.0:
        pct  = 1.0
        note = "Velocity at or below MVT — weight is at or above estimated 1RM"

    one_rm = round(weight / pct, 1)
    return {
        "projected_1rm": one_rm,
        "percent_1rm":   round(pct, 3),
        "note":          note,
    }
