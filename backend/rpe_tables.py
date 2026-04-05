"""
VBT-to-RPE lookup tables based on published research.

Sources:
  Squat  : González-Badillo & Sánchez-Medina (2010), Pareja-Blanco et al. (2020)
  Bench  : Zourdos et al. (2016), Jovanovic & Flanagan (2014)
  Deadlift: Ritti-Dias et al., assembled from meta-analyses (less data than squat/bench)

Values represent Mean Concentric Velocity (MCV) in m/s corresponding to each RPE.
Lower velocity → higher relative load → higher RPE.
"""

import numpy as np

# Format: [(velocity_m_s, rpe), ...] sorted ascending by velocity
VBT_TABLES: dict[str, list[tuple[float, float]]] = {
    # Derived from Helms et al. (2017) regression equations.
    # Equation rearranged: velocity = (intercept − %1RM) / slope
    # %1RM mapped to RPE: 100%=10, 97.5%=9.5, 95%=9, 92.5%=8.5,
    #                      90%=8, 87.5%=7.5, 85%=7, 82.5%=6.5, 80%=6
    # Squat: y = −0.449x + 1.096  →  x = (1.096 − y) / 0.449
    "squat": [
        (0.21, 10.0),
        (0.27,  9.5),
        (0.32,  9.0),
        (0.38,  8.5),
        (0.44,  8.0),
        (0.49,  7.5),
        (0.55,  7.0),
        (0.60,  6.5),
        (0.66,  6.0),
    ],
    # Bench: y = −0.600x + 1.051  →  x = (1.051 − y) / 0.600
    "bench": [
        (0.08, 10.0),
        (0.13,  9.5),
        (0.17,  9.0),
        (0.21,  8.5),
        (0.25,  8.0),
        (0.29,  7.5),
        (0.33,  7.0),
        (0.38,  6.5),
        (0.42,  6.0),
    ],
    # Deadlift: y = −0.600x + 1.076  →  x = (1.076 − y) / 0.600
    "deadlift": [
        (0.13, 10.0),
        (0.17,  9.5),
        (0.21,  9.0),
        (0.25,  8.5),
        (0.29,  8.0),
        (0.34,  7.5),
        (0.38,  7.0),
        (0.42,  6.5),
        (0.46,  6.0),
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
    Interpolate RPE from mean concentric velocity using VBT lookup table.

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

    table = VBT_TABLES[lift_type]
    velocities = [row[0] for row in table]
    rpes       = [row[1] for row in table]
    mcv = mean_concentric_velocity

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
