"""
VBT-to-RPE lookup tables derived from lift-specific peer-reviewed research.

Each lift uses its own dedicated load-velocity study rather than a single
shared regression, because the velocity profiles differ substantially between
squat, bench press, and deadlift.

Sources & methodology
─────────────────────
Squat:
  Sanchez-Medina et al. (PMC6226068) — 80 strength-trained males.
  Full back squat MPV at each %1RM. MVT at 1RM = 0.32 m/s.
  Regression: Load (%1RM) = –5.961 × MPV² – 50.71 × MPV + 117.0  (R²=0.954)

Bench:
  González-Badillo & Sánchez-Medina 2010 (PubMed 20180176) — 120 trained males.
  Bench press MPV at each %1RM. MVT at 1RM = 0.16 m/s.
  Cross-checked against Balsalobre-Fernandez 2022 (PMC9180020) recreational
  athlete sample (men: MVT = 0.21 m/s).
  Final values use midpoints between the two populations.

Deadlift:
  Benavides-Ubric et al. (PMC7429441) — trained subjects.
  Deadlift MV at each %1RM. MVT at 1RM = 0.33 m/s.

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

# Format: [(velocity_m_s, rpe), ...] sorted ascending by velocity
# All velocities are Mean Concentric Velocity (MCV / MPV) in m/s.
VBT_TABLES: dict[str, list[tuple[float, float]]] = {

    # ── Squat (Sanchez-Medina et al.) ─────────────────────────────────────
    # MPV at each %1RM read directly from the study data table:
    #   100%=0.32, 90%=0.51, 80%=0.68, 70%=0.84, 60%=1.00, 50%=1.14
    # Half-step RPE values interpolated linearly between study points.
    "squat": [
        (0.32, 10.0),   # 100% 1RM
        (0.40,  9.5),   # ~97.5%
        (0.51,  9.0),   # 90%
        (0.59,  8.5),   # ~87.5%
        (0.68,  8.0),   # 80%
        (0.76,  7.5),   # ~77.5%
        (0.84,  7.0),   # 70%
        (0.92,  6.5),   # ~65%
        (1.00,  6.0),   # 60%
    ],

    # ── Bench Press (González-Badillo 2010 + Balsalobre 2022 midpoints) ───
    # González-Badillo MVT at 1RM = 0.16 m/s; Balsalobre male MVT = 0.21 m/s.
    # Midpoint MVT used = ~0.19 m/s, scaled proportionally across the range.
    # Study velocities at key %1RM:
    #   González-Badillo: 95%≈0.26, 85%≈0.45, 75%≈0.62, 65%≈0.77, 55%≈0.89
    #   Balsalobre:       80%≈0.42, 70%≈0.57, 60%≈0.70, 50%≈0.92
    # Values below use midpoints between the two sources.
    "bench": [
        (0.17, 10.0),   # 100% 1RM  (midpoint of 0.10–0.21 MVTs, biased toward general)
        (0.22,  9.5),   # ~97.5%
        (0.28,  9.0),   # ~95%
        (0.34,  8.5),   # ~92.5%
        (0.42,  8.0),   # ~87.5% / 80%
        (0.50,  7.5),   # ~82.5%
        (0.57,  7.0),   # ~75% / 70%
        (0.64,  6.5),   # ~67.5%
        (0.70,  6.0),   # ~65% / 60%
    ],

    # ── Deadlift (Benavides-Ubric et al.) ─────────────────────────────────
    # MV at each %1RM from study data table:
    #   100%=0.33, 90%=0.45, 80%=0.57, 70%=0.68, 60%=0.80, 50%=0.91
    # Half-step RPE values interpolated linearly between study points.
    "deadlift": [
        (0.33, 10.0),   # 100% 1RM
        (0.39,  9.5),   # ~97.5%
        (0.45,  9.0),   # 90%
        (0.51,  8.5),   # ~87.5%
        (0.57,  8.0),   # 80%
        (0.63,  7.5),   # ~77.5%
        (0.68,  7.0),   # 70%
        (0.74,  6.5),   # ~65%
        (0.80,  6.0),   # 60%
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
