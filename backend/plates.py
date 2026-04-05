"""
Plate diameter lookup table.

Diameters are in metres. Values are standard/typical for each weight class.
45 lb / 20 kg are fully standardised at 450 mm; lighter plates vary more
between manufacturers, so the values below are common averages.

Future improvement: per-brand exact specifications.
"""

# key: display label shown in the UI
# value: diameter in metres used for pixel-to-metre calibration
PLATE_DIAMETERS: dict[str, float] = {
    # ── lb plates ──────────────────────────────────────────────
    "45 lb":  0.450,   # standardised Olympic diameter
    "35 lb":  0.420,
    "25 lb":  0.380,
    "10 lb":  0.280,
    "5 lb":   0.220,
    "2.5 lb": 0.178,
    # ── kg plates ──────────────────────────────────────────────
    "25 kg":  0.450,   # same outer diameter as 45 lb
    "20 kg":  0.450,   # standardised Olympic diameter
    "15 kg":  0.400,
    "10 kg":  0.320,
    "5 kg":   0.240,
    "2.5 kg": 0.190,
}

DEFAULT_PLATE = "45 lb"


def get_diameter(plate_label: str) -> float:
    """Return plate diameter in metres. Raises ValueError if label unknown."""
    try:
        return PLATE_DIAMETERS[plate_label]
    except KeyError:
        valid = ", ".join(f'"{k}"' for k in PLATE_DIAMETERS)
        raise ValueError(f"Unknown plate '{plate_label}'. Valid options: {valid}")
