# Methodology — Velocity-to-RPE Mapping

This document describes the research basis for the velocity-to-RPE and velocity-to-%1RM
models used in this application.

---

## Regression Model

All three lifts use regression equations from **Helms et al. 2017**:

> Helms ER et al. "RPE and Velocity Relationships for the Back Squat, Bench Press, and
> Deadlift in Powerlifters." *Journal of Strength and Conditioning Research*, 31(2), 292–297.
> [PubMed 27243918](https://pubmed.ncbi.nlm.nih.gov/27243918/)

**Sample:** 15 competitive powerlifters (12M, 3F), mean age 28.4 ± 8.5 years.

| Lift | Equation (%1RM from MCV) | r | R² |
|---|---|---|---|
| Back Squat | %1RM = −0.449 × MCV + 1.096 | −0.91 | 0.83 |
| Bench Press | %1RM = −0.600 × MCV + 1.051 | −0.90 | 0.81 |
| Deadlift | %1RM = −0.600 × MCV + 1.076 | −0.92 | 0.85 |

Projected 1RM is derived by rearranging: **1RM = bar weight / %1RM**

---

## %1RM → RPE Mapping

Standard VBT convention used throughout the field:

| %1RM | RPE |
|---|---|
| 100% | 10.0 |
| 97.5% | 9.5 |
| 95% | 9.0 |
| 92.5% | 8.5 |
| 90% | 8.0 |
| 87.5% | 7.5 |
| 85% | 7.0 |
| 82.5% | 6.5 |
| 80% | 6.0 |

---

## Minimum Velocity Thresholds at 1RM

From the same study (ACV at 1RM, RPE ~9.6–9.7):

| Lift | MVT (m/s) |
|---|---|
| Back Squat | 0.23 ± 0.05 |
| Bench Press | 0.10 ± 0.04 |
| Deadlift | 0.14 ± 0.05 |

---

## Population Considerations

These tables are calibrated on competitive powerlifters. Key implications:

- **General trained lifters** have higher MVTs (squat ~0.30–0.34, bench ~0.15–0.16 m/s)
  — their RPE will be slightly underestimated by these tables
- **Women** may have slightly lower MVTs (squat ~0.23, bench ~0.19 m/s)
- Day-to-day velocity fluctuation from fatigue/sleep means RPE estimates carry ~±0.5
  uncertainty even with a perfect model
- Most reliable near failure (RPE ≥ 8); less precise at RPE 6–7

---

## Mean Concentric Velocity Calculation

MCV is computed over the **burst window** — frames within the concentric phase where
velocity ≥ 20% of the peak concentric velocity. This trims the near-zero reversal
frames at the bottom of the squat (where the bar changes direction) and matches the
measurement window used by dedicated VBT devices such as GymAware and PUSH.

Concentric phase boundaries:
- **Start:** position minimum (bottom of the rep)
- **End:** first local position maximum after the bottom (initial lockout, before bar whip oscillation)

---

## Pixel-to-Metre Calibration

The outer plate diameter is used as a known reference object. Supported plates:

| Plate | Diameter (m) |
|---|---|
| 45 lb / 25 kg / 20 kg | 0.450 |
| 35 lb | 0.420 |
| 25 lb / 15 kg | 0.400 / 0.380 |
| 10 lb / 10 kg | 0.280 / 0.320 |

The Hough-detected plate radius in pixels gives `m/px = plate_diameter_m / plate_diameter_px`,
which is applied to all subsequent displacement calculations.

---

## References

- Helms et al. 2017 — [PubMed](https://pubmed.ncbi.nlm.nih.gov/27243918/)
- González-Badillo & Sánchez-Medina 2010 — [PubMed 20180176](https://pubmed.ncbi.nlm.nih.gov/20180176/)
- Sanchez-Medina et al. — [PMC6226068](https://pmc.ncbi.nlm.nih.gov/articles/PMC6226068/)
- Benavides-Ubric et al. — [PMC7429441](https://pmc.ncbi.nlm.nih.gov/articles/PMC7429441/)
- Balsalobre-Fernandez et al. 2022 — [PMC9180020](https://pmc.ncbi.nlm.nih.gov/articles/PMC9180020/)
