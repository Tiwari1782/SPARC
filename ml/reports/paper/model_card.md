# SPARC Model Card

## Model Overview

- **Name**: SPARC Clip Predictor (LightGBM, version 2)
- **Task**: Binary classification: will a clipping event begin within the next 300 m for a given driver at a given telemetry sample?
- **Type**: Gradient-boosted decision trees (LightGBM)
- **Output**: Probability score [0, 1]; debounced to at most one alert per straight

## Intended Use

Real-time early warning of hybrid battery energy clipping for 2026 Formula 1 cars using only public broadcast telemetry. Designed for:
- Fan and analyst dashboards showing energy state across the grid
- Research and benchmarking of energy prediction from public data
- Educational demonstrations of digital twin architecture

## Out-of-Scope Uses

- Official race strategy or engineering decisions (SoC is estimated, not measured)
- Application to pre-2026 regulation eras without re-calibration
- Any safety-critical or financial decision making

## Training Data

- **Source**: FastF1 public telemetry from the 2026 FIA Formula 1 World Championship
- **Volume**: 15 Grand Prix, 8,282,692 telemetry samples at 4 Hz
- **Labels**: 7,471 clipping events detected via the adaptive acceleration-envelope method (v2 labels with circuit-specific thresholds)
- **Prevalence**: 1.16% positive class (highly imbalanced)

## Evaluation Protocol

- **Split strategy**: 5-Fold GroupKFold by race (leave-race-out); no random splitting
- **Threshold selection**: Inner-validation on training folds only; never on test data
- **Debouncing**: At most one alert per straight segment in event-level metrics
- **Leakage check**: Shuffled-label control collapsed PR-AUC to empirical prevalence (0.023 vs 0.012)

## Quantitative Metrics (Full Test Folds)

| Metric | Value |
|---|---|
| PR-AUC (macro-average over 5 folds) | 0.1696 |
| PR-AUC (95% bootstrap CI over 15 races) | [0.0937, 0.2392] |
| Event Recall at 1 FA/lap-driver | 0.2861 (28.61%) |
| Mean Lead Distance | 190.3 m before clip onset |
| False Alarms per Lap-Driver | 1.06 |

## Baseline Comparisons

| Baseline | PR-AUC |
|---|---|
| SoC Threshold (<25%) | 0.0383 |
| Logistic Regression | 0.0409 |
| Position-Only (track geometry) | 0.1181 |
| **LightGBM (this model)** | **0.1696** |

## Statistical Significance

- LightGBM vs Position-Only: Wilcoxon p = 0.0043, Cohen's d = 0.812 (large)
- LightGBM vs SoC Threshold: Wilcoxon p = 0.0035, Cohen's d = 0.968 (large)

## Feature Groups (by importance)

1. **Track geometry**: `straight_length`, `distance_to_straight_end`, `lap_fraction`
2. **Driver kinematics**: `speed`, `throttle_mean_50m`, `speed_slope_50m`
3. **Energy estimates**: `soc_est`, `edr_lap`, `cpi` (contribute but are outperformed by geometry)
4. **Race context**: `fuel_load_est_kg`, `tyre_age`, `gap_ahead_s`

## Known Limitations

1. **Energy features do not dominate**: Ablation shows removing all energy features yields similar or slightly higher PR-AUC (0.1983). The model primarily learns where on the track clipping tends to occur, not the driver's internal energy state.
2. **SoC is an estimate**: True battery state-of-charge is proprietary team data. Our physics proxy is validated against observed clipping events but cannot be verified against ground truth.
3. **Label noise**: Clipping labels are derived from telemetry heuristics (acceleration-envelope method). Some labels may be false positives (traffic, deliberate lift) or false negatives (subtle clipping below detection threshold).
4. **Tactical unobservability**: Driver engine mode settings, overtake button usage, and lift-and-coast strategy are invisible in broadcast telemetry. These directly cause or prevent clipping.
5. **Circuit generalization**: Performance varies significantly across circuits (fold-level PR-AUC std = 0.1137). Circuits with long straights and frequent clipping are easier to predict.

## Ethical Considerations

- Model outputs should be labelled "experimental" in any user-facing display
- SoC values should be labelled "estimated" to avoid confusion with real team data
- The project is unofficial and is not affiliated with Formula 1, the FIA, or any team

## Reproducibility

- **Random seed**: 42 (fixed everywhere)
- **Pipeline**: `process_races.py` -> `build_labels.py` -> `calibrate_teams.py` -> `evaluate_paper.py`
- **Total runtime**: ~22 minutes on Intel i5, 16 GB RAM, no GPU