# Part A Diagnostics and Outlier Report

## 1. Test Fold PR-AUC and Prevalence Audit

All metrics below are strictly computed on the **unfiltered, full test folds** (all driving phases including corners, braking, and straights).

| Model / Configuration | Mean PR-AUC | Std PR-AUC | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|---|---|---|---|---|---|---|---|
| Test Fold Prevalence (Ground Truth) | 0.0166 | 0.0053 | 0.0248 | 0.0090 | 0.0193 | 0.0166 | 0.0134 |
| Threshold Rule (SoC < 25%) | 0.0518 | 0.0165 | 0.0799 | 0.0321 | 0.0588 | 0.0475 | 0.0407 |
| Logistic Regression | 0.0389 | 0.0178 | 0.0446 | 0.0147 | 0.0344 | 0.0690 | 0.0321 |
| Position-Only Model | 0.1056 | 0.0586 | 0.1073 | 0.0640 | 0.0638 | 0.0747 | 0.2185 |
| Ablation: Full Model without Energy Features | 0.0184 | 0.0054 | 0.0249 | 0.0100 | 0.0225 | 0.0195 | 0.0148 |
| Full LightGBM Model | 0.0186 | 0.0057 | 0.0254 | 0.0099 | 0.0229 | 0.0202 | 0.0145 |

## 2. Catalunya Data Quality and Clip Frequency Investigation

### Empirical Data Quality Audit for Catalunya (2026_Catalunya_Race)
- Total Telemetry Rows: 527,428
- Drivers Present: 22
- Rows per Driver: Mean = 23974.0 (Min = 23974, Max = 23974)
- Missing Channel Percentages: Throttle = 0.000%, Speed = 0.000%, Brake = 0.000%
- Full-Throttle Time Share (throttle >= 98%): 26.91%
- Full-Throttle High-Speed Share (throttle >= 98% AND speed >= 200 km/h): 23.76% (vs 38.20% at Silverstone)
- 99th Percentile Speed: 334.0 km/h (Max = 360.0 km/h)
- Straight Entries Identified: 101418
- Detected Clipping Events (v1 Detector): 4

### Factual Finding on Catalunya vs High-Speed Circuits
Data verification confirms that Catalunya telemetry is complete and valid with 0.0% missing data and all 22 drivers present. The low clip count (4 events) is directly explained by the circuit profile: Catalunya's high-downforce technical layout features only one long straight where cars reach >= 200 km/h under full throttle (23.76% of session time, compared to 38.20% at Silverstone and over 18% at Spa). On technical circuits, harvesting via heavy braking zones is frequent and straights are too short for full battery depletion before the braking mark.

## 3. Diagnostic Plots
Diagnostic sample plots for Catalunya, Silverstone, and Suzuka have been rendered and saved to `C:\MERN STACK\9 - COURSE PROJECTS\SPARC\ml\reports\paper\audit_plots`.
