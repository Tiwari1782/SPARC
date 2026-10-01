# Paper Outline: Benchmarking Early Hybrid Energy Clipping Prediction from Public Broadcast Telemetry in Formula 1

## Target Venue
Applied Sports Analytics, IEEE/ACM AI in Sports Workshop, or arXiv preprint.

## Framing
This is an honest negative-result and benchmark paper. The realistic target is arXiv, a workshop, or a student/sports-analytics venue. It documents what public telemetry can and cannot do, and provides a reusable benchmark for future work.

---

## Abstract (~150 words)

The 2026 Formula 1 regulations triple the electric motor output to 350 kW, making battery energy management the single largest tactical factor in modern racing. When a car exhausts its deployment allowance mid-straight, it loses 40-50 km/h relative to rivals -- a phenomenon called clipping. Team battery data is secret, but speed, throttle, brake, and gear are public. We present SPARC, a digital twin that reconstructs a physics-informed battery state-of-charge proxy from public telemetry and attempts to predict clipping 300 m ahead of time using a LightGBM classifier. Evaluated across 15 Grand Prix with strict leave-race-out cross-validation, the model achieves PR-AUC 0.17 against a 1.2% prevalence (14x skill over random), statistically beating a track-position-only baseline (p = 0.004). However, ablation reveals that track geometry -- not energy state -- dominates prediction. We characterise this information ceiling as a fundamental limit of broadcast telemetry and release the pipeline as a benchmark.

---

## 1. Introduction (1-1.5 pages)

### 1.1 The 2026 Energy Problem
- 50:50 ICE/electric power split under new regulations
- MGU-K output 350 kW (up from 120 kW), MGU-H eliminated
- Battery capacity ~4 MJ usable per lap, ~8.5 MJ recovery cap
- Clipping: car runs out of deployment allowance mid-straight, loses 40-50 km/h
- Real-world examples: 2024 Canadian GP (Russell vs Verstappen), 2024 British GP (Norris defending)

### 1.2 The Public Data Gap
- Team ECU data (true SoC, cell temperatures, driver mode switches) is secret
- Public broadcast telemetry: speed, throttle, brake, gear, position, lap timing
- No existing public tool reconstructs battery state or predicts clipping

### 1.3 Our Contribution
1. A physics-informed SoC proxy validated against observable clipping events
2. Three proposed energy indices: CPI, EDR, BAI (with literature novelty caveat)
3. A 15-race benchmark with strict evaluation protocol
4. An honest characterisation of what public telemetry cannot predict

### 1.4 Paper Organisation
Section 2: Related work. Section 3: System design. Section 4: Data and labels. Section 5: Experiments. Section 6: Results. Section 7: Discussion. Section 8: Conclusion.

---

## 2. Related Work (0.5-1 page)

### 2.1 Motorsport Telemetry Analysis
- FastF1 and OpenF1 as public telemetry sources
- Prior telemetry studies (lap time prediction, tyre degradation, pit strategy)
- Note: most published work focuses on ICE-era features (fuel, tyres), not electric energy

### 2.2 Energy Management in Hybrid Racing
- Team-side deployment optimisation (Lap-by-lap energy strategy papers from SAE, F1 engineering)
- Simulation-based approaches (AMuSE, rFactor, Assetto Corsa as proxies)
- Key gap: no published work predicts clipping from public data alone

### 2.3 Digital Twins in Sport
- AeroTwin pattern (turbofan fatigue monitoring as structural analogy)
- Real-time estimation from partial observability
- State estimation under unobservable driver intent

### 2.4 Literature Search for CPI, EDR, BAI
- **Mandatory before submission**: Search Google Scholar, arXiv, SAE for similar energy indices
- If prior art exists, cite it and reframe as "we adopt/extend" rather than "we propose"

---

## 3. System Design (1.5-2 pages)

### 3.1 Architecture Overview
- Data source -> Energy engine -> Clip detector -> Feature engineer -> ML model -> Alert engine -> Dashboard
- Diagram: Figure showing the pipeline

### 3.2 Physics-Informed SoC Proxy
- SoC update equation: SoC(k+1) = clamp(SoC(k) + [E_harvest - E_deploy] / E_battery * 100, 0, 100)
- Harvest model: braking intensity, lift-off regen, per-team regen sensitivity
- Deploy model: throttle intensity, speed taper, per-team deploy sensitivity
- Fuel mass correction: m_fuel(lap) = m_fuel_start - burn_rate * lap

### 3.3 Energy Indices
- CPI: (D_ahead / R_avail)^1.3 * (1 + 0.5 * P_rival) * W_fuel
- EDR: (deploy_lap - harvest_lap) / lap_length
- BAI: CPI_defender - CPI_attacker
- All constants are fitted, not physical laws (state this explicitly)

### 3.4 Per-Team Calibration
- Fitted parameters: capacity_kwh, regen_sens, deploy_sens per team
- Objective: minimise SoC at observed clip onset, maximise SoC elsewhere
- Calibration is on training data only; never leaked into test evaluation

---

## 4. Data and Labels (1-1.5 pages)

### 4.1 Data Source
- FastF1 public telemetry, 2026 season, 15 Grand Prix
- Channels: Speed, Throttle, Brake, nGear, RPM, X, Y, Distance, lap timing
- Resampled to 4 Hz unified time grid across all 22 drivers
- 8.28 million telemetry samples total

### 4.2 Clipping Label Construction
- Acceleration-envelope method: compare observed acceleration to the car's own expected envelope
- Conditions: throttle >= 98%, no braking, acceleration deficit sustained for minimum distance
- Exclusion: terminal (drag-limited) speed reached
- v2 labels with circuit-specific adaptive thresholds (vs global thresholds in v1)
- Label count: 7,471 events, prevalence 1.16%

### 4.3 Label Sensitivity Analysis
- Table: v1 global thresholds vs v2 circuit-adaptive thresholds
- Sensitivity to threshold parameters (Table from label_sensitivity.csv)

### 4.4 Human Audit (if completed)
- 150 stratified samples, annotated by two raters
- Cohen's Kappa agreement score
- **Note**: If audit is not completed before submission, state: "Human audit is planned but not yet complete. Results should be interpreted with this caveat."

---

## 5. Experimental Setup (1 page)

### 5.1 Features
- Table of all features grouped by category (energy, track, kinematics, race context, conditions, team)
- 25+ features total

### 5.2 Models
- **Baselines**: SoC threshold (<25%), Logistic Regression, Position-Only (track geometry features only)
- **Primary**: LightGBM, Random Forest
- All CPU-only, no GPU required

### 5.3 Evaluation Protocol
- 5-Fold GroupKFold by race (leave-race-out)
- Threshold selection on inner validation splits (never test data)
- Event-level debouncing: at most one alert per straight per driver
- Metrics: PR-AUC, event recall, event precision, mean lead distance, false alarms per lap-driver
- Leakage control: shuffled-label test

### 5.4 Statistical Tests
- Bootstrap 95% CI on PR-AUC across 15 races
- Wilcoxon signed-rank for paired model comparisons
- Cohen's d effect size

---

## 6. Results (1.5 pages)

### 6.1 Main Results Table
- Table 1: All models compared (from results.md Section 2)
- Highlight: LightGBM PR-AUC 0.17, 14x skill over random prevalence
- Operating point: 28.61% event recall at <= 1 FA per lap-driver

### 6.2 Statistical Significance
- LightGBM vs Position-Only: p = 0.004, d = 0.812
- LightGBM vs SoC Threshold: p = 0.004, d = 0.968
- Bootstrap CI: [0.094, 0.239]

### 6.3 Ablation Analysis
- Table 2: Feature group ablations (from results.md Section 4)
- Key finding: removing all energy features yields PR-AUC 0.198 (same or higher)
- Removing kinematic acceleration features: PR-AUC 0.155 (circularity check passes)

### 6.4 Feature Importance
- Figure 3: Top features by gain and split frequency
- Dominated by: straight_length, distance_to_straight_end, lap_fraction, speed

### 6.5 SoC Proxy Validation
- Figure 4: SoC distance profiles show 50-62 percentage point separation between pre-clip and normal driving
- Confirms physical consistency of the proxy even if predictive power is limited

### 6.6 Leakage Control
- Shuffled-label PR-AUC: 0.023 (vs prevalence 0.012) -- confirms no leakage

---

## 7. Discussion (1-1.5 pages)

### 7.1 What the Model Learns
- The model learns where on the track clipping tends to happen (long straights, late in the straight)
- It does not learn the driver's internal energy state from public data
- This is an information ceiling, not a modelling failure

### 7.2 The Information Ceiling of Broadcast Telemetry
- True SoC, driver mode switches, cell temperatures are unobservable
- With proprietary data, precision/recall would likely exceed 90% (clipping is a direct threshold function of remaining Joules vs straight length)
- The contribution is demonstrating what independent researchers can achieve without that data

### 7.3 Comparison to What Teams Have
- Table: Public vs Proprietary features (from claims_and_proofs.md Section 2)
- The paper answers: "Can independent researchers predict clipping from public data?"
- Answer: "Partially. ~29% of events, ~190 m early, with meaningful skill over baselines, but bounded by tactical unobservability."

### 7.4 Threats to Validity
- Label noise (no human audit completed yet, if applicable)
- SoC proxy is not validated against ground truth (it cannot be)
- Performance varies across circuits (fold-level std 0.11)
- Limited to 2026 season data; cannot test on 2025 or earlier (different regulations)

### 7.5 Implications for Broadcasters and Fans
- Even partial clipping prediction adds value for live commentary
- The SoC proxy provides a previously unavailable energy visualisation
- CPI, EDR, BAI indices can enhance fan engagement and tactical commentary

---

## 8. Conclusion (~0.5 page)

- We present the first public benchmark for hybrid energy clipping prediction in Formula 1
- The pipeline reconstructs physically consistent battery state from broadcast telemetry
- Early prediction is possible (14x skill over random) but fundamentally bounded by unobservable driver intent
- Track geometry, not energy state, is the dominant predictor on public data
- Future work: additional data sources (onboard audio for engine mode changes, steering wheel camera for switch positions), sequence models, more seasons

---

## Figures and Tables

| Label | Content | Source File |
|---|---|---|
| Figure 1 | PR curves for all models | `fig1_pr_curves.png` |
| Figure 2 | Ablation bar chart | `fig2_ablations.png` |
| Figure 3 | Feature importance (gain + splits) | `fig3_feature_importance.png` |
| Figure 4 | SoC vs distance profiles | `fig4_soc_distance_profiles.png` |
| Table 1 | Main results comparison | `results.md` Section 2 |
| Table 2 | Feature ablation results | `results.md` Section 4 |
| Table 3 | Label sensitivity analysis | `label_sensitivity.csv` |
| Table 4 | Public vs proprietary features | `claims_and_proofs.md` Section 2 |

---

## Appendix (optional)

- A: Full feature list with descriptions
- B: Per-circuit breakdown of model performance
- C: Model card (responsible AI documentation)
- D: Reproducibility checklist

---

## Writing Tone Reminders

1. **Do not overclaim.** The model is 14x prevalence, not "high-accuracy."
2. **Label SoC as "estimated"** and the predictor as "experimental" throughout.
3. **The ablation is the most important finding.** Energy features are outperformed by geometry. Lead with this.
4. **Frame as a benchmark**, not a solved problem. The contribution is the pipeline, the evaluation protocol, and the honest characterisation of the information ceiling.
5. **Acknowledge label noise.** State the human audit status explicitly.
6. **Check CPI/EDR/BAI novelty** against literature before claiming "we propose."
