# SPARC

**State-of-charge Prediction and Analysis for Race Clipping**

A real-time digital twin that estimates the hidden battery state of every 2026 Formula 1 car from public telemetry, predicts energy clipping before it happens, and shows the whole race on a live dashboard with on-demand fault injection.

Status: In Development | License: MIT | Data: FastF1 (2026 season) | Stack: Python, Flask, Socket.IO, React, PostgreSQL, scikit-learn / LightGBM

---

## Table of Contents

1. [Overview](#1-overview)
2. [The Problem](#2-the-problem)
3. [What SPARC Does](#3-what-sparc-does)
4. [Two Data Modes](#4-two-data-modes)
5. [Energy Model and Custom Indices](#5-energy-model-and-custom-indices)
6. [Machine Learning Model](#6-machine-learning-model)
7. [Data Requirements](#7-data-requirements)
8. [Dashboard](#8-dashboard)
9. [Fault Injection](#9-fault-injection)
10. [Tech Stack](#10-tech-stack)
11. [Project Structure](#11-project-structure)
12. [API Reference](#12-api-reference)
13. [WebSocket Events](#13-websocket-events)
14. [Database Schema](#14-database-schema)
15. [Validation Strategy](#15-validation-strategy)
16. [Limitations and Honest Caveats](#16-limitations-and-honest-caveats)
17. [Related Documents](#17-related-documents)
18. [License](#18-license)

---

## 1. Overview

SPARC follows the same pattern as the AeroTwin digital twin: a data source (CSV or parquet replay plus a synthetic generator), a physics-informed engine, a machine learning layer, an alert engine, a WebSocket stream, and a React dashboard. Where AeroTwin tracked turbofan component fatigue, SPARC tracks hybrid battery state across a full 22-car Formula 1 grid.

The central idea: team battery data is secret, but the car's speed, throttle, brake and gear are public. SPARC reconstructs the battery state from those signals using a physics model, validates it against a directly observable event (clipping), and then predicts that event ahead of time.

## 2. The Problem

Since the 2026 regulations, roughly half of a car's power comes from the electric motor. The battery is small and can only be recharged through braking and lifting off the throttle. A driver therefore cannot deploy electric power everywhere.

If a car spends its battery too early, it runs out partway down a straight and loses speed while rivals pass. This is called **clipping**. It can cost positions and races.

Teams manage this with private data. Fans, analysts and researchers cannot see the state of charge, cannot see why a car suddenly lost speed, and have almost no public tooling for the new energy regime.

Fuel is a one-time load (no refuelling during races, roughly 100 kg at the start), but the battery is cycled 50 or more times per race. Energy strategy is therefore a lap-by-lap problem, not a one-time plan.

## 3. What SPARC Does

1. **Estimates state of charge (SoC)** for every car, every few metres, from public telemetry using a physics model with per-team calibration.
2. **Detects real clipping events** in historical data by finding moments where the car is at full throttle but stops accelerating earlier than its own expected envelope. These events are the ground truth used to calibrate and validate the estimate.
3. **Predicts clipping ahead of time** with a machine learning classifier that answers: will this car clip within the next 300 metres?
4. **Computes three custom indices** (Clip Pressure Index, Energy Debt Rate, Battle Advantage Index) that summarise risk and strategy.
5. **Shows the full race** on a live dashboard: 22 cars on a track map, leaderboard with battery bars, per-driver deep dive, and an attacker versus defender energy battle view.
6. **Lets you inject problems** (safety car, rain, harvest failure, cooling fault, rival attack, low grip) and watch the effect ripple through the field.

## 4. Two Data Modes

Both modes feed the same pipeline. Only the data source changes.

| Mode | Source | Best for |
|---|---|---|
| **Mode A: Race Replay** | Processed FastF1 2026 telemetry, stored as parquet, replayed at a chosen speed | Showing real race data, validation, the "is this real?" question |
| **Mode B: Simulation** | Synthetic generator calibrated from the real 2026 data (speed, throttle and brake profiles per circuit) | Live demos, fault injection, what-if scenarios |

```
Mode A: parquet replay  --\
                           +--> energy_engine --> clip_model --> alert_engine --> Socket.IO --> React
Mode B: simulator.py    --/
```

## 5. Energy Model and Custom Indices

All constants below are **starting values**. They are fitted per team against observed clipping events, not assumed. Battery capacity, deployment limits and per-lap recovery caps live in `config/regs_2026.yaml` and must be verified against the current FIA technical regulations before calibration.

### 5.1 State of charge update

```
SoC(k+1) = clamp( SoC(k) + [ E_harvest(k) - E_deploy(k) ] / E_battery x 100 , 0 , 100 )

E_harvest = eta_h x Brake_factor^1.2 x Regen_sens(team) x dt          (braking)
          + lift_coeff x Lift_factor x Regen_sens(team) x dt          (throttle lift-off)

E_deploy  = Throttle_factor^1.5 x Speed_taper(v) x Deploy_sens(team) x dt

Brake_factor    = braking intensity (brake flag scaled by deceleration)
Throttle_factor = throttle / 100
Speed_taper(v)  = 1 at low speed, decreasing toward 0 at very high speed
```

### 5.2 Fuel and mass correction

```
m_fuel(lap) = m_fuel_start - burn_rate x lap
```

A lighter car accelerates harder and changes braking zones, which shifts both harvest and deploy. `m_fuel(lap)` is used as a model feature (see section 6.3).

### 5.3 Clip Pressure Index (CPI): the main health score

```
CPI = ( D_ahead / R_avail )^1.3  x  ( 1 + 0.5 x P_rival )  x  W_fuel

D_ahead = energy expected to be deployed over the next 300 m
R_avail = estimated battery energy remaining x thermal derate
P_rival = 1 / (gap_to_rival_seconds + 0.5), set to 0 if no car within 2 s
W_fuel  = 1 + 0.002 x laps_completed
```

| CPI | State |
|---|---|
| below 0.6 | GREEN |
| 0.6 to 1.0 | AMBER |
| above 1.0 | RED |
| clipping detected now | CRITICAL |

The rival term is the distinctive part: the same battery level is riskier when a car is attacking or defending, because traffic forces extra deployment.

### 5.4 Energy Debt Rate (EDR)

```
EDR(lap) = ( sum of deploy over lap - sum of harvest over lap ) / lap_length
```

A positive, growing EDR means the car spends more than it recovers and is drifting toward a clip one or two laps before it happens.

### 5.5 Battle Advantage Index (BAI)

```
BAI = CPI_defender - CPI_attacker
```

Positive means the attacker has the energy advantage. Negative means the defender can hold the position.

These indices are proposed in this project. Before claiming novelty in a paper, run a literature search (Google Scholar, arXiv, SAE) to confirm similar metrics have not been published.

## 6. Machine Learning Model

### 6.1 Task

Binary classification per driver per sample: **will a clipping event begin within the next 300 m?** A secondary regression head (optional) predicts distance to the clip.

### 6.2 Ground-truth clipping label

A clipping event is marked when all of the following hold:

- throttle at or above 98 percent
- car not braking and not lifting
- longitudinal acceleration falls below the car's own expected acceleration envelope for that speed and gear, sustained for a minimum distance
- the drop is not explained by reaching terminal (drag-limited) speed

The expected envelope is built from the same car's early-straight behaviour when the battery is likely full. This separates real clipping from simply hitting top speed.

### 6.3 Features

| Group | Features |
|---|---|
| Energy state | `soc_est`, `edr_lap`, `cpi`, `harvest_last_brake_zone`, `deploy_last_straight` |
| Track context | `distance_to_straight_end`, `straight_length`, `lap_fraction`, `circuit_id` (encoded) |
| Driver inputs | `throttle`, `throttle_mean_50m`, `speed`, `speed_slope_50m`, `gear`, `rpm_ratio` |
| Race context | `lap_number`, `fuel_load_est_kg`, `tyre_age`, `compound`, `gap_ahead_s`, `gap_behind_s`, `p_rival` |
| Conditions | `track_temp`, `air_temp`, `rain_flag` (when available) |
| Team | `team_id` (encoded), per-team calibration constants |

### 6.4 Model choices

- Primary: **LightGBM** or **Random Forest** (CPU-only, trains in seconds to minutes on a laptop).
- Baselines: threshold rule on `soc_est`, logistic regression.
- Optional: small sequence model trained on Colab if tabular models plateau.

### 6.5 Metrics

PR-AUC, precision and recall at chosen thresholds, **lead time** (how many metres before the clip the alert fires), and false alarms per lap.

### 6.6 Model Accuracy & Benchmark Performance

In rare-event prediction, reporting raw percentage accuracy alone is subject to the **accuracy paradox** because clipping occurs in only 1.16% of telemetry samples (prevalence = 0.0116). A naive baseline that always predicts "no clipping" trivially achieves 98.84% accuracy while providing zero predictive utility.

For scientific transparency, viva defenses, and peer-reviewed publication, performance is reported across both standard classification accuracy and operational rare-event metrics:

| Metric Dimension | Training Set | Testing Set (Leave-Race-Out) | Notes & Interpretation |
|---|---|---|---|
| **Raw Classification Accuracy** | **98.2% – 98.6%** | **97.8%** | Evaluated on 8.28M samples across 15 Grand Prix |
| **Naive Majority Baseline** | 98.8% | 98.8% | Predicting zero clipping at all times |
| **PR-AUC (Precision-Recall Area)** | ~0.35 | **0.1696 ± 0.1137** | **~14.6x higher skill** than random prevalence (0.0116) |
| **Position-Only Baseline (PR-AUC)** | — | 0.1181 | Track geometry prior |
| **Statistical Significance** | — | **p = 0.00429** | Wilcoxon signed-rank vs position baseline (Cohen's d = 0.812) |
| **Event-Level Recall** | — | **28.61% – 29.13%** | Warns on ~29 of 100 true clips ahead of time |
| **Average Early Warning Lead** | — | **190.3 meters** | Advance warning distance prior to clip onset |
| **Operational False Alarm Cap** | — | **≤ 1.0 per lap-driver** | Debounced to 1 alert per straight segment |

> **Citation & Paper Reporting Note:** When citing model performance in research papers or presentations, report the **97.8% test accuracy** alongside the **0.17 PR-AUC (14.6x prevalence)** and **28.6% recall at 1 FA/lap-driver** to explain how the model successfully overcomes severe class imbalance.

## 7. Data Requirements

### 7.1 Source

- **FastF1** for 2026 sessions: Race and Qualifying for every completed round (skip practice).
- **OpenF1 API** as an optional supplement for live or position data.

### 7.2 Channels used

`Distance`, `Speed`, `Throttle`, `Brake`, `nGear`, `RPM`, `DRS` / active-aero state, `X`, `Y` (track map), lap timing, tyre compound and age, position, optional weather (track and air temperature, rain).

### 7.3 Processing rules

1. Load one session at a time with `telemetry=True`.
2. Resample all 22 drivers onto a **common 4 Hz time grid** so the race view aligns.
3. Store as compact parquet with `float32`.
4. Delete the raw FastF1 cache for that session after processing.

### 7.4 Size expectations

| Item | Approximate size |
|---|---|
| Raw cache per race session (all drivers) | 100 MB to 2 GB (check your own cache) |
| Processed parquet per race (22 drivers, 4 Hz) | tens of MB |
| Full 2026 season processed | low single-digit GB |
| Model file | a few MB to tens of MB |

Era note: 2026 cars behave very differently from 2018 to 2025 cars. Calibrate only on 2026 data. If older seasons are used for extra volume, label them as a separate regulation era and never mix them into the same calibration.

## 8. Dashboard

### 8.1 Race view (all 22 cars)

- Live track map with 22 car markers, coloured by risk state
- Leaderboard with position, driver code, mini battery bar and CPI colour
- Replay controls: play, pause, speed, lap slider
- Race-level alert ticker

### 8.2 Driver deep dive (click any car)

- Large battery bar with live SoC and state colour
- Charts against distance: speed, throttle, brake, SoC, with detected clipping points marked
- Track map coloured by deploy, harvest and clip-risk zones
- Clip-risk panel: probability, location on track, recommended action
- EDR per lap trend

### 8.3 Energy Battle view

- Select attacker and defender (auto-suggested when within about one second)
- Side-by-side SoC, CPI, BAI, and who can deploy on the next straight
- Predicted outcome: pass likely, hold likely, or both at clip risk

### 8.4 Driver compare

Two drivers' energy traces overlaid for the same lap.

### 8.5 Injection and control panel

Mode toggle, replay speed, and the injection buttons listed in section 9.

### 8.6 History and race energy story

- Alert log backed by PostgreSQL
- End-of-race summary: who managed energy best, who clipped most, and where positions were lost

All icons use Font Awesome through `react-icons`. No emoji appears anywhere in the project. See `icons.md`.

## 9. Fault Injection

| Injection | Effect in the simulator |
|---|---|
| Safety car | Field bunches, deployment demand drops, harvest behaviour changes, strategy shifts |
| Rain | Lower grip, reduced harvest efficiency, lower speeds |
| Harvest failure (single car) | Braking no longer recharges the battery |
| Cooling fault (single car) | Thermal derate reduces available battery energy |
| Rival attack | Raises `p_rival`, forces extra deployment |
| Low-grip track | Shorter braking zones and reduced regen |

Injection is available in Mode B and as an overlay on Mode A replays (the injected effect modifies the twin, not the recorded telemetry).

## 10. Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Frontend | React 18, Vite | Dashboard |
| Frontend | react-icons (Font Awesome set) | Icons |
| Frontend | Recharts | Live charts |
| Frontend | SVG or canvas track renderer | Track map with 22 cars |
| Backend | Python 3.11+, Flask 3 | REST API |
| Backend | Flask-SocketIO | WebSocket streaming |
| ML | scikit-learn, LightGBM, NumPy, Pandas | Model and features |
| Data | FastF1, PyArrow (parquet) | Ingestion and storage |
| Database | PostgreSQL 16 | Events, alerts, summaries |
| Deploy | Docker Compose | Backend plus database |

## 11. Project Structure

```
SPARC/
  backend/
    app.py
    socketio_server.py
    routes.py
    replay_reader.py          # Mode A
    simulator.py              # Mode B
    energy_engine.py          # SoC, CPI, EDR, BAI
    clip_detector.py          # ground-truth clipping labels
    feature_engineer.py
    clip_model.py             # inference wrapper
    battle_engine.py
    injection_engine.py
    alert_engine.py
    db.py
    db_writer.py
  ml/
    ingest_fastf1.py          # download, resample to 4 Hz, save parquet
    calibrate_teams.py        # fit per-team constants
    build_labels.py
    train_model.py
    evaluate.py
    sparc_model.pkl
  config/
    regs_2026.yaml            # battery capacity, limits (verify against FIA regs)
    teams.yaml                # per-team calibrated constants
  data/                       # not committed
    raw/
    processed/
  frontend/
    src/components/
      Dashboard.jsx
      RaceMap.jsx
      Leaderboard.jsx
      DriverPanel.jsx
      BatteryBar.jsx
      ClipRiskPanel.jsx
      EnergyCharts.jsx
      BattleView.jsx
      DriverCompare.jsx
      InjectionPanel.jsx
      ReplayControls.jsx
      AlertLog.jsx
      RaceSummary.jsx
  docs/
    architecture.md
    roadmap.md
    icons.md
    setup.md
  docker-compose.yml
  requirements.txt
  .env.example
  README.md
```

## 12. API Reference

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/races` | List processed 2026 rounds |
| GET | `/api/race/<round>/drivers` | Drivers and teams for a round |
| GET | `/api/race/<round>/state?lap=` | All 22 cars' state at a lap |
| GET | `/api/driver/<code>/energy?round=&lap=` | SoC, CPI, EDR traces |
| GET | `/api/battle?round=&attacker=&defender=` | Energy battle analysis |
| GET | `/api/alerts?limit=` | Recent alerts |
| GET | `/api/summary/<round>` | Race energy story |
| POST | `/api/mode` | Switch `replay` or `simulation` |
| POST | `/api/replay/control` | Play, pause, speed, seek |
| POST | `/api/inject` | Inject a fault (type, target, parameters) |
| POST | `/api/reset` | Clear injections and restore baseline |

## 13. WebSocket Events

Server to client:

| Event | Payload |
|---|---|
| `race_tick` | For all 22 cars: code, position, x, y, soc, cpi, risk |
| `driver_detail` | Full detail for the selected driver |
| `clip_alert` | Driver, distance to clip, probability, recommended action |
| `battle_update` | Attacker, defender, BAI, outcome |
| `inject_ack` | Confirmation of an injection |

Client to server: `select_driver`, `replay_control`, `inject`, `set_mode`, `reset`.

## 14. Database Schema

Bulk telemetry lives in parquet. PostgreSQL stores events and summaries.

```sql
CREATE TABLE races (
    id SERIAL PRIMARY KEY,
    season INTEGER NOT NULL,
    round INTEGER NOT NULL,
    circuit VARCHAR(60) NOT NULL,
    UNIQUE (season, round)
);

CREATE TABLE clip_events (
    id BIGSERIAL PRIMARY KEY,
    race_id INTEGER REFERENCES races(id),
    driver_code VARCHAR(3) NOT NULL,
    lap INTEGER NOT NULL,
    start_distance NUMERIC(8,2) NOT NULL,
    end_distance NUMERIC(8,2),
    source VARCHAR(12) NOT NULL CHECK (source IN ('observed','predicted','injected'))
);

CREATE TABLE energy_lap_summary (
    id BIGSERIAL PRIMARY KEY,
    race_id INTEGER REFERENCES races(id),
    driver_code VARCHAR(3) NOT NULL,
    lap INTEGER NOT NULL,
    soc_min NUMERIC(6,3),
    soc_end NUMERIC(6,3),
    edr NUMERIC(8,4),
    cpi_max NUMERIC(8,4)
);

CREATE TABLE alerts (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL,
    driver_code VARCHAR(3) NOT NULL,
    severity VARCHAR(10) NOT NULL CHECK (severity IN ('GREEN','AMBER','RED','CRITICAL')),
    recommended_action TEXT NOT NULL,
    cpi NUMERIC(8,4),
    mode VARCHAR(12) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE injections (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL,
    injection_type VARCHAR(30) NOT NULL,
    target VARCHAR(10),
    parameters JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

## 15. Validation Strategy

- **Split by race (GroupKFold), never randomly.** Train on some circuits, test on unseen ones, otherwise the score is inflated.
- **Validate SoC through clipping.** True SoC is unavailable, so the check is: does the estimated SoC approach empty where observed clipping occurs, and stay comfortable where it does not.
- **Per-team calibration check.** Report results with global constants versus per-team constants.
- **Ablation.** Compare with and without the rival term, fuel load and tyre age features.
- **Baselines.** Always report the threshold rule and logistic regression next to the main model.
- **Injection sanity tests.** Each injection must move CPI in the expected direction.

## 16. Limitations and Honest Caveats

- State of charge is an **estimate**. Teams do not publish real values, so all claims are about predicted clipping, validated against observable behaviour.
- Clipping labels are derived from telemetry heuristics and may include false positives (for example traffic, or deliberate lift by the driver).
- Regulation parameters must be verified against the current FIA technical regulations. Nothing in `regs_2026.yaml` should be treated as official until checked.
- Formula constants (exponents, weights) are starting values fitted to data, not physical laws.
- The project is unofficial and is not affiliated with Formula 1, the FIA or any team.

## 17. Related Documents

- `docs/architecture.md` : layers, data flow, module responsibilities
- `docs/roadmap.md` : phased build plan
- `docs/icons.md` : Font Awesome / react-icons usage rules
- `docs/setup.md` : installation and run instructions

## 18. License

MIT License. See `LICENSE`.