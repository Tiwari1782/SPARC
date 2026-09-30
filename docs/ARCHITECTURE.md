# SPARC Architecture

This document describes how SPARC is layered, how data moves through it, and what each module is responsible for.

---

## 1. Design Principles

1. **One pipeline, two sources.** Replay (Mode A) and simulation (Mode B) produce the same data contract, so everything downstream is identical.
2. **Precompute offline, stream live.** Heavy work (resampling, labelling, calibration, training) happens once. The live system replays processed data and runs light inference.
3. **Physics first, ML second.** The energy engine gives an interpretable state. The ML model only predicts clipping on top of it.
4. **Validate on observable events.** True battery state is hidden, so clipping is the ground truth.
5. **Replace one file to change the source.** Swapping `replay_reader.py` or `simulator.py` should not touch any other module.

## 2. Layered View

```
+--------------------------------------------------------------------+
| LAYER 5: PRESENTATION                                              |
| React 18 | Race map (22 cars) | Leaderboard | Driver panel         |
| Energy battle view | Injection panel | Alert log | Race summary    |
+------------------------------+-------------------------------------+
                               | WebSocket (Socket.IO)
+------------------------------+-------------------------------------+
| LAYER 4: APPLICATION AND API                                       |
| Flask REST API | Socket.IO server | Mode controller | Replay clock  |
+---------------+----------------------------------+-----------------+
                | read/write                       | read
+---------------+-----------+      +---------------+-----------------+
| STORAGE                   |      | LAYER 3: INTELLIGENCE           |
| PostgreSQL (events,       |<---->| energy_engine (SoC, CPI, EDR)   |
| alerts, summaries)        |      | clip_model (LightGBM / RF)      |
| Parquet (telemetry)       |      | battle_engine (BAI)             |
+---------------------------+      | alert_engine                    |
                                   | injection_engine                |
                                   +---------------+-----------------+
                                                   |
+--------------------------------------------------+-----------------+
| LAYER 2: DATA CONTRACT                                             |
| Unified tick: driver, t, distance, speed, throttle, brake, gear,   |
| rpm, x, y, lap, tyre, gap_ahead, gap_behind                        |
+--------------------------------------------------+-----------------+
                                                   |
+--------------------------------------------------+-----------------+
| LAYER 1: DATA SOURCE (only this layer changes between modes)       |
| MODE A: replay_reader.py (2026 FastF1 parquet)                     |
| MODE B: simulator.py (synthetic, calibrated from real data)        |
+--------------------------------------------------------------------+
```

## 3. Offline Pipeline (run once)

```
FastF1 session
   -> ingest_fastf1.py     load, resample all drivers to 4 Hz, add gaps, save parquet
   -> build_labels.py      detect observed clipping events (ground truth)
   -> calibrate_teams.py   fit per-team Regen_sens, Deploy_sens, taper so estimated SoC
                           approaches empty where observed clips occur
   -> train_model.py       build features, train classifier, GroupKFold by race
   -> evaluate.py          metrics, baselines, ablations
   -> sparc_model.pkl + config/teams.yaml
```

## 4. Live Data Flow (one tick)

| # | Layer | Action | Module |
|---|---|---|---|
| 1 | Source | Emit the next 4 Hz tick for all 22 drivers | `replay_reader.py` / `simulator.py` |
| 2 | Application | Apply active injections to the tick or twin parameters | `injection_engine.py` |
| 3 | Intelligence | Update SoC per driver using team constants and fuel mass | `energy_engine.py` |
| 4 | Intelligence | Compute CPI and EDR | `energy_engine.py` |
| 5 | Intelligence | Build features from a rolling window | `feature_engineer.py` |
| 6 | Intelligence | Predict clip probability and distance | `clip_model.py` |
| 7 | Intelligence | For cars within about 1 s, compute BAI | `battle_engine.py` |
| 8 | Intelligence | Evaluate alert thresholds | `alert_engine.py` |
| 9 | Storage | Persist alerts and lap summaries | `db_writer.py` |
| 10 | Application | Emit `race_tick`, `clip_alert`, `battle_update` | `socketio_server.py` |
| 11 | Presentation | Update map, leaderboard, battery bars, charts | React components |

The emitted `race_tick` is light (code, position, x, y, soc, cpi, risk). Full detail is sent only for the selected driver.

## 5. Module Responsibilities

### Backend

| Module | Responsibility |
|---|---|
| `app.py` | Flask init, Socket.IO init, config load, startup checks |
| `replay_reader.py` | Read parquet, serve ticks on a replay clock (play, pause, speed, seek) |
| `simulator.py` | Generate synthetic ticks from per-circuit speed, throttle and brake profiles |
| `energy_engine.py` | SoC update, fuel mass correction, CPI, EDR |
| `clip_detector.py` | Observed clipping labels from telemetry (used offline and for live display) |
| `feature_engineer.py` | Shared feature builder (same code for training and inference) |
| `clip_model.py` | Load model, return probability and distance to clip |
| `battle_engine.py` | Find attacker and defender pairs, compute BAI and outcome |
| `injection_engine.py` | Apply safety car, rain, harvest failure, cooling fault, rival attack, low grip |
| `alert_engine.py` | Map CPI and model output to GREEN, AMBER, RED, CRITICAL and recommended action |
| `db.py`, `db_writer.py` | Connection, table init, inserts |
| `routes.py` | REST endpoints |
| `socketio_server.py` | Socket.IO events and the tick broadcaster |

### Frontend

| Component | Responsibility |
|---|---|
| `Dashboard.jsx` | Root, socket state, mode and view routing |
| `RaceMap.jsx` | Track outline from X, Y and 22 car markers |
| `Leaderboard.jsx` | Positions, battery mini bars, risk colour |
| `DriverPanel.jsx` | Deep dive for the selected driver |
| `BatteryBar.jsx` | Reusable SoC bar with state colour |
| `ClipRiskPanel.jsx` | Probability, location, recommended action |
| `EnergyCharts.jsx` | Speed, throttle, brake, SoC against distance |
| `BattleView.jsx` | Attacker versus defender comparison |
| `DriverCompare.jsx` | Two-driver overlay |
| `InjectionPanel.jsx` | Injection buttons and reset |
| `ReplayControls.jsx` | Play, pause, speed, lap slider |
| `AlertLog.jsx` | Alert history |
| `RaceSummary.jsx` | End-of-race energy story |

## 6. Data Contract (unified tick)

```json
{
  "t": 1234.25,
  "lap": 23,
  "driver": "VER",
  "team": "RBR",
  "distance": 3120.4,
  "speed": 312.0,
  "throttle": 100,
  "brake": 0,
  "gear": 8,
  "rpm": 11800,
  "x": 1204.3,
  "y": -532.8,
  "tyre_compound": "MEDIUM",
  "tyre_age": 14,
  "gap_ahead": 0.8,
  "gap_behind": 1.6
}
```

Mode B must emit exactly this shape. Field names are fixed so the intelligence layer never branches on mode.

## 7. Performance Decisions

- 22 drivers at 4 Hz is about 88 samples per second, well within a single Flask process.
- Energy and feature computation are vectorised per tick across all drivers.
- Model inference is batched (one call for all 22 drivers per tick).
- The Socket.IO payload for the full grid stays small; detail is requested per selected driver.
- PostgreSQL receives events and lap summaries only, not every tick.

## 8. Calibration and Validation Flow

1. Detect observed clipping events per race.
2. Fit team constants so estimated SoC is low before observed clips and healthy elsewhere.
3. Freeze constants into `config/teams.yaml`.
4. Train the classifier with race-grouped folds so unseen circuits are tested.
5. Report PR-AUC, lead time and false alarms per lap against the baselines.

## 9. Extension Points

| Change | Where |
|---|---|
| Add a new circuit | Run `ingest_fastf1.py` for the round; add a simulator profile |
| Add an injection type | New handler in `injection_engine.py`, new button in `InjectionPanel.jsx` |
| Swap the model | Replace the file behind `clip_model.py` |
| Real-time feed | New reader module implementing the same tick contract |