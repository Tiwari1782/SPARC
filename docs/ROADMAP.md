# SPARC Roadmap

A phased plan. Each phase has a deliverable and an exit check, so you know when to move on. Durations are estimates for one to four people and can be stretched or compressed.

Rule for the whole project: get **one circuit fully working for all 22 cars** before scaling to more circuits.

---

## Phase 0: Foundations (Week 1)

- Create the GitHub repo, branches, issue labels and project board
- Set up backend and frontend skeletons (Flask, Socket.IO, React with Vite)
- Add `react-icons` and establish the no-emoji rule
- Verify the 2026 calendar with `fastf1.get_event_schedule(2026)`
- Read the current FIA technical regulations for battery capacity, deployment power and per-lap recovery limits; record them in `config/regs_2026.yaml`

**Exit check:** backend and frontend run, one FastF1 session loads, regulation values are written down with sources.

## Phase 1: Data Pipeline (Weeks 1 to 2)

- Write `ingest_fastf1.py`: load a session, resample all drivers to a 4 Hz grid, add gaps and tyre data, save parquet, clear raw cache
- Process **one race** fully (all 22 drivers)
- Define and freeze the unified tick contract
- Write `replay_reader.py` with play, pause, speed and seek

**Exit check:** one race replays from parquet with all 22 drivers aligned on the same clock.

## Phase 2: Energy Engine (Weeks 2 to 3)

- Implement `energy_engine.py`: SoC update, speed taper, fuel mass correction
- Implement CPI, EDR and BAI
- Use conservative starting constants and unit-test basic behaviour (braking raises SoC, full throttle lowers it, SoC stays within 0 to 100)
- Plot SoC against distance for several drivers to sanity check

**Exit check:** SoC traces look physically sensible over a full race for the processed circuit.

## Phase 3: Ground Truth and Calibration (Weeks 3 to 4)

- Implement `clip_detector.py` using the expected-acceleration-envelope method
- Manually inspect a sample of detected clips against speed traces
- Fit per-team constants in `calibrate_teams.py`
- Check that estimated SoC approaches empty at observed clips

**Exit check:** observed clips are detected reliably and calibrated SoC agrees with them on the training race.

## Phase 4: ML Model (Weeks 4 to 5)

- Build `feature_engineer.py` (shared for training and inference)
- Train baselines (threshold rule, logistic regression) and the main model (LightGBM or Random Forest)
- Use GroupKFold by race; once more races are processed, hold out whole circuits
- Report PR-AUC, lead time and false alarms per lap
- Run ablations: without rival term, fuel load, tyre age

**Exit check:** the main model beats the baselines on held-out races and lead time is meaningful.

## Phase 5: Backend Integration (Weeks 5 to 6)

- Wire replay, engine, model, battle engine and alert engine into the tick loop
- Implement `socketio_server.py` events and REST routes
- Add PostgreSQL tables and `db_writer.py`
- Build `simulator.py` (Mode B) from per-circuit profiles in the real data
- Implement `injection_engine.py` for all six injections

**Exit check:** both modes stream the same tick shape and every injection changes CPI in the expected direction.

## Phase 6: Dashboard (Weeks 6 to 7)

- Race map with 22 markers and leaderboard with battery bars
- Driver deep dive with charts and clip-risk panel
- Energy battle view and driver compare
- Injection panel, replay controls, alert log
- Apply the icon mapping from `icons.md`

**Exit check:** full demo flow works end to end, including a live injection on all 22 cars.

## Phase 7: Scale Out (Weeks 7 to 8)

- Process the remaining 2026 rounds in a batch (overnight run, then cached)
- Re-fit team constants with more data
- Re-run validation with whole circuits held out
- Race energy story summary page

**Exit check:** multiple circuits available in the dashboard, results reported across circuits.

## Phase 8: Polish and Write-up (Weeks 8 to 9)

- README, screenshots and demo script
- Docker Compose for one-command start
- Paper or report draft: problem, method, indices, validation, limitations
- Literature search to confirm novelty claims for CPI, EDR and BAI

**Exit check:** a stranger can clone, follow `setup.md`, and run the demo; the write-up is ready for review.

---

## Demo Script (5 minutes)

| Time | Action | What the audience sees |
|---|---|---|
| 0:00 to 0:40 | Open Race view | 22 cars on the map, battery bars on the leaderboard |
| 0:40 to 1:40 | Start replay | Bars rise under braking and fall on straights |
| 1:40 to 2:30 | Open a driver deep dive | SoC, clip-risk panel and a detected clip point |
| 2:30 to 3:30 | Open Energy Battle | Attacker versus defender, BAI, predicted outcome |
| 3:30 to 4:30 | Inject a safety car, then a harvest failure | Strategy shifts across the field, one car turns red |
| 4:30 to 5:00 | Show race energy story | Who managed energy best and where places were lost |

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| FastF1 loading is slow | Batch script, run overnight, rely on the cache |
| SoC cannot be verified directly | Validate through observable clipping and say so openly |
| Clip labels contain false positives | Manual inspection sample, sustained-distance rule, drag-limit exclusion |
| Regulation values uncertain | Keep all in `regs_2026.yaml`, cite the source, re-check before calibration |
| Scope creep | Finish one circuit for 22 cars first; scale only after the exit check |
| Laptop memory | Parquet, `float32`, process one race at a time |

## Stretch Goals

- Live data feed via OpenF1
- Sequence model trained on Colab
- Strategy recommender that suggests where to save energy ahead of a predicted clip
- Public read-only deployment of the dashboard