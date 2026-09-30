# SPARC Setup Guide

Step-by-step instructions to install, process 2026 data, train the model and run SPARC locally. Tested on the assumption of a laptop with an Intel i5 and 8 to 16 GB RAM, using VS Code. No GPU is required.

---

## 1. Prerequisites

| Tool | Version | Check |
|---|---|---|
| Python | 3.11 or newer | `python --version` |
| Node.js | 18 or newer | `node --version` |
| Git | any recent | `git --version` |
| PostgreSQL | 16 (local or Docker) | `psql --version` |
| Docker (optional) | any recent | `docker --version` |

Recommended VS Code extensions: Python, Pylance, ESLint, ES7+ React snippets.

Free disk space: keep at least 10 GB free while ingesting data (the raw cache is deleted after each race is processed).

## 2. Clone

```bash
git clone https://github.com/Tiwari1782/SPARC.git
cd SPARC
```

## 3. Backend Setup

```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate
# Linux or Mac
source venv/bin/activate

pip install -r ../requirements.txt
cp ../.env.example ../.env
```

Suggested `requirements.txt`:

```
flask>=3.0
flask-socketio>=5.3
flask-cors
eventlet
psycopg2-binary
python-dotenv
pandas
numpy
pyarrow
scikit-learn
lightgbm
joblib
fastf1
pyyaml
```

## 4. Environment Variables

Edit `.env` in the project root.

```env
# PostgreSQL
DATABASE_URL=postgresql://sparc_user:yourpassword@localhost:5432/sparc_db

# Flask
FLASK_ENV=development
FLASK_SECRET_KEY=change-me

# Replay and simulation
DEFAULT_MODE=replay            # replay or simulation
TICK_HZ=4
REPLAY_SPEED=1.0

# Paths
FASTF1_CACHE=data/raw/cache
PROCESSED_DIR=data/processed
MODEL_PATH=ml/sparc_model.pkl
REGS_CONFIG=config/regs_2026.yaml
TEAMS_CONFIG=config/teams.yaml

# Season
SEASON=2026
```

Never commit `.env`. It is listed in `.gitignore`.

## 5. Database Setup

Option A: Docker (recommended)

```bash
docker-compose up -d postgres
```

Option B: local PostgreSQL

```bash
psql -U postgres -c "CREATE DATABASE sparc_db;"
psql -U postgres -c "CREATE USER sparc_user WITH PASSWORD 'yourpassword';"
psql -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE sparc_db TO sparc_user;"
```

Tables are created automatically when the backend starts.

## 6. Regulation Config

Open `config/regs_2026.yaml` and fill in the values from the current FIA technical regulations. Cite the document and date in a comment.

```yaml
# Source: FIA 2026 Formula 1 Technical Regulations (verify version and date)
battery:
  usable_energy_mj: null        # fill from regulations
deployment:
  max_power_kw: null            # fill from regulations
  speed_taper: null             # fill from regulations
recovery:
  max_per_lap_mj: null          # fill from regulations
fuel:
  start_mass_kg: 100            # approximate, verify
  burn_rate_kg_per_lap: null    # estimate from race distance
```

Do not calibrate until these values are filled and checked.

## 7. Ingest 2026 Data

First, see which rounds have run:

```bash
python -c "import fastf1; print(fastf1.get_event_schedule(2026)[['RoundNumber','EventName']])"
```

Process one race first:

```bash
cd ml
python ingest_fastf1.py --season 2026 --round 1 --session R
```

Minimal ingest outline for `ml/ingest_fastf1.py`:

```python
import argparse, shutil, pandas as pd, fastf1

p = argparse.ArgumentParser()
p.add_argument("--season", type=int, required=True)
p.add_argument("--round", type=int, required=True)
p.add_argument("--session", default="R")
a = p.parse_args()

fastf1.Cache.enable_cache("../data/raw/cache")
s = fastf1.get_session(a.season, a.round, a.session)
s.load(telemetry=True, weather=True, messages=False)

frames = []
for drv in s.drivers:
    laps = s.laps.pick_drivers(drv)
    for _, lap in laps.iterlaps():
        tel = lap.get_telemetry()[["Time","Distance","Speed","Throttle","Brake",
                                   "nGear","RPM","X","Y"]].copy()
        tel["driver"] = lap["Driver"]
        tel["team"] = lap["Team"]
        tel["lap"] = lap["LapNumber"]
        tel["compound"] = lap["Compound"]
        tel["tyre_age"] = lap["TyreLife"]
        frames.append(tel)

df = pd.concat(frames)
# Resample every driver to a common 4 Hz grid here, add gap_ahead and gap_behind,
# cast to float32, then save.
df.to_parquet(f"../data/processed/{a.season}_R{a.round:02d}_{a.session}.parquet", index=False)
shutil.rmtree("../data/raw/cache", ignore_errors=True)   # optional cleanup
```

The first load per session is slow (often several minutes). After that the cache is reused. Once one race works, run the rest in a batch and leave it overnight.

Batch example:

```bash
for r in 1 2 3 4 5 6; do python ingest_fastf1.py --season 2026 --round $r --session R; done
```

## 8. Build Labels and Calibrate

```bash
python build_labels.py          # detect observed clipping events
python calibrate_teams.py       # fit per-team constants into config/teams.yaml
```

Open a few detected clips against speed traces and confirm they look like real clipping before moving on.

## 9. Train the Model

```bash
python train_model.py           # trains, saves ml/sparc_model.pkl
python evaluate.py              # PR-AUC, lead time, false alarms, baselines, ablations
```

Training uses the CPU only. A few hundred thousand rows train in seconds to a few minutes. Use `n_jobs=-1` for Random Forest to use all cores. Splits are always grouped by race.

## 10. Run the Backend

```bash
cd backend
python app.py
```

Expected output:

```
[INFO] PostgreSQL connected
[INFO] Model loaded: sparc_model.pkl
[INFO] Processed races found: 6
[INFO] SPARC backend running on http://localhost:5000
```

## 11. Run the Frontend

```bash
cd frontend
npm install
npm install react-icons recharts socket.io-client
npm run dev
```

Open `http://localhost:3000`.

## 12. Verify the Install

| Check | How | Expected |
|---|---|---|
| API reachable | `GET http://localhost:5000/api/races` | List of processed rounds |
| Stream works | Open dashboard, press play | 22 cars move, battery bars change |
| Mode switch | Toggle to simulation | Same UI, synthetic data |
| Injection | Press Safety car | CPI changes across the field |
| Alerts stored | `GET /api/alerts` | Recent alerts returned |

## 13. Docker (optional full stack)

```bash
docker-compose up --build
```

## 14. Troubleshooting

| Problem | Likely cause and fix |
|---|---|
| FastF1 load is very slow | Normal on first run; leave it running, then rely on the cache |
| Out of memory during ingest | Process one race at a time, cast to `float32`, write parquet immediately |
| Drivers do not line up in the race view | Resample everyone onto the common 4 Hz grid before saving |
| Model score looks too good | You probably split randomly; use GroupKFold by race |
| SoC stays at 0 or 100 | Constants not calibrated, or `regs_2026.yaml` values missing |
| No clip events detected | Check the expected-acceleration envelope and minimum sustained distance |
| CORS errors in browser | Enable `flask-cors` for `http://localhost:3000` |
| Socket.IO not connecting | Confirm backend port 5000 and matching client URL |
| Icons missing | `npm install react-icons` and import from `react-icons/fa6` |

## 15. Git Hygiene

```bash
git checkout -b feat/energy-engine
git add .
git commit -m "feat(energy): add SoC update with per-team constants"
```

Keep `data/` and `.env` out of Git. Commit `ml/sparc_model.pkl` only if it is small.