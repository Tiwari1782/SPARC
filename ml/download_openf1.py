"""
SPARC data downloader.
Pulls 2026 race data from the OpenF1 API and saves CSV files into ml/data/raw_csv/.
Run from the project root:  python ml/download_openf1.py
Safe to re-run: finished races are skipped.
"""
import sys
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://api.openf1.org/v1"
OUT = Path("ml/data/raw_csv")
YEAR = 2026
SESSION_NAME = "Race"        # change to "Qualifying" or "Sprint" for other sessions
CHUNK_MINUTES = 15           # telemetry is fetched in windows to keep each request small
PAUSE = 0.4                  # seconds between requests (stays under the free-tier limit)
FMT = "%Y-%m-%dT%H:%M:%S"


def fetch(url):
    """GET with retries. Returns a list of rows ([] when the API has no results)."""
    for attempt in range(6):
        try:
            r = requests.get(url, timeout=90)
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code == 200:
            time.sleep(PAUSE)
            return r.json()
        if r.status_code == 404:          # OpenF1 answers 404 when a query has no rows
            return []
        time.sleep(2 * (attempt + 1))     # 429 or temporary server errors: back off
    raise RuntimeError(f"Request failed after retries: {url}")


def query(endpoint, **filters):
    qs = "&".join(f"{k}={v}" for k, v in filters.items())
    return fetch(f"{BASE}/{endpoint}?{qs}")


def windowed(endpoint, session_key, driver, start, end):
    """Fetch a high-rate endpoint (car_data, location) in time windows."""
    rows = []
    step = pd.Timedelta(minutes=CHUNK_MINUTES)
    t = start
    while t < end:
        t2 = min(t + step, end)
        url = (
            f"{BASE}/{endpoint}?session_key={session_key}&driver_number={driver}"
            f"&date>={t.strftime(FMT)}&date<{t2.strftime(FMT)}"
        )
        rows += fetch(url)
        t = t2
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    sessions = pd.DataFrame(query("sessions", year=YEAR, session_name=SESSION_NAME))
    if sessions.empty:
        sys.exit(f"No {SESSION_NAME} sessions found for {YEAR}. Check openf1.org for availability.")
    sessions.to_csv(OUT / "sessions.csv", index=False)
    print(f"[INFO] Found {len(sessions)} {SESSION_NAME} sessions for {YEAR}")

    for _, s in sessions.iterrows():
        sk = int(s["session_key"])
        tag = f"{YEAR}_{s['circuit_short_name']}_{SESSION_NAME}".replace(" ", "_")
        folder = OUT / tag
        folder.mkdir(exist_ok=True)
        if (folder / "DONE").exists():
            print(f"[SKIP] {tag} already downloaded")
            continue

        print(f"[INFO] Downloading {tag} (session_key={sk})")
        start = pd.to_datetime(s["date_start"], utc=True).tz_localize(None)
        end = pd.to_datetime(s["date_end"], utc=True).tz_localize(None)

        # Small tables: one request each
        for ep in ["drivers", "laps", "stints", "pit", "weather",
                   "position", "intervals", "race_control"]:
            pd.DataFrame(query(ep, session_key=sk)).to_csv(folder / f"{ep}.csv", index=False)

        drivers = pd.read_csv(folder / "drivers.csv")
        numbers = [int(d) for d in drivers["driver_number"].dropna().unique()]

        # Big tables: per driver, in time windows
        for ep in ["car_data", "location"]:
            frames = []
            for d in numbers:
                rows = windowed(ep, sk, d, start, end)
                if rows:
                    frames.append(pd.DataFrame(rows))
                print(f"       {ep}: driver {d} done ({len(rows)} rows)")
            if frames:
                pd.concat(frames, ignore_index=True).to_csv(folder / f"{ep}.csv", index=False)

        (folder / "DONE").write_text("ok")
        print(f"[INFO] Finished {tag}")


if __name__ == "__main__":
    main()
