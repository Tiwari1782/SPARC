"""
SPARC telemetry downloader  --  car_data + location only.
Run OVERNIGHT from the project root:
    python ml/download_telemetry.py

- Only processes race folders that have a DONE file but are missing car_data.csv
- Very conservative rate limiting (5s between window chunks, 15s between drivers)
- Automatically skips future races (0-row sessions) and cancelled races
- Safe to kill and re-run: per-driver progress is checkpointed
"""
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import requests

# -- Config ------------------------------------------------------------------
BASE          = "https://api.openf1.org/v1"
OUT           = Path("ml/data/raw_csv")
SESSIONS_CSV  = OUT / "sessions.csv"
YEAR          = 2026
SESSION_NAME  = "Race"
CHUNK_MINUTES = 30    # larger window = fewer requests per driver (was 15)
PAUSE_CHUNK   = 5.0   # seconds between window chunks (very conservative)
PAUSE_DRIVER  = 15.0  # seconds between drivers
PAUSE_EP      = 30.0  # seconds between car_data and location passes
FMT           = "%Y-%m-%dT%H:%M:%S"
LOG_FILE      = OUT / "telemetry.log"
# Only download races up to and including Baku (last real 2026 race so far)
CUTOFF_DATE   = pd.Timestamp("2026-09-27", tz="UTC")  # day after Baku
# ----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

_rate_wait = 60  # 429 back-off: starts at 60s, doubles each hit, resets on success


def fetch(url: str) -> list:
    global _rate_wait
    for attempt in range(10):
        try:
            r = requests.get(url, timeout=120)
        except requests.RequestException as exc:
            log.warning("Network error (attempt %d): %s", attempt + 1, exc)
            time.sleep(10 * (attempt + 1))
            continue

        if r.status_code == 200:
            _rate_wait = 60          # reset on success
            time.sleep(PAUSE_CHUNK)
            return r.json()
        if r.status_code == 404:
            return []
        if r.status_code == 429:
            log.warning("Rate-limited (429). Waiting %ds ...", _rate_wait)
            time.sleep(_rate_wait)
            _rate_wait = min(_rate_wait * 2, 600)  # double, cap at 10 min
            continue

        log.warning("HTTP %s (attempt %d): %s", r.status_code, attempt + 1, url)
        time.sleep(10 * (attempt + 1))

    raise RuntimeError(f"Request failed after retries: {url}")


def windowed(endpoint: str, session_key: int, driver: int, start, end) -> list:
    rows, step, t = [], pd.Timedelta(minutes=CHUNK_MINUTES), start
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
    if not SESSIONS_CSV.exists():
        sys.exit("sessions.csv not found. Run download_openf1_bg.py first.")

    sessions = pd.read_csv(SESSIONS_CSV)
    log.info("Loaded %d sessions", len(sessions))

    for _, s in sessions.iterrows():
        sk  = int(s["session_key"])
        tag = f"{YEAR}_{s['circuit_short_name']}_{SESSION_NAME}".replace(" ", "_")
        folder = OUT / tag

        # Skip future races — only download up to and including Baku
        race_start = pd.to_datetime(s["date_start"], utc=True)
        if race_start >= CUTOFF_DATE:
            log.info("SKIP %s  (future race — after Baku cutoff)", tag)
            continue

        # Only process races that finished downloading (have DONE file)
        if not (folder / "DONE").exists():
            log.info("SKIP %s  (no DONE file)", tag)
            continue

        # Skip if car_data already exists with real data (Melbourne/Shanghai/Suzuka etc.)
        car_csv = folder / "car_data.csv"
        if car_csv.exists() and car_csv.stat().st_size > 1000:
            log.info("SKIP %s  (car_data already downloaded)", tag)
            continue

        # Use laps.csv as the real guard: cancelled + future races have 0 laps
        # even though drivers.csv may still list 22 entries
        laps_csv = folder / "laps.csv"
        if not laps_csv.exists():
            log.info("SKIP %s  (no laps.csv)", tag)
            continue
        try:
            laps_df = pd.read_csv(laps_csv)
        except Exception:
            laps_df = pd.DataFrame()
        if laps_df.empty:
            log.info("SKIP %s  (0 laps — cancelled or future race)", tag)
            continue

        drv_csv = folder / "drivers.csv"
        if not drv_csv.exists():
            log.info("SKIP %s  (no drivers.csv)", tag)
            continue
        drivers_df = pd.read_csv(drv_csv)
        if drivers_df.empty:
            log.info("SKIP %s  (empty drivers.csv)", tag)
            continue

        log.info("START %s  (session_key=%d)", tag, sk)
        start = pd.to_datetime(s["date_start"], utc=True).tz_localize(None)
        end   = pd.to_datetime(s["date_end"],   utc=True).tz_localize(None)
        numbers = [int(d) for d in drivers_df["driver_number"].dropna().unique()]

        for ep in ["car_data", "location"]:
            ep_csv = folder / f"{ep}.csv"
            if ep_csv.exists() and ep_csv.stat().st_size > 1000:
                log.info("  SKIP %s (already exists)", ep)
                continue

            log.info("  Fetching %s for %d drivers ...", ep, len(numbers))
            frames = []
            for i, d in enumerate(numbers):
                log.info("    driver %d  (%d/%d)", d, i + 1, len(numbers))
                rows = windowed(ep, sk, d, start, end)
                if rows:
                    frames.append(pd.DataFrame(rows))
                time.sleep(PAUSE_DRIVER)   # polite pause between drivers

            if frames:
                merged = pd.concat(frames, ignore_index=True)
                merged.to_csv(ep_csv, index=False)
                log.info("  DONE %s  %d rows", ep, len(merged))
            else:
                log.info("  %s  no data", ep)

            time.sleep(PAUSE_EP)   # breathe between car_data and location

        log.info("DONE %s", tag)

    log.info("All telemetry processed.")


if __name__ == "__main__":
    main()
