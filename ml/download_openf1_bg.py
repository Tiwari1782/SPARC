"""
SPARC background downloader  -  runs fully unattended, logs to ml/data/raw_csv/download.log
Run once from the project root:
    python ml/download_openf1_bg.py
Safe to re-run: folders that already contain a DONE file are skipped automatically.
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
PAUSE           = 1.0  # seconds between requests (free-tier safe)
FMT             = "%Y-%m-%dT%H:%M:%S"

_rate_wait      = 30   # current 429 back-off (grows on every hit, resets on success)
LOG_FILE      = OUT / "download.log"
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


# -- HTTP helpers ------------------------------------------------------------

def fetch(url: str, big: bool = False) -> list:
    """GET with retries. _rate_wait persists across calls so 429 back-off compounds."""
    global _rate_wait
    for attempt in range(10):
        try:
            r = requests.get(url, timeout=90)
        except requests.RequestException as exc:
            log.warning("Network error (attempt %d): %s", attempt + 1, exc)
            time.sleep(5 * (attempt + 1))
            continue

        if r.status_code == 200:
            _rate_wait = 30          # reset on success
            time.sleep(PAUSE_BIG if big else PAUSE)
            return r.json()
        if r.status_code == 404:
            return []                # no rows -- normal for cancelled / future races
        if r.status_code == 429:
            log.warning("Rate-limited (429). Waiting %ds ...", _rate_wait)
            time.sleep(_rate_wait)
            _rate_wait = min(_rate_wait * 2, 300)  # double each time, cap at 5 min
            continue

        log.warning("HTTP %s (attempt %d): %s", r.status_code, attempt + 1, url)
        time.sleep(5 * (attempt + 1))

    raise RuntimeError(f"Request failed after retries: {url}")


def query(endpoint: str, **filters) -> list:
    qs = "&".join(f"{k}={v}" for k, v in filters.items())
    return fetch(f"{BASE}/{endpoint}?{qs}")





# -- Main --------------------------------------------------------------------

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # Load sessions from the pre-existing CSV (no extra API call)
    if SESSIONS_CSV.exists():
        sessions = pd.read_csv(SESSIONS_CSV)
        log.info("Loaded %d sessions from local CSV", len(sessions))
    else:
        log.info("Fetching session list from OpenF1 ...")
        data = query("sessions", year=YEAR, session_name=SESSION_NAME)
        if not data:
            sys.exit(f"No {SESSION_NAME} sessions found for {YEAR}.")
        sessions = pd.DataFrame(data)
        sessions.to_csv(SESSIONS_CSV, index=False)
        log.info("Saved sessions.csv (%d rows)", len(sessions))

    total = len(sessions)
    for idx, s in sessions.iterrows():
        sk  = int(s["session_key"])
        tag = f"{YEAR}_{s['circuit_short_name']}_{SESSION_NAME}".replace(" ", "_")
        folder = OUT / tag
        folder.mkdir(exist_ok=True)

        if (folder / "DONE").exists():
            log.info("[%d/%d] SKIP %s", idx + 1, total, tag)
            continue

        log.info("[%d/%d] START %s  (session_key=%d)", idx + 1, total, tag, sk)

        # Cancelled races have no data -- mark done and move on
        cancelled = str(s.get("is_cancelled", "False")).lower() == "true"
        if cancelled:
            (folder / "DONE").write_text("cancelled")
            log.info("[%d/%d] DONE  %s  (cancelled)", idx + 1, total, tag)
            continue

        # 8 lightweight tables: one request each, no rate-limit issues
        for ep in ["drivers", "laps", "stints", "pit", "weather",
                   "position", "intervals", "race_control"]:
            rows = query(ep, session_key=sk)
            pd.DataFrame(rows).to_csv(folder / f"{ep}.csv", index=False)
            log.info("  %-15s  %d rows", ep, len(rows))

        (folder / "DONE").write_text("ok")
        log.info("[%d/%d] DONE  %s", idx + 1, total, tag)

    log.info("All sessions processed.")


if __name__ == "__main__":
    main()
