"""
SPARC  --  ingest_fastf1.py
Loads each 2026 Race session via FastF1, resamples all 22 drivers to 4 Hz,
adds gap_ahead / gap_behind, tyre info, and saves a parquet file per race.

Run from the project root:
    python ml/ingest_fastf1.py

Safe to re-run: races that already have a parquet file are skipped.
FastF1 caches raw data in  ml/data/fastf1_cache/  so re-runs are instant.
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import fastf1
except ModuleNotFoundError:
    sys.exit("FastF1 not installed. Run:  pip install fastf1")

# -- Config ------------------------------------------------------------------
YEAR        = 2026
SESSION     = "Race"
CACHE_DIR   = Path("ml/data/fastf1_cache")
OUT_DIR     = Path("ml/data/parquet")
LOG_FILE    = OUT_DIR / "ingest.log"
FREQ        = "250ms"   # 4 Hz grid
# ----------------------------------------------------------------------------

OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def resample_driver(car: pd.DataFrame, pos: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample one driver's telemetry + position to a uniform time grid."""
    # car has: Speed, Throttle, Brake, Gear, RPM  indexed by SessionTime
    # pos has: X, Y, Z                            indexed by SessionTime
    car = car[["Speed", "Throttle", "Brake", "Gear", "RPM"]].copy()
    pos = pos[["X", "Y"]].copy()

    # Combine and resample
    combined = car.join(pos, how="outer").sort_index()
    combined = combined.resample(freq).mean().interpolate("time")
    combined.columns = [c.lower() for c in combined.columns]
    return combined


def build_gaps(all_drivers: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Add gap_ahead and gap_behind for each driver at every tick."""
    # Use a single shared time index (union of all)
    shared_idx = sorted(set().union(*[df.index for df in all_drivers.values()]))
    shared_idx = pd.DatetimeIndex(shared_idx)

    # Align all drivers to the shared index
    aligned = {
        drv: df.reindex(shared_idx, method="nearest", tolerance=pd.Timedelta("300ms"))
        for drv, df in all_drivers.items()
    }

    drivers = list(aligned.keys())
    n = len(drivers)

    for i, drv in enumerate(drivers):
        ahead = aligned[drivers[(i - 1) % n]]["distance"] if n > 1 else None
        behind = aligned[drivers[(i + 1) % n]]["distance"] if n > 1 else None
        d = aligned[drv]["distance"] if "distance" in aligned[drv].columns else 0
        aligned[drv]["gap_ahead"] = (ahead - d).clip(lower=0) if ahead is not None else np.nan
        aligned[drv]["gap_behind"] = (d - behind).clip(lower=0) if behind is not None else np.nan

    return aligned


def process_session(event_name: str, round_number: int) -> Path | None:
    tag = f"{YEAR}_{event_name.replace(' ', '_')}_{SESSION}"
    out_path = OUT_DIR / f"{tag}.parquet"

    if out_path.exists():
        log.info("SKIP %s  (parquet already exists)", tag)
        return out_path

    log.info("Loading %s via FastF1 ...", tag)
    try:
        session = fastf1.get_session(YEAR, round_number, SESSION)
        session.load(telemetry=True, laps=True, weather=True, messages=True)
    except Exception as exc:
        log.warning("Could not load %s: %s", tag, exc)
        return None

    if session.laps.empty:
        log.info("SKIP %s  (no lap data — cancelled or future)", tag)
        return None

    drivers = session.drivers
    log.info("  %d drivers found", len(drivers))

    frames = []
    for drv in drivers:
        try:
            laps = session.laps.pick_driver(drv)
            if laps.empty:
                continue

            # Get full telemetry for this driver
            tel = laps.get_telemetry(frequency=FREQ).copy()
            if tel.empty:
                continue

            tel = tel.reset_index(drop=True)

            # Add lap-level context: tyre compound, tyre age, lap number
            lap_map = laps[["LapNumber", "Compound", "TyreLife", "DriverAhead",
                             "DistanceToDriverAhead"]].copy()

            # Build a per-row lookup using LapNumber from telemetry
            if "LapNumber" not in tel.columns and "SessionTime" in tel.columns:
                tel["LapNumber"] = np.nan  # will be filled below

            # Merge lap context onto telemetry by LapNumber
            tel = tel.merge(
                lap_map.rename(columns={
                    "Compound": "tyre_compound",
                    "TyreLife": "tyre_age",
                }),
                on="LapNumber", how="left"
            )

            tel["driver"] = drv
            tel["team"]   = session.get_driver(drv)["TeamName"] if drv in session.drivers else ""

            # Rename columns to lowercase contract
            tel.rename(columns={
                "Speed":      "speed",
                "Throttle":   "throttle",
                "Brake":      "brake",
                "Gear":       "gear",
                "RPM":        "rpm",
                "X":          "x",
                "Y":          "y",
                "Distance":   "distance",
                "LapNumber":  "lap",
                "SessionTime":"t",
            }, inplace=True)

            # Keep only contract columns (extras are fine, will be dropped later)
            frames.append(tel)
            log.info("    driver %s: %d rows", drv, len(tel))

        except Exception as exc:
            log.warning("    driver %s failed: %s", drv, exc)
            continue

    if not frames:
        log.warning("No telemetry for %s", tag)
        return None

    combined = pd.concat(frames, ignore_index=True)

    # Add weather columns (merge on session time proximity)
    try:
        weather = session.weather_data.copy()
        weather["t_sec"] = weather["Time"].dt.total_seconds()
        combined["t_sec"] = combined["t"].dt.total_seconds() if hasattr(combined["t"], "dt") else combined["t"]
        combined = pd.merge_asof(
            combined.sort_values("t_sec"),
            weather[["t_sec", "AirTemp", "TrackTemp", "Rainfall", "WindSpeed"]].rename(
                columns={"AirTemp": "air_temp", "TrackTemp": "track_temp",
                         "Rainfall": "rain", "WindSpeed": "wind_speed"}
            ).sort_values("t_sec"),
            on="t_sec", direction="nearest"
        )
        combined.drop(columns=["t_sec"], inplace=True, errors="ignore")
    except Exception as exc:
        log.warning("  Weather merge failed: %s", exc)

    combined["year"]  = YEAR
    combined["event"] = event_name

    combined.to_parquet(out_path, index=False, compression="snappy")
    log.info("DONE %s  ->  %s  (%d rows, %.1f MB)",
             tag, out_path.name, len(combined), out_path.stat().st_size / 1e6)
    return out_path


def main():
    log.info("=== SPARC FastF1 Ingest  year=%d ===", YEAR)

    schedule = fastf1.get_event_schedule(YEAR, include_testing=False)
    races = schedule[schedule["EventFormat"] != "testing"].reset_index(drop=True)
    log.info("Found %d events in the %d schedule", len(races), YEAR)

    done, skipped, failed = 0, 0, 0
    for _, row in races.iterrows():
        name   = row["EventName"]
        rnd    = int(row["RoundNumber"])
        result = process_session(name, rnd)
        if result is None:
            failed += 1
        elif "SKIP" in str(result):
            skipped += 1
        else:
            done += 1

    log.info("=== Finished: %d done, %d skipped, %d failed/future ===",
             done, skipped, failed)


if __name__ == "__main__":
    main()
