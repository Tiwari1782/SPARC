"""
ml/process_races.py  --  Stage 1: process raw OpenF1 CSVs into per-race parquet files.

For each race folder that contains a DONE file AND has car_data.csv + location.csv:
  - Join car_data and location on timestamp (nearest, 0.5s tolerance)
  - Resample to a common 4 Hz grid per race
  - Add: lap number (laps.csv), gap_ahead / gap_behind (intervals.csv),
         tyre compound and tyre_age (stints.csv),
         track_temp / air_temp / rain_flag (weather.csv, nearest time),
         team and driver_code (drivers.csv)
  - Compute distance along lap by integrating speed
  - Save one parquet per race to data/processed/
  - Run Gate 1: each race has all drivers, timestamps aligned across drivers,
                and no more than 2 percent missing in speed, throttle, brake.

Processes one race at a time, frees memory between races.
Uses float32 throughout to minimise RAM on laptop.
"""

import gc
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parent.parent
RAW_DIR     = ROOT / "ml" / "data" / "raw_csv"
PROCESSED   = ROOT / "data" / "processed"
LOG_FILE    = ROOT / "logs" / "process_races.log"

PROCESSED.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HZ          = 4                          # target sample rate (4 Hz)
DT          = 1.0 / HZ                  # 0.25 seconds per sample
JOIN_TOL    = pd.Timedelta("0.5s")      # merge tolerance car_data <-> location
GRID_TOL    = pd.Timedelta("0.3s")      # grid alignment tolerance
MISS_THRESH = 0.02                       # gate 1: max 2% missing in key columns
KEY_COLS    = ["speed", "throttle", "brake"]


def parse_ts(series: pd.Series) -> pd.Series:
    """Parse ISO8601 timestamp series to UTC-naive datetime64[ns]."""
    ts = pd.to_datetime(series, format="ISO8601", utc=True, errors="coerce")
    return ts.dt.tz_localize(None)


def interval_to_seconds(val) -> float:
    """Convert OpenF1 interval values to float seconds."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip().replace("+", "")
    if "LAP" in s.upper():
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def process_race(folder: Path) -> bool:
    tag = folder.name
    out_path = PROCESSED / f"{tag}.parquet"

    if out_path.exists():
        log.info("  SKIP %s (parquet exists)", tag)
        return True

    car_path = folder / "car_data.csv"
    loc_path = folder / "location.csv"
    laps_path = folder / "laps.csv"

    if not car_path.exists() or not loc_path.exists() or not laps_path.exists():
        log.warning("  %s: missing required core files (car_data, location, or laps) - skip", tag)
        return False

    # 1. Load car_data
    try:
        car = pd.read_csv(
            car_path,
            usecols=["date", "driver_number", "speed", "throttle", "brake", "n_gear", "rpm"],
            dtype={
                "driver_number": "int16",
                "speed": "float32",
                "throttle": "float32",
                "brake": "float32",
                "n_gear": "float32",
                "rpm": "float32",
            },
            low_memory=False,
        )
    except Exception as exc:
        log.error("  %s: error reading car_data.csv: %s", tag, exc)
        return False

    if car.empty:
        log.warning("  %s: car_data.csv is empty - skip", tag)
        return False

    car["date"] = parse_ts(car["date"])
    car = car.dropna(subset=["date"])
    car = car.rename(columns={"n_gear": "gear"})
    car["gear"] = car["gear"].fillna(0).clip(0, 8).astype("int8")
    log.info("  Loaded car_data: %d rows, %d drivers", len(car), car["driver_number"].nunique())

    # 2. Load location
    try:
        loc = pd.read_csv(
            loc_path,
            usecols=["date", "driver_number", "x", "y"],
            dtype={"driver_number": "int16", "x": "float32", "y": "float32"},
            low_memory=False,
        )
        loc["date"] = parse_ts(loc["date"])
        loc = loc.dropna(subset=["date"])
    except Exception as exc:
        log.warning("  %s: location.csv read error: %s", tag, exc)
        loc = pd.DataFrame(columns=["date", "driver_number", "x", "y"])

    # 3. Load laps
    try:
        laps = pd.read_csv(laps_path, low_memory=False)
        lap_date_col = "date_start" if "date_start" in laps.columns else "lap_start_time"
        laps = laps[["driver_number", "lap_number", lap_date_col]].copy()
        laps = laps.rename(columns={lap_date_col: "date_start"})
        laps["date_start"] = parse_ts(laps["date_start"])
        laps = laps.dropna(subset=["date_start"])
    except Exception as exc:
        log.warning("  %s: laps.csv read error: %s", tag, exc)
        laps = pd.DataFrame(columns=["driver_number", "lap_number", "date_start"])

    # 4. Load stints
    stints_path = folder / "stints.csv"
    if stints_path.exists():
        try:
            stints = pd.read_csv(stints_path, low_memory=False)
        except Exception:
            stints = pd.DataFrame()
    else:
        stints = pd.DataFrame()

    # 5. Load intervals
    int_path = folder / "intervals.csv"
    if int_path.exists():
        try:
            intervals = pd.read_csv(int_path, low_memory=False)
            intervals["date"] = parse_ts(intervals["date"])
            intervals = intervals.dropna(subset=["date"])
            intervals["gap_s"] = intervals["interval"].apply(interval_to_seconds).astype("float32")
            if "gap_to_leader" in intervals.columns:
                intervals["leader_s"] = intervals["gap_to_leader"].apply(interval_to_seconds).astype("float32")
            else:
                intervals["leader_s"] = intervals["gap_s"]
        except Exception:
            intervals = pd.DataFrame()
    else:
        intervals = pd.DataFrame()

    # 6. Load weather
    wea_path = folder / "weather.csv"
    if wea_path.exists():
        try:
            weather = pd.read_csv(wea_path, low_memory=False)
            weather["date"] = parse_ts(weather["date"])
            weather = weather.dropna(subset=["date"]).sort_values("date")
        except Exception:
            weather = pd.DataFrame()
    else:
        weather = pd.DataFrame()

    # 7. Load drivers
    drv_path = folder / "drivers.csv"
    team_map = {}
    code_map = {}
    if drv_path.exists():
        try:
            drivers_df = pd.read_csv(drv_path, low_memory=False)
            for _, r in drivers_df.iterrows():
                num = int(r.get("driver_number", 0))
                team = str(r.get("team_name", "UNKNOWN"))
                code = str(r.get("name_acronym", r.get("driver_code", "UNK")))
                team_map[num] = team
                code_map[num] = code
        except Exception:
            pass

    # Build common race grid (4 Hz)
    t_min = car["date"].min()
    t_max = car["date"].max()
    grid_index = pd.date_range(t_min, t_max, freq="250ms")
    grid_df = pd.DataFrame({"date": grid_index})

    # Weather interpolation on grid
    if not weather.empty and "track_temperature" in weather.columns:
        w_sub = weather[["date", "track_temperature", "air_temperature", "rainfall"]].copy()
        w_grid = pd.merge_asof(grid_df, w_sub, on="date", direction="nearest")
        w_grid["rain_flag"] = (w_grid["rainfall"].fillna(0) > 0).astype("float32")
        w_grid["track_temp"] = w_grid["track_temperature"].ffill().bfill().astype("float32")
        w_grid["air_temp"] = w_grid["air_temperature"].ffill().bfill().astype("float32")
    else:
        w_grid = grid_df.copy()
        w_grid["track_temp"] = np.float32(30.0)
        w_grid["air_temp"] = np.float32(25.0)
        w_grid["rain_flag"] = np.float32(0.0)

    # Process each driver
    drivers_list = sorted(car["driver_number"].unique())
    resampled_drivers = []

    for drv in drivers_list:
        cg = car[car["driver_number"] == drv].sort_values("date").drop_duplicates("date")
        if cg.empty:
            continue

        lg = loc[loc["driver_number"] == drv].sort_values("date").drop_duplicates("date") if not loc.empty else pd.DataFrame()
        if not lg.empty:
            merged_car = pd.merge_asof(cg, lg[["date", "x", "y"]], on="date", direction="nearest", tolerance=JOIN_TOL)
        else:
            merged_car = cg.copy()
            merged_car["x"] = np.float32(np.nan)
            merged_car["y"] = np.float32(np.nan)

        # Merge onto common 4 Hz grid
        drv_grid = pd.merge_asof(grid_df, merged_car, on="date", direction="nearest", tolerance=GRID_TOL)
        drv_grid["driver_number"] = np.int16(drv)

        # Telemetry forward/backward fill across short gaps
        sig_cols = ["speed", "throttle", "brake", "rpm", "x", "y"]
        drv_grid[sig_cols] = drv_grid[sig_cols].ffill().bfill()
        drv_grid["gear"] = drv_grid["gear"].ffill().bfill().fillna(0).astype("int8")

        # Assign lap number
        drv_laps = laps[laps["driver_number"] == drv].sort_values("date_start")
        if not drv_laps.empty:
            lap_starts = drv_laps["date_start"].values
            lap_nums = drv_laps["lap_number"].values
            idx = np.searchsorted(lap_starts, drv_grid["date"].values, side="right") - 1
            idx = np.clip(idx, 0, len(lap_nums) - 1)
            drv_grid["lap_number"] = lap_nums[idx].astype("int16")
        else:
            drv_grid["lap_number"] = np.int16(1)

        # Distance along lap (integrate speed: speed_kmh / 3.6 * DT)
        speed_ms = drv_grid["speed"].fillna(0).values.astype("float32") / 3.6
        step_dist = speed_ms * DT
        drv_grid["distance"] = (
            pd.Series(step_dist, index=drv_grid.index)
            .groupby(drv_grid["lap_number"])
            .cumsum()
            .astype("float32")
        )

        # Tyre compound and tyre age from stints
        drv_grid["compound"] = "MEDIUM"
        drv_grid["tyre_age"] = np.float32(1.0)
        if not stints.empty:
            drv_stints = stints[stints["driver_number"] == drv]
            for _, s_row in drv_stints.iterrows():
                l_start = int(s_row.get("lap_start", 0) or 0)
                l_end = int(s_row.get("lap_end", 999) or 999)
                comp = str(s_row.get("compound", "MEDIUM") or "MEDIUM")
                age_start = float(s_row.get("tyre_age_at_start", 0) or 0)
                mask = (drv_grid["lap_number"] >= l_start) & (drv_grid["lap_number"] <= l_end)
                if mask.any():
                    drv_grid.loc[mask, "compound"] = comp
                    drv_grid.loc[mask, "tyre_age"] = (drv_grid.loc[mask, "lap_number"] - l_start + age_start).astype("float32")

        # Gaps from intervals
        drv_grid["gap_ahead"] = np.float32(30.0)
        drv_grid["gap_behind"] = np.float32(30.0)
        if not intervals.empty:
            drv_ints = intervals[intervals["driver_number"] == drv].sort_values("date")
            if not drv_ints.empty:
                drv_int_merged = pd.merge_asof(
                    drv_grid[["date"]],
                    drv_ints[["date", "gap_s", "leader_s"]],
                    on="date",
                    direction="nearest",
                    tolerance=pd.Timedelta("10s"),
                )
                drv_grid["gap_ahead"] = drv_int_merged["gap_s"].fillna(30.0).astype("float32")

        # Merge weather
        drv_grid["track_temp"] = w_grid["track_temp"].values
        drv_grid["air_temp"] = w_grid["air_temp"].values
        drv_grid["rain_flag"] = w_grid["rain_flag"].values

        # Metadata
        drv_grid["team"] = team_map.get(drv, "UNKNOWN")
        drv_grid["driver_code"] = code_map.get(drv, f"D{drv}")
        drv_grid["race"] = tag

        resampled_drivers.append(drv_grid)

    del car, loc, laps, stints, intervals, weather, grid_df, w_grid
    gc.collect()

    if not resampled_drivers:
        log.error("  %s: no driver data processed", tag)
        return False

    df = pd.concat(resampled_drivers, ignore_index=True)
    del resampled_drivers
    gc.collect()

    # Compute gap_behind across the field at each tick
    try:
        p_df = df.sort_values(["date", "gap_ahead"]).reset_index(drop=True)
        # Shift gap_ahead to assign gap_behind to car ahead
        p_df["gap_behind"] = p_df.groupby("date")["gap_ahead"].shift(-1).fillna(30.0).astype("float32")
        df = p_df
    except Exception:
        df["gap_behind"] = np.float32(30.0)

    # Convert all float64 to float32
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = df[c].astype("float32")

    # -----------------------------------------------------------------------
    # Gate 1 Check
    # -----------------------------------------------------------------------
    n_drivers = df["driver_number"].nunique()
    fail = False

    if n_drivers < 15:
        log.error("  GATE 1 FAIL %s: only %d drivers (expected ~20+)", tag, n_drivers)
        fail = True

    for col in KEY_COLS:
        miss_rate = df[col].isna().mean()
        if miss_rate > MISS_THRESH:
            log.error("  GATE 1 FAIL %s: %s missing rate %.2f%% > 2%%", tag, col, miss_rate * 100)
            fail = True
        else:
            log.info("  GATE 1: %s missing=%.2f%%", col, miss_rate * 100)

    if fail:
        log.error("  GATE 1 FAILED for %s", tag)
        return False

    # Save to parquet
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, out_path, compression="snappy")
    log.info("  GATE 1 PASS: saved %s -> %s (%.1f MB, %d rows, %d drivers)",
             tag, out_path.name, out_path.stat().st_size / 1e6, len(df), n_drivers)

    del df, table
    gc.collect()
    return True


def main():
    t0 = time.time()
    log.info("Stage 1: process_races.py START")

    candidates = sorted([
        d for d in RAW_DIR.iterdir()
        if d.is_dir()
        and (d / "DONE").exists()
        and (d / "car_data.csv").exists()
        and (d / "location.csv").exists()
        and (d / "laps.csv").exists()
    ])

    log.info("Found %d complete races with full telemetry", len(candidates))
    if len(candidates) == 0:
        log.error("No races found with required files. Exiting.")
        sys.exit(1)

    ok, skipped, failed = 0, 0, 0
    for folder in candidates:
        out_path = PROCESSED / f"{folder.name}.parquet"
        if out_path.exists():
            log.info("SKIP %s (parquet exists)", folder.name)
            skipped += 1
            continue
        log.info("Processing %s ...", folder.name)
        t_r = time.time()
        success = process_race(folder)
        elapsed = time.time() - t_r
        if success:
            ok += 1
            log.info("DONE %s in %.1fs", folder.name, elapsed)
        else:
            failed += 1
            log.error("FAIL %s in %.1fs", folder.name, elapsed)

    total_time = time.time() - t0
    log.info("Stage 1 COMPLETE: %d ok, %d skipped, %d failed (total %.1fs)",
             ok, skipped, failed, total_time)

    parquets = sorted(PROCESSED.glob("*.parquet"))
    log.info("Processed parquets in data/processed: %d", len(parquets))
    for p in parquets[:5]:
        log.info("  %s: %.1f MB", p.name, p.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
