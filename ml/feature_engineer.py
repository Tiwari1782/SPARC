"""
ml/feature_engineer.py  --  Stage 4a: build features for training and live inference.

Amendments implemented:
1. Dual Tasks:
   - Task 1: 300 m horizon prediction (clip_label).
   - Task 2: Straight-level prediction at 40% straight length mark (straight_clip_label_40pct).
     Excludes straights where clipping starts before the 40% mark.
2. New features added (strictly backward-looking, no leakage):
   - energy_deployed_since_brake
   - speed_delta_vs_reference (reference speed per circuit/position built on training folds)
   - accel_100m, accel_200m, accel_trend
   - distance_since_last_harvest, harvest_duration_last_zone
   - team_straight_clip_rate (computed out-of-fold within training set)
3. Circularity testing support: clean separation of kinematic acceleration features vs energy features.

Zero emojis anywhere.
"""

from __future__ import annotations

import gc
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HZ              = 4
DT              = 1.0 / HZ
RPM_MAX         = 15000.0
WINDOW_SAMPLES  = 3
P_RIVAL_THRESH  = 2.0

STRAIGHT_MIN_SPEED    = 200.0
STRAIGHT_MIN_THROTTLE = 90.0
STRAIGHT_MIN_DIST     = 150.0

# ---------------------------------------------------------------------------
# Encoding maps & Reference Profiles
# ---------------------------------------------------------------------------
_team_encoder:    Optional[dict] = None
_circuit_encoder: Optional[dict] = None
_speed_ref_maps:  dict = {}  # circuit -> distance_bin -> median speed


def fit_encoders(df: pd.DataFrame) -> None:
    global _team_encoder, _circuit_encoder
    teams    = sorted(df["team"].dropna().unique()) if "team" in df.columns else []
    circuits = sorted(df["race"].dropna().unique()) if "race" in df.columns else (
        sorted(df["circuit"].dropna().unique()) if "circuit" in df.columns else []
    )
    _team_encoder    = {t: i for i, t in enumerate(teams)}
    _circuit_encoder = {c: i for i, c in enumerate(circuits)}
    log.info("Encoders fitted: %d teams, %d circuits", len(teams), len(circuits))


def fit_reference_speed_profiles(df_train: pd.DataFrame) -> None:
    """Build speed reference curve per circuit/distance bin from training races."""
    global _speed_ref_maps
    _speed_ref_maps = {}
    if "race" not in df_train.columns or "distance" not in df_train.columns:
        return
    for r, rdf in df_train.groupby("race"):
        # Bin distance by 25m
        d_bins = (rdf["distance"] // 25 * 25).astype(int)
        med_speed = rdf.groupby(d_bins)["speed"].median().to_dict()
        _speed_ref_maps[r] = med_speed


def encode_team(team: str) -> int:
    if _team_encoder is None:
        return -1
    return _team_encoder.get(str(team), -1)


def encode_circuit(circuit: str) -> int:
    if _circuit_encoder is None:
        return -1
    return _circuit_encoder.get(str(circuit), -1)


def get_encoders() -> tuple[dict, dict]:
    return (_team_encoder or {}, _circuit_encoder or {})


def load_encoders(team_enc: dict, circuit_enc: dict) -> None:
    global _team_encoder, _circuit_encoder
    _team_encoder    = team_enc
    _circuit_encoder = circuit_enc


# ---------------------------------------------------------------------------
# Fast rolling features (backward window, zero leakage)
# ---------------------------------------------------------------------------

def _rolling_mean(arr: np.ndarray, w: int = WINDOW_SAMPLES) -> np.ndarray:
    return pd.Series(arr).rolling(w, min_periods=1).mean().values.astype("float32")


def _rolling_slope(arr: np.ndarray, w: int = WINDOW_SAMPLES) -> np.ndarray:
    out = np.zeros(len(arr), dtype="float32")
    if len(arr) >= 3:
        out[2:] = ((arr[2:] - arr[:-2]) / 2.0).astype("float32")
    return out


def _detect_straights_meta(speed: np.ndarray, throttle: np.ndarray, dist: np.ndarray):
    """
    Returns (dist_to_end, seg_len, frac_straight, straight_id).
    """
    n = len(speed)
    on_straight = ((speed >= STRAIGHT_MIN_SPEED) & (throttle >= STRAIGHT_MIN_THROTTLE))
    dist_to_end   = np.zeros(n, dtype="float32")
    seg_len       = np.zeros(n, dtype="float32")
    frac_straight = np.zeros(n, dtype="float32")
    straight_id   = np.full(n, -1, dtype="int32")

    current_id = 0
    i = 0
    while i < n:
        if on_straight[i]:
            j = i
            while j < n and on_straight[j]:
                j += 1
            l = dist[j-1] - dist[i] if j > i else 0.0
            if l >= STRAIGHT_MIN_DIST:
                seg_len[i:j] = float(l)
                d_end = dist[j-1]
                dist_to_end[i:j] = (d_end - dist[i:j]).astype("float32")
                frac_straight[i:j] = np.clip((dist[i:j] - dist[i]) / max(l, 1.0), 0.0, 1.0).astype("float32")
                straight_id[i:j] = current_id
                current_id += 1
            i = j
        else:
            i += 1
    return dist_to_end, seg_len, frac_straight, straight_id


def _harvest_and_deploy_history(brake: np.ndarray, throttle: np.ndarray,
                                 e_harvest: np.ndarray, e_deploy: np.ndarray,
                                 dist: np.ndarray):
    """
    Computes:
      - harvest_last_brake_zone
      - deploy_last_straight
      - energy_deployed_since_brake
      - distance_since_last_harvest
      - harvest_duration_last_zone
    """
    n = len(brake)
    harvest_last_zone   = np.zeros(n, dtype="float32")
    deploy_last_str     = np.zeros(n, dtype="float32")
    deploy_since_brake  = np.zeros(n, dtype="float32")
    dist_since_harvest  = np.zeros(n, dtype="float32")
    harvest_dur_last    = np.zeros(n, dtype="float32")

    last_harvest_total = 0.0
    last_harvest_dur = 0.0
    current_harvest_total = 0.0
    current_harvest_start_d = 0.0
    in_brake = False

    cum_deploy_since_b = 0.0
    last_harvest_end_d = 0.0

    on_str = (throttle >= STRAIGHT_MIN_THROTTLE)
    last_str_total = 0.0
    curr_str_total = 0.0
    in_str = False

    for i in range(n):
        # Brake / harvest tracking
        if brake[i] > 0 or e_harvest[i] > 0:
            if not in_brake:
                in_brake = True
                current_harvest_total = 0.0
                current_harvest_start_d = dist[i]
            current_harvest_total += float(e_harvest[i])
            cum_deploy_since_b = 0.0
            last_harvest_end_d = dist[i]
        else:
            if in_brake:
                in_brake = False
                last_harvest_total = current_harvest_total
                last_harvest_dur = max(0.0, dist[i] - current_harvest_start_d)
                last_harvest_end_d = dist[i]
            cum_deploy_since_b += float(e_deploy[i])

        harvest_last_zone[i]   = last_harvest_total
        harvest_dur_last[i]    = last_harvest_dur
        deploy_since_brake[i]  = cum_deploy_since_b
        dist_since_harvest[i]  = max(0.0, dist[i] - last_harvest_end_d)

        # Straight deploy tracking
        if on_str[i]:
            if not in_str:
                in_str = True
                curr_str_total = 0.0
            curr_str_total += float(e_deploy[i])
        else:
            if in_str:
                in_str = False
                last_str_total = curr_str_total
        deploy_last_str[i] = last_str_total

    return harvest_last_zone, deploy_last_str, deploy_since_brake, dist_since_harvest, harvest_dur_last


def _rolling_spatial_accel(accel: np.ndarray, dist: np.ndarray):
    """
    Computes backward accel over 100m, 200m and trend (accel_100m - accel_200m).
    """
    n = len(accel)
    accel_100 = np.zeros(n, dtype="float32")
    accel_200 = np.zeros(n, dtype="float32")
    
    # 100m is approx 15-20 samples at 4Hz race speed
    # Vectorised rolling window using sample approximations
    w100 = max(1, int(100.0 / 15.0))
    w200 = max(1, int(200.0 / 15.0))

    s_acc = pd.Series(accel)
    accel_100 = s_acc.rolling(w100, min_periods=1).mean().values.astype("float32")
    accel_200 = s_acc.rolling(w200, min_periods=1).mean().values.astype("float32")
    accel_trend = (accel_100 - accel_200).astype("float32")

    return accel_100, accel_200, accel_trend


# ---------------------------------------------------------------------------
# Feature Column Definitions
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    # Energy state
    "soc_est", "edr_lap", "cpi",
    "harvest_last_brake_zone", "deploy_last_straight",
    "energy_deployed_since_brake", "distance_since_last_harvest",
    "harvest_duration_last_zone",
    # Track context
    "distance_to_straight_end", "straight_length", "lap_fraction", "distance_since_last_braking",
    # Driver inputs & kinematics
    "throttle", "throttle_mean_50m", "speed", "speed_slope_50m",
    "speed_delta_vs_reference",
    "accel_100m", "accel_200m", "accel_trend",
    "gear", "rpm_ratio",
    # Race context
    "lap_number", "fuel_load_est_kg", "tyre_age", "compound_enc",
    "gap_ahead_s", "gap_behind_s", "p_rival",
    "track_temp", "rain_flag",
    "team_id", "circuit_id", "team_straight_clip_rate"
]

KINEMATIC_ACCEL_COLS = [
    "speed_slope_50m", "speed_delta_vs_reference",
    "accel_100m", "accel_200m", "accel_trend"
]

ENERGY_COLS = [
    "soc_est", "edr_lap", "cpi",
    "harvest_last_brake_zone", "deploy_last_straight",
    "energy_deployed_since_brake", "distance_since_last_harvest",
    "harvest_duration_last_zone"
]


def build_features(df: pd.DataFrame, circuit: str = "",
                   team_clip_rates: Optional[dict] = None) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    n = len(df)

    speed    = df["speed"].values.astype("float32")
    throttle = df["throttle"].values.astype("float32")
    brake    = df["brake"].values.astype("float32")
    gear     = df["gear"].values.astype("float32") if "gear" in df.columns else np.ones(n, "float32")
    rpm      = df["rpm"].values.astype("float32") if "rpm" in df.columns else np.zeros(n, "float32")
    dist     = df["distance"].values.astype("float32")
    lap      = df["lap_number"].values.astype("int16")

    # Rolling inputs
    df["throttle_mean_50m"] = _rolling_mean(throttle, WINDOW_SAMPLES)
    df["speed_slope_50m"]   = _rolling_slope(speed, WINDOW_SAMPLES)
    df["rpm_ratio"]         = (rpm / RPM_MAX).clip(0, 1).astype("float32")

    # Acceleration kinematics
    accel = df["accel"].values.astype("float32") if "accel" in df.columns else _rolling_slope(speed, 5)
    a100, a200, atrend = _rolling_spatial_accel(accel, dist)
    df["accel_100m"]  = a100
    df["accel_200m"]  = a200
    df["accel_trend"] = atrend

    # Speed delta vs training reference profile
    circuit_str = circuit or (str(df["race"].iloc[0]) if "race" in df.columns else "")
    ref_map = _speed_ref_maps.get(circuit_str, {})
    if ref_map:
        d_bins = (dist // 25 * 25).astype(int)
        ref_speeds = np.array([ref_map.get(b, s) for b, s in zip(d_bins, speed)], dtype="float32")
        df["speed_delta_vs_reference"] = (speed - ref_speeds).astype("float32")
    else:
        df["speed_delta_vs_reference"] = np.float32(0.0)

    # Straight metadata & Task 2 markers
    d_to_end, s_len, frac_st, st_id = _detect_straights_meta(speed, throttle, dist)
    df["distance_to_straight_end"]     = d_to_end
    df["straight_length"]              = s_len
    df["straight_fraction"]            = frac_st
    df["straight_id"]                  = st_id
    df["distance_since_last_braking"]  = np.maximum(0.0, s_len - d_to_end).astype("float32")

    # Lap fraction
    max_d_lap = df.groupby("lap_number")["distance"].transform("max").fillna(1.0).values.astype("float32")
    df["lap_fraction"] = (dist / np.maximum(max_d_lap, 1.0)).astype("float32")

    # Energy features
    e_harvest = df["e_harvest_cum"].values.astype("float32") if "e_harvest_cum" in df.columns else np.zeros(n, "float32")
    e_deploy  = df["e_deploy_cum"].values.astype("float32") if "e_deploy_cum" in df.columns else np.zeros(n, "float32")

    hlz, dls, dsb, dsh, hdl = _harvest_and_deploy_history(brake, throttle, e_harvest, e_deploy, dist)
    df["harvest_last_brake_zone"]      = hlz
    df["deploy_last_straight"]         = dls
    df["energy_deployed_since_brake"]  = dsb
    df["distance_since_last_harvest"]  = dsh
    df["harvest_duration_last_zone"]   = hdl

    # EDR
    lap_s = pd.Series(lap)
    dep_lap = pd.Series(e_deploy).groupby(lap_s).cumsum().values.astype("float32")
    har_lap = pd.Series(e_harvest).groupby(lap_s).cumsum().values.astype("float32")
    df["edr_lap"] = ((dep_lap - har_lap) / np.maximum(dist, 1.0)).astype("float32")

    # Gaps & rival pressure
    gap_ahead  = df["gap_ahead"].values.astype("float32") if "gap_ahead" in df.columns else np.full(n, 30.0, "float32")
    gap_behind = df["gap_behind"].values.astype("float32") if "gap_behind" in df.columns else np.full(n, 30.0, "float32")
    df["gap_ahead_s"]  = gap_ahead
    df["gap_behind_s"] = gap_behind
    p_rival = np.where((gap_ahead <= P_RIVAL_THRESH) & (gap_ahead > 0), 1.0 / (gap_ahead + 0.5), 0.0).astype("float32")
    df["p_rival"] = p_rival

    # CPI
    if "soc_est" in df.columns:
        soc = np.maximum(df["soc_est"].values.astype("float32"), 1.0)
        d_ahead = ((throttle / 100.0) ** 1.5 * 0.5).astype("float32")
        w_fuel = (1.0 + 0.002 * lap).astype("float32")
        ratio = d_ahead / soc
        df["cpi"] = np.clip((ratio ** 1.3) * (1.0 + 0.5 * p_rival) * w_fuel, 0.0, 10.0).astype("float32")
    else:
        df["cpi"] = np.float32(0.0)

    # Fuel load
    if "fuel_load_est_kg" not in df.columns:
        from backend.energy_engine import fuel_load_est_kg as _flkg
        df["fuel_load_est_kg"] = df["lap_number"].apply(_flkg).astype("float32")

    # Tyre age
    df["tyre_age"] = df["tyre_age"].fillna(0.0).astype("float32") if "tyre_age" in df.columns else np.float32(0.0)

    # Compound encoding
    compound_order = {"SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTERMEDIATE": 3, "WET": 4, "UNKNOWN": -1}
    comp_col = df["compound"].fillna("UNKNOWN") if "compound" in df.columns else pd.Series(["UNKNOWN"] * n)
    df["compound_enc"] = comp_col.map(lambda c: compound_order.get(str(c).upper(), -1)).astype("float32")

    # Weather
    if "track_temp" not in df.columns:
        df["track_temp"] = np.float32(25.0)
    if "rain_flag" not in df.columns:
        df["rain_flag"] = np.float32(0.0)

    # Encodings
    team_str = str(df["team"].iloc[0]) if "team" in df.columns else "UNKNOWN"
    df["team_id"]    = np.float32(encode_team(team_str))
    df["circuit_id"] = np.float32(encode_circuit(circuit_str))

    # Out-of-fold team straight clip rate
    if team_clip_rates and team_str in team_clip_rates:
        df["team_straight_clip_rate"] = np.float32(team_clip_rates[team_str])
    else:
        df["team_straight_clip_rate"] = np.float32(0.05)

    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = np.float32(0.0)
        df[col] = df[col].astype("float32")

    return df
