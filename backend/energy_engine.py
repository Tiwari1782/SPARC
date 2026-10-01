"""
backend/energy_engine.py  --  SoC update, CPI, EDR, BAI.

Implements the energy model exactly as specified in README.md section 5.
Config is loaded from config/regs_2026.yaml and config/teams.yaml.
Parameters marked PROVISIONAL are fitted values, not official numbers.

This module is designed to be:
  - Importable by both the offline ML pipeline and the live backend server.
  - Stateless per call: pass in the tick data and the previous state.
  - Vectorisable: operates on numpy arrays for batch processing of all drivers.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
_ROOT   = Path(__file__).resolve().parent.parent
_REGS   = _ROOT / "config" / "regs_2026.yaml"
_TEAMS  = _ROOT / "config" / "teams.yaml"


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


_regs_cfg  = _load_yaml(_REGS)
_teams_cfg = _load_yaml(_TEAMS)


def reload_config():
    """Reload config files after calibration."""
    global _regs_cfg, _teams_cfg
    _regs_cfg  = _load_yaml(_REGS)
    _teams_cfg = _load_yaml(_TEAMS)


def _get(d: dict, *keys, default=None):
    """Safe nested dict get."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur if cur is not None else default


# ---------------------------------------------------------------------------
# Default physics parameters (overridden by config/teams.yaml per team)
# ---------------------------------------------------------------------------
DEFAULT_CAPACITY_KWH  = float(_get(_regs_cfg, "battery", "capacity_kwh", default=1.11) or 1.11)
DEFAULT_REGEN_SENS    = 1.0        # per-team fitted value
DEFAULT_DEPLOY_SENS   = 1.0        # per-team fitted value
DEFAULT_TAPER_V_START = float(_get(_regs_cfg, "physics", "taper_v_start", default=280.0) or 280.0)
DEFAULT_TAPER_V_ZERO  = float(_get(_regs_cfg, "physics", "taper_v_zero",  default=370.0) or 370.0)
ETA_HARVEST           = float(_get(_regs_cfg, "physics", "eta_harvest",    default=0.85) or 0.85)
LIFT_COEFF            = float(_get(_regs_cfg, "physics", "lift_coeff",     default=0.30) or 0.30)
FUEL_START_KG         = float(_get(_regs_cfg, "fuel", "start_mass_kg",     default=70.0) or 70.0)
BURN_RATE             = float(_get(_regs_cfg, "fuel", "burn_rate_kg_per_lap", default=1.2) or 1.2)
SOC_INIT              = float(_get(_regs_cfg, "soc", "initial",            default=80.0) or 80.0)

# CPI rival distance threshold (seconds)
P_RIVAL_GAP_THRESHOLD = 2.0


# ---------------------------------------------------------------------------
# Team parameter lookup
# ---------------------------------------------------------------------------

def get_team_params(team: str) -> dict:
    """Return per-team fitted constants, falling back to defaults."""
    teams = _get(_teams_cfg, "teams") or {}
    t = teams.get(team, {}) or {}
    return {
        "capacity_kwh": float(t.get("capacity_kwh") or DEFAULT_CAPACITY_KWH),
        "regen_sens":   float(t.get("regen_sens")   or DEFAULT_REGEN_SENS),
        "deploy_sens":  float(t.get("deploy_sens")  or DEFAULT_DEPLOY_SENS),
        "taper_v_start":float(t.get("taper_v_start")or DEFAULT_TAPER_V_START),
        "taper_v_zero": float(t.get("taper_v_zero") or DEFAULT_TAPER_V_ZERO),
    }


# ---------------------------------------------------------------------------
# Speed taper
# ---------------------------------------------------------------------------

def speed_taper(v: np.ndarray, v_start: float, v_zero: float) -> np.ndarray:
    """
    Linear taper from 1.0 at v_start to 0.0 at v_zero.
    Clips to [0, 1].
    """
    span = max(float(v_zero - v_start), 1.0)
    t = 1.0 - (v - v_start) / span
    return np.clip(t, 0.0, 1.0).astype("float32")


# ---------------------------------------------------------------------------
# SoC update (one tick)
# ---------------------------------------------------------------------------

def soc_step(
    soc: float,
    throttle: float,
    brake: float,
    speed_kmh: float,
    dt: float,
    team_params: dict,
) -> tuple[float, float, float]:
    """
    Compute next SoC given current state per README section 5.1.
    Returns (soc_next, e_harvest, e_deploy).
    """
    cap_kwh     = team_params["capacity_kwh"]
    regen_sens  = team_params["regen_sens"]
    deploy_sens = team_params["deploy_sens"]
    v_start     = team_params["taper_v_start"]
    v_zero      = team_params["taper_v_zero"]

    taper = float(speed_taper(np.array([speed_kmh], dtype="float32"), v_start, v_zero)[0])
    bf = float(np.clip(brake / 100.0, 0.0, 1.0) ** 1.2)
    tf = float((throttle / 100.0) ** 1.5)
    coast = 1.0 if (throttle < 5.0 and brake < 5.0) else 0.0

    e_deploy  = tf * taper * deploy_sens * (0.45 * dt)
    e_harvest = (ETA_HARVEST * bf * regen_sens * (1.30 * dt)) + (LIFT_COEFF * coast * regen_sens * (0.25 * dt))

    scale = 100.0 / max(cap_kwh * 3.6, 0.36)
    delta = (e_harvest - e_deploy) * scale
    soc_next = float(np.clip(soc + delta, 0.0, 100.0))

    return soc_next, e_harvest, e_deploy


# ---------------------------------------------------------------------------
# Vectorised SoC trajectory for a driver sequence
# ---------------------------------------------------------------------------

def compute_soc_trajectory(
    throttle:        np.ndarray,
    brake:           np.ndarray,
    speed_kmh:       np.ndarray,
    dt:              float,
    team_params:     dict,
    soc_init:        float = SOC_INIT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute SoC, cumulative harvest, cumulative deploy for a full driver sequence.
    Returns (soc, e_harvest_cum, e_deploy_cum) each (N,) float32.
    """
    n = len(throttle)
    soc_arr         = np.empty(n, dtype="float32")
    e_harvest_arr   = np.empty(n, dtype="float32")
    e_deploy_arr    = np.empty(n, dtype="float32")

    cap_kwh     = team_params["capacity_kwh"]
    regen_sens  = team_params["regen_sens"]
    deploy_sens = team_params["deploy_sens"]
    v_start     = team_params["taper_v_start"]
    v_zero      = team_params["taper_v_zero"]

    taper_arr = speed_taper(speed_kmh, v_start, v_zero)
    bf_arr    = np.clip(brake / 100.0, 0.0, 1.0).astype("float32") ** 1.2
    tf_arr    = (throttle / 100.0).astype("float32") ** 1.5
    coast_arr = ((throttle < 5.0) & (brake < 5.0)).astype("float32")

    e_dep_step  = tf_arr * taper_arr * deploy_sens * (0.45 * dt)
    e_harv_step = (ETA_HARVEST * bf_arr * regen_sens * (1.30 * dt)) + (LIFT_COEFF * coast_arr * regen_sens * (0.25 * dt))

    scale = 100.0 / max(cap_kwh * 3.6, 0.36)
    delta_arr = (e_harv_step - e_dep_step) * scale

    soc = float(soc_init)
    cum_h = 0.0
    cum_d = 0.0

    for i in range(n):
        soc = float(np.clip(soc + delta_arr[i], 0.0, 100.0))
        cum_h += float(e_harv_step[i])
        cum_d += float(e_dep_step[i])
        soc_arr[i]       = soc
        e_harvest_arr[i] = cum_h
        e_deploy_arr[i]  = cum_d

    return soc_arr, e_harvest_arr, e_deploy_arr


# ---------------------------------------------------------------------------
# CPI (Clip Pressure Index)
# ---------------------------------------------------------------------------

def compute_cpi(
    d_ahead:        float,
    r_avail:        float,
    gap_ahead_s:    float,
    laps_completed: int,
    thermal_derate: float = 1.0,
) -> float:
    """
    CPI = (D_ahead / R_avail)^1.3 * (1 + 0.5 * P_rival) * W_fuel
    per README section 5.3.
    """
    r_eff = max(r_avail * thermal_derate, 1e-2)

    if 0.0 < gap_ahead_s <= P_RIVAL_GAP_THRESHOLD:
        p_rival = 1.0 / (gap_ahead_s + 0.5)
    else:
        p_rival = 0.0

    w_fuel  = 1.0 + 0.002 * max(laps_completed, 0)
    ratio   = max(d_ahead, 0.0) / r_eff
    cpi     = (ratio ** 1.3) * (1.0 + 0.5 * p_rival) * w_fuel
    return float(np.clip(cpi, 0.0, 10.0))


def cpi_risk_level(cpi: float) -> str:
    """Map CPI value to risk string."""
    if cpi < 0.6:
        return "GREEN"
    if cpi <= 1.0:
        return "AMBER"
    return "RED"


# ---------------------------------------------------------------------------
# EDR (Energy Debt Rate) - per lap
# ---------------------------------------------------------------------------

def compute_edr(e_deploy_lap: float, e_harvest_lap: float, lap_length_m: float) -> float:
    """
    EDR(lap) = (sum of deploy - sum of harvest) / lap_length
    Positive means more spent than recovered per README section 5.4.
    """
    if lap_length_m <= 0:
        return 0.0
    return float((e_deploy_lap - e_harvest_lap) / lap_length_m)


# ---------------------------------------------------------------------------
# BAI (Battle Advantage Index)
# ---------------------------------------------------------------------------

def compute_bai(cpi_defender: float, cpi_attacker: float) -> float:
    """
    BAI = CPI_defender - CPI_attacker
    Positive -> attacker has energy advantage.
    Negative -> defender can hold position.
    """
    return float(cpi_defender - cpi_attacker)


# ---------------------------------------------------------------------------
# Fuel mass correction
# ---------------------------------------------------------------------------

def fuel_load_est_kg(lap_number: int, start_mass: float = FUEL_START_KG,
                     burn_rate: float = BURN_RATE) -> float:
    """Estimate remaining fuel mass at start of lap."""
    return max(0.0, start_mass - burn_rate * max(lap_number - 1, 0))


# ---------------------------------------------------------------------------
# Vectorised batch: process a full race DataFrame
# ---------------------------------------------------------------------------

def apply_energy_model(df, team_col="team", dt=0.25) -> "pd.DataFrame":
    """
    Apply energy model to a full race DataFrame in-place.
    Adds columns: soc_est, e_harvest_cum, e_deploy_cum, fuel_load_est_kg.
    """
    import pandas as pd

    soc_col    = np.full(len(df), np.nan, "float32")
    harv_col   = np.full(len(df), np.nan, "float32")
    dep_col    = np.full(len(df), np.nan, "float32")

    for drv, grp in df.groupby("driver_number", sort=False):
        idx   = grp.index.values
        team  = str(grp[team_col].iloc[0]) if team_col in grp.columns else "UNKNOWN"
        tp    = get_team_params(team)

        soc, eh, ed = compute_soc_trajectory(
            throttle    = grp["throttle"].values.astype("float32"),
            brake       = grp["brake"].values.astype("float32"),
            speed_kmh   = grp["speed"].values.astype("float32"),
            dt          = dt,
            team_params = tp,
        )
        pos = np.searchsorted(df.index.values, idx)
        soc_col[pos]  = soc
        harv_col[pos] = eh
        dep_col[pos]  = ed

    df["soc_est"]       = soc_col
    df["e_harvest_cum"] = harv_col
    df["e_deploy_cum"]  = dep_col
    df["fuel_load_est_kg"] = df["lap_number"].apply(fuel_load_est_kg).astype("float32")

    return df
