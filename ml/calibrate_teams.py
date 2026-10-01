"""
ml/calibrate_teams.py  --  Stage 3: fit per-team energy model constants.

For each team, fit: capacity_kwh, regen_sens, deploy_sens, taper_v_start, taper_v_zero
so that estimated SoC is LOW in the 300 m before observed clips
and HIGH (or at least higher) elsewhere.

Uses only training races (first 70% of available races chronologically).
Saves fitted constants to config/teams.yaml marked PROVISIONAL.

Gate 3: per team, mean SoC in clip-preceding 300m must be lower than mean SoC elsewhere.
        If difference is not clear (< 5.0 SoC points), stop and report.
"""

import gc
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parent.parent
PROCESSED   = ROOT / "data" / "processed"
CONFIG_DIR  = ROOT / "config"

CONFIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging (StreamHandler only for clean shell redirection)
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
# Constants & Import energy engine
# ---------------------------------------------------------------------------
sys.path.insert(0, str(ROOT))
# pyrefly: ignore [missing-import]
from backend.energy_engine import (
    ETA_HARVEST, LIFT_COEFF, speed_taper,
)

DT          = 0.25   # 4 Hz
MIN_GAP_SOC = 5.0    # gate 3: minimum SoC difference (points)
SOC_INIT    = 80.0


def fast_soc_trajectory(
    bf_arr: np.ndarray,
    coast_arr: np.ndarray,
    tf_arr: np.ndarray,
    speed_kmh: np.ndarray,
    cap_kwh: float,
    regen_sens: float,
    deploy_sens: float,
    v_start: float,
    v_zero: float,
    soc_init: float = 80.0,
) -> np.ndarray:
    """Fast computation of SoC trajectory for calibration."""
    n = len(speed_kmh)
    soc_arr = np.empty(n, dtype=np.float32)

    taper = speed_taper(speed_kmh, v_start, v_zero)
    e_dep_step  = tf_arr * taper * deploy_sens * (0.45 * DT)
    e_harv_step = (ETA_HARVEST * bf_arr * regen_sens * (1.30 * DT)) + (LIFT_COEFF * coast_arr * regen_sens * (0.25 * DT))

    scale = 100.0 / max(float(cap_kwh) * 3.6, 0.36)
    delta_arr = (e_harv_step - e_dep_step) * scale

    s = float(soc_init)
    for i in range(n):
        s = float(np.clip(s + delta_arr[i], 0.0, 100.0))
        soc_arr[i] = s

    return soc_arr


def calibrate_team(team: str, train_dfs: list[pd.DataFrame]) -> dict:
    """Fit energy parameters for one team using scipy L-BFGS-B."""
    driver_chunks = []
    for df in train_dfs:
        if len(driver_chunks) >= 2:
            break
        tm_df = df[df["team"] == team]
        if tm_df.empty or "clip_label" not in tm_df.columns:
            continue
        for drv, grp in tm_df.groupby("driver_number", sort=False):
            if len(driver_chunks) >= 2:
                break
            grp = grp.sort_values(["lap_number", "distance"]).reset_index(drop=True)
            if len(grp) < 100:
                continue
            has_clips = grp["clip_label"].sum() > 0
            if has_clips:
                # Take slice of 3000 rows around clips
                sub = grp.iloc[:min(len(grp), 3000)].copy()
                driver_chunks.append({
                    "speed": sub["speed"].values.astype(np.float32),
                    "throttle": sub["throttle"].values.astype(np.float32),
                    "brake": sub["brake"].values.astype(np.float32),
                    "clip_mask": (sub["clip_label"].values > 0).astype(bool),
                })

    if not driver_chunks:
        log.warning("  team %s: no training chunks", team)
        return {
            "capacity_kwh": 1.0,
            "regen_sens": 1.0,
            "deploy_sens": 1.0,
            "taper_v_start": 280.0,
            "taper_v_zero": 370.0,
            "status": "PROVISIONAL",
        }

    # Precompute arrays per driver chunk
    for ch in driver_chunks:
        ch["bf_arr"] = np.clip(ch["brake"] / 100.0, 0.0, 1.0).astype(np.float32) ** 1.2
        ch["tf_arr"] = (ch["throttle"] / 100.0).astype(np.float32) ** 1.5
        ch["coast_arr"] = ((ch["throttle"] < 5.0) & (ch["brake"] < 5.0)).astype(np.float32)

    total_pre = sum(ch["clip_mask"].sum() for ch in driver_chunks)
    log.info("  team %s: %d drivers, %d pre-clip samples", team, len(driver_chunks), total_pre)

    def objective(params):
        cap, regen, deploy, vstart, vzero = params
        if cap <= 0.1 or regen <= 0.1 or deploy <= 0.1 or vstart >= vzero:
            return 1e6

        pre_socs = []
        oth_socs = []
        for ch in driver_chunks:
            soc = fast_soc_trajectory(
                ch["bf_arr"], ch["coast_arr"], ch["tf_arr"], ch["speed"],
                cap, regen, deploy, vstart, vzero, soc_init=SOC_INIT,
            )
            m = ch["clip_mask"]
            if m.sum() > 0:
                pre_socs.extend(soc[m].tolist())
            oth = ~m
            if oth.sum() > 0:
                oth_socs.extend(soc[oth][::4].tolist())

        if not pre_socs or not oth_socs:
            return 1e6

        m_pre = float(np.mean(pre_socs))
        m_oth = float(np.mean(oth_socs))
        gap = m_oth - m_pre
        # Minimize pre-clip SoC, maximize gap
        loss = m_pre - 1.5 * gap
        return loss

    x0 = [1.0, 1.0, 1.0, 280.0, 370.0]
    bounds = [
        (0.5, 2.0),     # capacity_kwh
        (0.6, 2.2),     # regen_sens
        (0.6, 2.2),     # deploy_sens
        (250.0, 290.0), # taper_v_start
        (340.0, 380.0), # taper_v_zero
    ]

    try:
        res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 30, "ftol": 1e-3})
        p = res.x
        log.info("  team %s: fitted cap=%.3f regen=%.3f deploy=%.3f vs=%.1f vz=%.1f (loss=%.2f)",
                 team, p[0], p[1], p[2], p[3], p[4], res.fun)
        return {
            "capacity_kwh": float(round(p[0], 4)),
            "regen_sens": float(round(p[1], 4)),
            "deploy_sens": float(round(p[2], 4)),
            "taper_v_start": float(round(p[3], 1)),
            "taper_v_zero": float(round(p[4], 1)),
            "status": "PROVISIONAL",
        }
    except Exception as exc:
        log.warning("  team %s: optimization failed: %s, using defaults", team, exc)
        return {
            "capacity_kwh": 1.0,
            "regen_sens": 1.0,
            "deploy_sens": 1.0,
            "taper_v_start": 280.0,
            "taper_v_zero": 370.0,
            "status": "PROVISIONAL",
        }


def gate3_check(team: str, fitted: dict, train_dfs: list[pd.DataFrame]) -> tuple[bool, float, float, float]:
    """
    Check Gate 3: mean estimated SoC in 300m before clips vs elsewhere.
    Returns (passed, mean_pre, mean_other, gap).
    """
    soc_pre, soc_other = [], []
    for df in train_dfs:
        tm_df = df[df["team"] == team]
        if tm_df.empty or "clip_label" not in tm_df.columns:
            continue
        for drv, grp in tm_df.groupby("driver_number", sort=False):
            grp = grp.sort_values(["lap_number", "distance"]).reset_index(drop=True)
            if len(grp) < 100:
                continue

            speed = grp["speed"].values.astype(np.float32)
            throttle = grp["throttle"].values.astype(np.float32)
            brake = grp["brake"].values.astype(np.float32)
            clip_mask = (grp["clip_label"].values > 0).astype(bool)

            bf_arr = np.clip(brake / 100.0, 0.0, 1.0).astype(np.float32) ** 1.2
            tf_arr = (throttle / 100.0).astype(np.float32) ** 1.5
            coast_arr = ((throttle < 5.0) & (brake < 5.0)).astype(np.float32)

            soc = fast_soc_trajectory(
                bf_arr, coast_arr, tf_arr, speed,
                fitted["capacity_kwh"], fitted["regen_sens"], fitted["deploy_sens"],
                fitted["taper_v_start"], fitted["taper_v_zero"], soc_init=SOC_INIT,
            )

            if clip_mask.sum() > 0:
                soc_pre.extend(soc[clip_mask].tolist())
            other_idx = np.where(~clip_mask)[0]
            if len(other_idx) > 1000:
                other_idx = np.random.choice(other_idx, 1000, replace=False)
            soc_other.extend(soc[other_idx].tolist())

    if not soc_pre or not soc_other:
        return False, 0.0, 0.0, 0.0

    mean_pre = float(np.mean(soc_pre))
    mean_other = float(np.mean(soc_other))
    gap = mean_other - mean_pre
    passed = gap >= MIN_GAP_SOC
    return passed, mean_pre, mean_other, gap


def save_teams_yaml(fitted_teams: dict):
    out_path = CONFIG_DIR / "teams.yaml"
    header = (
        "# Per-team calibrated energy model constants\n"
        "# STATUS: PROVISIONAL - fitted against observed 2026 clipping events.\n"
        "# Not official FIA or team technical numbers.\n\n"
    )
    doc = {"teams": fitted_teams}
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.dump(doc, f, default_flow_style=False, allow_unicode=True)
    log.info("Saved config/teams.yaml with %d teams", len(fitted_teams))


def main():
    t0 = time.time()
    log.info("Stage 3: calibrate_teams.py START")

    labelled_files = sorted(PROCESSED.glob("*_labelled.parquet"))
    if not labelled_files:
        log.error("No labelled parquets found. Run build_labels.py first.")
        sys.exit(1)

    log.info("Found %d labelled race files", len(labelled_files))

    # Chronological training split: first 70% of available races
    n_train = max(1, int(len(labelled_files) * 0.70))
    train_files = labelled_files[:n_train]
    log.info("Using %d races for calibration training: %s",
             n_train, [f.stem.replace("_labelled", "") for f in train_files])

    train_dfs = []
    for f in train_files:
        df = pd.read_parquet(f)
        for c in df.select_dtypes("float64").columns:
            df[c] = df[c].astype("float32")
        train_dfs.append(df)
        log.info("  loaded %s: %d rows", f.stem, len(df))

    all_teams = set()
    for df in train_dfs:
        if "team" in df.columns:
            all_teams.update(df["team"].dropna().unique())
    all_teams.discard("UNKNOWN")
    teams_list = sorted(all_teams)
    log.info("Teams to calibrate (%d): %s", len(teams_list), teams_list)

    fitted_teams = {}
    gate3_results = {}

    for team in teams_list:
        t_tm = time.time()
        fitted = calibrate_team(team, train_dfs)
        passed, m_pre, m_oth, gap = gate3_check(team, fitted, train_dfs)
        elapsed = time.time() - t_tm
        fitted["fitted_on"] = [f.stem.replace("_labelled", "") for f in train_files]
        fitted_teams[team] = fitted
        gate3_results[team] = {
            "passed": passed,
            "mean_pre_clip_soc": round(m_pre, 2),
            "mean_other_soc": round(m_oth, 2),
            "gap_soc": round(gap, 2),
            "time_s": round(elapsed, 1),
        }

    # -----------------------------------------------------------------------
    # Gate 3 Evaluation
    # -----------------------------------------------------------------------
    log.info("==================================================")
    log.info("GATE 3 RESULTS PER TEAM:")
    log.info("%-25s  %-12s  %-12s  %-10s  %s", "Team", "Pre-Clip SoC", "Other SoC", "Gap (pts)", "Status")
    log.info("-" * 70)

    fail = False
    for team, res in gate3_results.items():
        status = "PASS" if res["passed"] else "FAIL"
        log.info("%-25s  %-12.2f  %-12.2f  %-10.2f  %s",
                 team, res["mean_pre_clip_soc"], res["mean_other_soc"], res["gap_soc"], status)
        if not res["passed"]:
            fail = True

    save_teams_yaml(fitted_teams)

    if fail:
        log.error("GATE 3 FAILED: one or more teams failed SoC separation check (< %.1f pts)", MIN_GAP_SOC)
        sys.exit(1)

    log.info("GATE 3 PASSED: all %d teams show clear SoC separation before clipping (gap >= %.1f points).",
             len(teams_list), MIN_GAP_SOC)
    log.info("Stage 3 COMPLETE in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
