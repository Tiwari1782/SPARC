"""
ml/build_labels.py -- Stage 2: detect observed clipping events (v1 and tightened v2).

Amendments implemented:
1. Tightened clip detector (v2): must start on a straight, exclude circuit-specific
   drag-limited top-speed saturation (derived from label-free speed distributions),
   and require acceleration shortfall to persist for >= 100 m.
2. Per-circuit speed percentile: default 60th percentile of full-throttle speed on
   that circuit (sensitivity analysis across 50th, 60th, 70th percentiles).
3. Both labels_v1 and labels_v2 preserved.
4. Human audit sheet: 100 clip events (stratified) + 50 non-clip straight segments,
   double-blind (neutral filenames and row ordering, key in audit_key.csv).

Zero emojis anywhere.
"""

import gc
import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.ndimage import uniform_filter1d

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parent.parent
PROCESSED   = ROOT / "data" / "processed"
REPORT_DIR  = ROOT / "ml" / "reports" / "paper"
AUDIT_PLOTS = REPORT_DIR / "audit_plots"
LOG_DIR     = ROOT / "logs"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_PLOTS.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

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
HZ              = 4
DT              = 1.0 / HZ

THROTTLE_CLIP   = 98.0          # threshold for full throttle (%)
BRAKE_MAX       = 2.0           # threshold for not braking (%)
LIFT_DTHROTTLE  = -5.0          # max throttle drop (%/s)
MIN_CLIP_DIST   = 100.0         # minimum sustained distance (m) to register clip
AHEAD_DIST      = 300.0         # prediction horizon (m)
ENVELOPE_LAP_MAX= 10            # early laps used for baseline acceleration envelope
SMOOTH_WINDOW   = 5             # sample window for acceleration smoothing
ACCEL_DEFICIT   = -1.0          # deficit below expected envelope (m/s^2)
MAX_CLIP_ACCEL  = 0.8           # actual acceleration plateau bound (m/s^2)


def smooth_accel(speed_kmh: np.ndarray) -> np.ndarray:
    """Compute longitudinal acceleration (m/s^2) from speed, smoothed."""
    speed_ms = speed_kmh.astype("float64") / 3.6
    accel = np.gradient(speed_ms, DT)
    accel = uniform_filter1d(accel, size=SMOOTH_WINDOW, mode="nearest")
    return accel.astype("float32")


def derive_circuit_speed_bounds(df_race: pd.DataFrame) -> dict:
    """
    Derive unsupervised speed percentiles and top-speed saturation ceiling.
    Excludes label information completely.
    """
    ft = df_race[(df_race["throttle"] >= THROTTLE_CLIP) & (df_race["brake"] <= BRAKE_MAX)]
    if ft.empty or len(ft) < 100:
        return {"p50": 180.0, "p60": 200.0, "p70": 220.0, "max_drag_speed": 315.0}

    speeds = ft["speed"].values
    p50 = float(np.percentile(speeds, 50))
    p60 = float(np.percentile(speeds, 60))
    p70 = float(np.percentile(speeds, 70))
    
    # 98.5th percentile on full-throttle early laps as circuit-specific drag ceiling
    early_ft = ft[ft["lap_number"] <= ENVELOPE_LAP_MAX]
    if len(early_ft) >= 50:
        max_drag = float(np.percentile(early_ft["speed"].values, 98.5))
    else:
        max_drag = float(np.percentile(speeds, 98.5))
    
    # Ensure reasonable physics bounds
    max_drag = max(290.0, min(345.0, max_drag))

    return {
        "p50": p50,
        "p60": p60,
        "p70": p70,
        "max_drag_speed": max_drag,
    }


def build_team_envelope(df_team: pd.DataFrame, min_speed: float, max_speed: float) -> dict:
    """Build expected acceleration envelope per team from early laps."""
    early = df_team[
        (df_team["lap_number"] <= ENVELOPE_LAP_MAX) &
        (df_team["throttle"] >= THROTTLE_CLIP) &
        (df_team["brake"] <= BRAKE_MAX) &
        (df_team["speed"] >= min_speed) &
        (df_team["speed"] < max_speed)
    ].copy()

    if early.empty or len(early) < 15:
        return {}

    early["accel"] = smooth_accel(early["speed"].values)
    early["speed_bin"] = (early["speed"] // 10 * 10).astype("int16")

    envelope_lookup = {}
    for (g, sb), grp in early.groupby(["gear", "speed_bin"], observed=True):
        if len(grp) >= 2:
            envelope_lookup[(int(g), int(sb))] = float(grp["accel"].median())

    return envelope_lookup


def get_expected_accel(speed: float, gear: int, env_lookup: dict, fallback_env: dict) -> float:
    sb = int(speed // 10 * 10)
    val = env_lookup.get((gear, sb))
    if val is not None:
        return val
    val = fallback_env.get((gear, sb))
    if val is not None:
        return val
    gear_bins = [k[1] for k in env_lookup if k[0] == gear] or [k[1] for k in fallback_env if k[0] == gear]
    if gear_bins:
        nearest_sb = min(gear_bins, key=lambda b: abs(b - sb))
        return env_lookup.get((gear, nearest_sb)) or fallback_env.get((gear, nearest_sb), 1.5)
    return 1.5


def detect_clips_core(df_drv: pd.DataFrame, team_env: dict, fallback_env: dict,
                      min_speed: float, max_speed: float,
                      require_straight_start: bool = True) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    Returns (is_clipping_sample, clip_label, events).
    """
    n = len(df_drv)
    speed    = df_drv["speed"].values.astype("float32")
    throttle = df_drv["throttle"].values.astype("float32")
    brake    = df_drv["brake"].values.astype("float32")
    gear     = df_drv["gear"].values.astype("int8") if "gear" in df_drv.columns else np.ones(n, "int8")
    dist     = df_drv["distance"].values.astype("float32")
    lap      = df_drv["lap_number"].values.astype("int16")

    accel = smooth_accel(speed)
    dthrottle = np.gradient(throttle.astype("float64"), DT).astype("float32")

    cond_throttle = throttle >= THROTTLE_CLIP
    cond_brake    = brake <= BRAKE_MAX
    cond_lifting  = dthrottle >= LIFT_DTHROTTLE
    cond_speed    = (speed >= min_speed) & (speed < max_speed)

    candidate = cond_throttle & cond_brake & cond_lifting & cond_speed

    exp_accel = np.full(n, np.nan, dtype="float32")
    for i in range(n):
        if candidate[i]:
            exp_accel[i] = get_expected_accel(float(speed[i]), int(gear[i]), team_env, fallback_env)

    accel_deficit = accel - exp_accel
    is_clipping_sample = candidate & (accel_deficit < ACCEL_DEFICIT) & (accel < MAX_CLIP_ACCEL)

    # Sustained events
    events = []
    in_event = False
    ev_start_i = 0
    for i in range(n):
        if is_clipping_sample[i] and not in_event:
            in_event = True
            ev_start_i = i
        elif not is_clipping_sample[i] and in_event:
            in_event = False
            ev_end_i = i - 1
            ev_dist = dist[ev_end_i] - dist[ev_start_i]
            
            # Straight start check: start must satisfy full throttle and speed
            valid_start = True
            if require_straight_start:
                valid_start = (throttle[ev_start_i] >= THROTTLE_CLIP) and (speed[ev_start_i] >= min_speed)

            if ev_dist >= MIN_CLIP_DIST and (lap[ev_end_i] == lap[ev_start_i]) and valid_start:
                events.append({
                    "lap_number":         int(lap[ev_start_i]),
                    "start_idx":          ev_start_i,
                    "end_idx":            ev_end_i,
                    "start_distance":     float(dist[ev_start_i]),
                    "end_distance":       float(dist[ev_end_i]),
                    "duration_m":         float(ev_dist),
                    "speed_mean":         float(speed[ev_start_i:ev_end_i+1].mean()),
                    "accel_deficit_mean": float(accel_deficit[ev_start_i:ev_end_i+1].mean()),
                })
    if in_event:
        ev_end_i = n - 1
        ev_dist = dist[ev_end_i] - dist[ev_start_i]
        valid_start = True
        if require_straight_start:
            valid_start = (throttle[ev_start_i] >= THROTTLE_CLIP) and (speed[ev_start_i] >= min_speed)
        if ev_dist >= MIN_CLIP_DIST and (lap[ev_end_i] == lap[ev_start_i]) and valid_start:
            events.append({
                "lap_number":         int(lap[ev_start_i]),
                "start_idx":          ev_start_i,
                "end_idx":            ev_end_i,
                "start_distance":     float(dist[ev_start_i]),
                "end_distance":       float(dist[ev_end_i]),
                "duration_m":         float(ev_dist),
                "speed_mean":         float(speed[ev_start_i:ev_end_i+1].mean()),
                "accel_deficit_mean": float(accel_deficit[ev_start_i:ev_end_i+1].mean()),
            })

    # Prediction label: clip_label = 1 if clip starts in next 300 m and not already clipping
    clip_label = np.zeros(n, dtype="int8")
    for ev in events:
        s_idx = ev["start_idx"]
        ev_lap = ev["lap_number"]
        ev_start_d = ev["start_distance"]
        j = s_idx - 1
        while j >= 0 and lap[j] == ev_lap and (ev_start_d - dist[j] <= AHEAD_DIST):
            if not is_clipping_sample[j]:
                clip_label[j] = 1
            j -= 1

    return is_clipping_sample, clip_label, events


def render_audit_plot(df_drv: pd.DataFrame, lap: int, start_d: float, end_d: float, out_path: Path, title: str):
    margin = 400.0
    sub = df_drv[
        (df_drv["lap_number"] == lap) &
        (df_drv["distance"] >= start_d - margin) &
        (df_drv["distance"] <= end_d + margin)
    ].sort_values("distance")

    if len(sub) < 5:
        return False

    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(sub["distance"], sub["speed"], color="#1a73e8", linewidth=1.2)
    axes[0].axvspan(start_d, end_d, alpha=0.2, color="red")
    axes[0].set_ylabel("Speed (km/h)")
    axes[0].set_title(title)

    axes[1].plot(sub["distance"], sub["throttle"], color="#34a853", linewidth=1.2)
    axes[1].axvspan(start_d, end_d, alpha=0.2, color="red")
    axes[1].set_ylabel("Throttle (%)")

    if "accel" in sub.columns:
        axes[2].plot(sub["distance"], sub["accel"], color="#e37400", linewidth=1.2, label="Accel")
    if "accel_exp" in sub.columns:
        axes[2].plot(sub["distance"], sub["accel_exp"], color="gray", linestyle="--", linewidth=1.0, label="Expected")
    axes[2].axvspan(start_d, end_d, alpha=0.2, color="red")
    axes[2].set_ylabel("Accel (m/s^2)")
    axes[2].set_xlabel("Distance (m)")
    axes[2].legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def label_all_races():
    t0 = time.time()
    log.info("Stage 2: Adaptive Labelling & Audit Generation START")

    parquets = sorted(PROCESSED.glob("*.parquet"))
    parquets = [p for p in parquets if "_labelled" not in p.stem and "_clips" not in p.stem]
    log.info("Found %d processed race parquets to label", len(parquets))

    sensitivity_records = []
    v1_v2_counts = []
    all_v2_events = []
    all_v1_only_events = []
    all_negative_segments = []

    for pq_path in parquets:
        race_tag = pq_path.stem
        df = pd.read_parquet(pq_path)
        for c in df.select_dtypes(include=["float64"]).columns:
            df[c] = df[c].astype("float32")

        # Unsupervised circuit speed bounds
        bounds = derive_circuit_speed_bounds(df)
        p50 = bounds["p50"]
        p60 = bounds["p60"]
        p70 = bounds["p70"]
        max_drag = bounds["max_drag_speed"]

        # Envelopes
        fb_env_v1 = build_team_envelope(df, 200.0, 315.0)
        fb_env_v2 = build_team_envelope(df, p60, max_drag)

        team_envs_v1 = {}
        team_envs_v2 = {}
        for tm, tdf in df.groupby("team"):
            team_envs_v1[tm] = build_team_envelope(tdf, 200.0, 315.0)
            team_envs_v2[tm] = build_team_envelope(tdf, p60, max_drag)

        # Sensitivity thresholds
        sens_envs = {
            "p50": {tm: build_team_envelope(tdf, p50, max_drag) for tm, tdf in df.groupby("team")},
            "p60": team_envs_v2,
            "p70": {tm: build_team_envelope(tdf, p70, max_drag) for tm, tdf in df.groupby("team")},
        }

        # Run detection per driver
        drv_dfs = []
        race_v1_clips = []
        race_v2_clips = []
        sens_counts = {"p50": 0, "p60": 0, "p70": 0}

        for drv, grp in df.groupby("driver_number", sort=False):
            grp = grp.sort_values(["lap_number", "distance"]).reset_index(drop=True)
            tm = grp["team"].iloc[0] if "team" in grp.columns else "UNKNOWN"

            # 1. v1 detection (fixed 200 km/h, 315 km/h)
            is_clip_v1, lab_v1, evs_v1 = detect_clips_core(
                grp, team_envs_v1.get(tm, {}), fb_env_v1, 200.0, 315.0, require_straight_start=False
            )
            # 2. v2 detection (adaptive p60, max_drag, strict straight start)
            is_clip_v2, lab_v2, evs_v2 = detect_clips_core(
                grp, team_envs_v2.get(tm, {}), fb_env_v2, p60, max_drag, require_straight_start=True
            )

            # Sensitivity checks
            for p_key, min_s in [("p50", p50), ("p60", p60), ("p70", p70)]:
                _, _, evs_s = detect_clips_core(
                    grp, sens_envs[p_key].get(tm, {}), fb_env_v2, min_s, max_drag, require_straight_start=True
                )
                sens_counts[p_key] += len(evs_s)

            for ev in evs_v1:
                ev["race"] = race_tag
                ev["driver_number"] = drv
                ev["team"] = tm
                race_v1_clips.append(ev)

            for ev in evs_v2:
                ev["race"] = race_tag
                ev["driver_number"] = drv
                ev["team"] = tm
                race_v2_clips.append(ev)

            grp["clip_label_v1"] = lab_v1
            grp["clip_label_v2"] = lab_v2
            # Maintain clip_label as v2 for downstream models
            grp["clip_label"]    = lab_v2
            grp["is_clipping"]   = is_clip_v2.astype("int8")
            grp["accel"]         = smooth_accel(grp["speed"].values)

            # Extract full-throttle non-clip segments for audit
            ft_non_clip = (grp["throttle"] >= THROTTLE_CLIP) & (grp["brake"] <= BRAKE_MAX) & (is_clip_v2 == 0)
            if ft_non_clip.any():
                diff = np.diff(ft_non_clip.astype(np.int8))
                st_starts = np.where(diff == 1)[0] + 1
                st_ends = np.where(diff == -1)[0] + 1
                for s_i, e_i in zip(st_starts[:3], st_ends[:3]):
                    if e_i > s_i and (grp["distance"].iloc[e_i] - grp["distance"].iloc[s_i] >= 150.0):
                        all_negative_segments.append({
                            "race": race_tag,
                            "driver_number": drv,
                            "lap_number": int(grp["lap_number"].iloc[s_i]),
                            "start_distance": float(grp["distance"].iloc[s_i]),
                            "end_distance": float(grp["distance"].iloc[e_i]),
                            "type": "negative_straight"
                        })

            drv_dfs.append(grp)

        labelled_df = pd.concat(drv_dfs, ignore_index=True)
        out_labelled = PROCESSED / f"{race_tag}_labelled.parquet"
        out_clips    = PROCESSED / f"{race_tag}_clips.parquet"

        pq.write_table(pa.Table.from_pandas(labelled_df, preserve_index=False), out_labelled, compression="snappy")
        pq.write_table(pa.Table.from_pandas(pd.DataFrame(race_v2_clips), preserve_index=False), out_clips, compression="snappy")

        v1_cnt = len(race_v1_clips)
        v2_cnt = len(race_v2_clips)
        v1_v2_counts.append({"Race": race_tag, "v1_Clips": v1_cnt, "v2_Clips": v2_cnt, "Reduction_pct": (v1_cnt - v2_cnt) / max(v1_cnt, 1) * 100})
        
        sensitivity_records.append({
            "Race": race_tag,
            "p50_speed_kmh": p50,
            "p60_speed_kmh": p60,
            "p70_speed_kmh": p70,
            "max_drag_speed_kmh": max_drag,
            "Clips_p50": sens_counts["p50"],
            "Clips_p60": sens_counts["p60"],
            "Clips_p70": sens_counts["p70"],
        })

        all_v2_events.extend(race_v2_clips)

        # Identify v1-only clips (discarded by v2)
        v2_start_keys = {(c["race"], c["driver_number"], c["lap_number"], round(c["start_distance"], -1)) for c in race_v2_clips}
        for c in race_v1_clips:
            k = (c["race"], c["driver_number"], c["lap_number"], round(c["start_distance"], -1))
            if k not in v2_start_keys:
                c["type"] = "v1_only_positive"
                all_v1_only_events.append(c)

        log.info("  %s -> v1: %d clips | v2: %d clips (p60=%.1f km/h, max_drag=%.1f km/h)",
                 race_tag, v1_cnt, v2_cnt, p60, max_drag)

        del labelled_df, df, drv_dfs
        gc.collect()

    # -----------------------------------------------------------------------
    # Double-Blind Human Audit Sheet & Plots
    # -----------------------------------------------------------------------
    log.info("Generating Double-Blind Human Audit Sheet...")
    df_v2 = pd.DataFrame(all_v2_events)
    df_v2["type"] = "detector_v2_positive"
    df_v1_only = pd.DataFrame(all_v1_only_events)
    df_neg = pd.DataFrame(all_negative_segments)

    # Sample: 70 v2 positives, 30 v1-only positives (total 100 clips) + 50 negatives
    n_v2_sample = min(70, len(df_v2))
    n_v1_sample = min(30, len(df_v1_only))
    n_neg_sample = min(50, len(df_neg))

    sample_v2 = df_v2.sample(n=n_v2_sample, random_state=42)
    sample_v1 = df_v1_only.sample(n=n_v1_sample, random_state=42)
    sample_neg = df_neg.sample(n=n_neg_sample, random_state=42)

    combined = pd.concat([sample_v2, sample_v1, sample_neg], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    audit_rows = []
    key_rows = []

    # Map driver data lookup on demand
    race_dfs = {}

    for idx, row in combined.iterrows():
        sample_id = f"sample_{idx+1:03d}"
        plot_name = f"{sample_id}.png"
        plot_path = AUDIT_PLOTS / plot_name

        race_name = row["race"]
        drv = int(row["driver_number"])
        lap = int(row["lap_number"])
        sd = float(row["start_distance"])
        ed = float(row.get("end_distance", sd + 150.0))
        ev_type = row["type"]

        if race_name not in race_dfs:
            race_dfs[race_name] = pd.read_parquet(PROCESSED / f"{race_name}_labelled.parquet")
        
        df_r = race_dfs[race_name]
        df_drv = df_r[df_r["driver_number"] == drv]

        rendered = render_audit_plot(
            df_drv, lap, sd, ed, plot_path,
            title=f"Telemetry Window: {sample_id}"
        )

        audit_rows.append({
            "event_id": sample_id,
            "race": race_name,
            "driver": drv,
            "lap": lap,
            "start_distance": sd,
            "plot_path": f"ml/reports/paper/audit_plots/{plot_name}",
            "human_label_clip_yes_no": "",
            "human_notes": "",
        })

        key_rows.append({
            "event_id": sample_id,
            "race": race_name,
            "driver": drv,
            "lap": lap,
            "start_distance": sd,
            "true_stratum_type": ev_type,
        })

    del race_dfs
    gc.collect()

    audit_sheet_csv = REPORT_DIR / "audit_sheet.csv"
    audit_key_csv   = REPORT_DIR / "audit_key.csv"

    pd.DataFrame(audit_rows).to_csv(audit_sheet_csv, index=False)
    pd.DataFrame(key_rows).to_csv(audit_key_csv, index=False)
    log.info("Saved blind audit sheet to %s (150 samples)", audit_sheet_csv)
    log.info("Saved blind audit key to %s", audit_key_csv)

    # -----------------------------------------------------------------------
    # Sensitivity & Comparison Report
    # -----------------------------------------------------------------------
    df_sens = pd.DataFrame(sensitivity_records)
    df_v1v2 = pd.DataFrame(v1_v2_counts)

    sens_csv = REPORT_DIR / "label_sensitivity.csv"
    v1v2_csv = REPORT_DIR / "v1_v2_clip_comparison.csv"
    df_sens.to_csv(sens_csv, index=False)
    df_v1v2.to_csv(v1v2_csv, index=False)

    log.info("Stage 2 COMPLETE in %.1fs", time.time() - t0)


if __name__ == "__main__":
    label_all_races()
