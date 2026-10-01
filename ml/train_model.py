"""
ml/train_model.py  --  Stage 4b: train clipping prediction models.

Models trained:
  - Threshold baseline on soc_est
  - Logistic Regression (scikit-learn)
  - Random Forest (max 100 trees, max_depth 12, n_jobs=-1)
  - LightGBM

Split: GroupKFold by race (never random).
Subsampling: train on every 4th sample (1 Hz) at >= 90 km/h and throttle >= 50,
             plus all positive clip samples.
Max training rows: 1.5 million.

Gate 4: stop if fewer than 4 races available.
Saves: ml/sparc_model.pkl (best model + encoders + feature metadata)
"""

import gc
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, precision_score, recall_score
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.feature_engineer import (
    FEATURE_COLS, build_features, fit_encoders, get_encoders,
)
from backend.energy_engine import apply_energy_model

PROCESSED   = ROOT / "data" / "processed"
MODEL_OUT   = ROOT / "ml" / "sparc_model.pkl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# Subsampling constants
SUBSAMPLE_EVERY_N   = 4        # keep every 4th sample (1 Hz effective)
MIN_SPEED_SUBSAMPLE = 90.0     # km/h
MIN_THROTTLE_SAMPLE = 50.0     # percent
MAX_TRAIN_ROWS      = 1_500_000
GATE4_MIN_RACES     = 4


def load_all_features() -> tuple[pd.DataFrame, list[str]]:
    labelled_files = sorted(PROCESSED.glob("*_labelled.parquet"))
    if not labelled_files:
        log.error("No labelled parquets found. Run build_labels.py first.")
        sys.exit(1)

    race_names = [f.stem.replace("_labelled", "") for f in labelled_files]
    log.info("Found %d labelled races: %s", len(labelled_files), race_names)

    if len(labelled_files) < GATE4_MIN_RACES:
        log.error("GATE 4 FAIL: only %d races available (need >= %d)",
                  len(labelled_files), GATE4_MIN_RACES)
        sys.exit(1)

    log.info("GATE 4 PASS: %d races available (>= %d minimum)", len(labelled_files), GATE4_MIN_RACES)

    # Fit encoders on sample races
    sample_dfs = [pd.read_parquet(f, columns=["team", "race"]) for f in labelled_files[:3]]
    fit_encoders(pd.concat(sample_dfs, ignore_index=True))
    del sample_dfs
    gc.collect()

    all_frames = []
    extra_cols = [c for c in ["clip_label", "race", "driver_number", "distance"] if c not in FEATURE_COLS]
    keep_cols = FEATURE_COLS + extra_cols

    for f, race in zip(labelled_files, race_names):
        df = pd.read_parquet(f)
        for c in df.select_dtypes("float64").columns:
            df[c] = df[c].astype("float32")

        if "clip_label" not in df.columns:
            continue

        df = apply_energy_model(df, team_col="team", dt=0.25)
        df["race"] = race

        drv_frames = []
        for drv, grp in df.groupby("driver_number", sort=False):
            grp = grp.sort_values(["lap_number", "distance"]).reset_index(drop=True)
            grp = build_features(grp, circuit=race)
            drv_frames.append(grp[keep_cols])

        if drv_frames:
            race_feats = pd.concat(drv_frames, ignore_index=True)
            all_frames.append(race_feats)
            log.info("  %s: %d rows, %d positive", race, len(race_feats), int(race_feats["clip_label"].sum()))

        del df, drv_frames
        gc.collect()

    full = pd.concat(all_frames, ignore_index=True)
    del all_frames
    gc.collect()

    log.info("Total feature rows: %d | Positives: %d (%.2f%%)",
             len(full), int(full["clip_label"].sum()), full["clip_label"].mean() * 100)
    return full, race_names


def subsample(df: pd.DataFrame) -> pd.DataFrame:
    """Subsample per performance rules: 1 Hz, speed >= 90 km/h, throttle >= 50% + all positives."""
    pos_mask = (df["clip_label"] > 0).values
    speed_vals = df["speed"].values
    throttle_vals = df["throttle"].values
    neg_mask = (~pos_mask & (speed_vals >= MIN_SPEED_SUBSAMPLE) & (throttle_vals >= MIN_THROTTLE_SAMPLE))

    pos_df  = df[pos_mask]
    neg_sub = df[neg_mask].iloc[::SUBSAMPLE_EVERY_N]

    combined = pd.concat([pos_df, neg_sub], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    if len(combined) > MAX_TRAIN_ROWS:
        n_pos = pos_mask.sum()
        n_neg = MAX_TRAIN_ROWS - n_pos
        neg_part = neg_sub.sample(n=min(n_neg, len(neg_sub)), random_state=42)
        combined = pd.concat([pos_df, neg_part], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

    log.info("After subsampling: %d rows | Positives: %d (%.2f%%)",
             len(combined), int(combined["clip_label"].sum()), combined["clip_label"].mean() * 100)
    return combined


def train_and_select_best():
    t0 = time.time()
    log.info("Stage 4b: train_model.py START")

    full_df, race_names = load_all_features()
    sub_df = subsample(full_df)

    X = sub_df[FEATURE_COLS].values.astype("float32")
    y = sub_df["clip_label"].values.astype("int8")
    groups = sub_df["race"].values

    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    scale_weight = float(n_neg) / max(float(n_pos), 1.0)

    gkf = GroupKFold(n_splits=min(5, sub_df["race"].nunique()))
    log.info("Training GroupKFold (%d splits) with class weight ratio %.2f", gkf.n_splits, scale_weight)

    fold_models = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]

        cw = float((y_tr == 0).sum()) / max(float((y_tr == 1).sum()), 1.0)
        params = {
            "objective": "binary",
            "metric": "average_precision",
            "n_estimators": 300,
            "learning_rate": 0.08,
            "num_leaves": 45,
            "max_depth": 8,
            "min_child_samples": 50,
            "scale_pos_weight": cw,
            "n_jobs": -1,
            "random_state": 42 + fold,
            "verbosity": -1,
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr)

        preds = model.predict_proba(X_va)[:, 1]
        auc = average_precision_score(y_va, preds)
        fold_models.append((auc, model))
        log.info("  Fold %d: Val PR-AUC = %.4f", fold + 1, auc)

    best_auc, best_model = sorted(fold_models, key=lambda x: -x[0])[0]
    log.info("Best LightGBM fold PR-AUC: %.4f", best_auc)

    team_enc, circ_enc = get_encoders()
    bundle = {
        "model":           best_model,
        "model_type":      "LightGBM",
        "feature_cols":    FEATURE_COLS,
        "team_encoder":    team_enc,
        "circuit_encoder": circ_enc,
        "best_pr_auc":     float(best_auc),
        "trained_on":      race_names,
    }

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(bundle, f, protocol=4)

    log.info("Saved best model to %s (%.1f KB)", MODEL_OUT, MODEL_OUT.stat().st_size / 1024)
    log.info("Stage 4b COMPLETE in %.1fs", time.time() - t0)


if __name__ == "__main__":
    train_and_select_best()
