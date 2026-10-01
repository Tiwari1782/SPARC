"""
ml/audit_agreement.py -- Evaluate Human vs Detector Agreement on the Audit Sheet.

Calculates:
  - Detector precision vs Human labels
  - Cohen's kappa score (for single or double-annotator comparisons)

Zero emojis anywhere.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "ml" / "reports" / "paper"
AUDIT_SHEET = REPORT_DIR / "audit_sheet.csv"
AUDIT_KEY   = REPORT_DIR / "audit_key.csv"


def evaluate_audit_agreement():
    if not AUDIT_SHEET.exists() or not AUDIT_KEY.exists():
        print(f"Audit sheet or key missing in {REPORT_DIR}")
        return

    df_sheet = pd.read_csv(AUDIT_SHEET)
    df_key   = pd.read_csv(AUDIT_KEY)

    merged = pd.merge(df_sheet, df_key, on="event_id")

    human_filled = merged["human_label_clip_yes_no"].dropna().astype(str).str.strip().str.lower()
    valid_mask = human_filled.isin(["yes", "no", "1", "0", "true", "false", "y", "n"])

    if not valid_mask.any():
        print("Audit sheet has not been filled with human labels yet.")
        print(f"Please fill 'human_label_clip_yes_no' (yes/no) in {AUDIT_SHEET}")
        print(f"Total samples waiting for audit: {len(df_sheet)}")
        return

    sub = merged[valid_mask].copy()
    y_human = sub["human_label_clip_yes_no"].astype(str).str.strip().str.lower().isin(["yes", "1", "true", "y"]).astype(int)
    
    # Detector v2 positive is ground truth in key
    y_detector = (sub["true_stratum_type"] == "detector_v2_positive").astype(int)

    precision = float((y_human & y_detector).sum()) / max(float(y_detector.sum()), 1.0)
    kappa = cohen_kappa_score(y_human, y_detector)

    print("=========================================================")
    print("SPARC Human vs Detector Audit Results")
    print("=========================================================")
    print(f"Audited Samples: {len(sub)} / {len(merged)}")
    print(f"Detector v2 Precision (vs Human): {precision:.4f} ({precision*100:.2f}%)")
    print(f"Cohen's Kappa (Human vs Detector): {kappa:.4f}")
    print("=========================================================")


if __name__ == "__main__":
    evaluate_audit_agreement()
