"""Study 2 ICD-10 AUC analysis: positive (HF/sepsis) vs negative controls.

Computes ROC AUC of BioViL-T disease_score (at each time anchor before admittime)
vs ground-truth label (1 = ICD-10 positive, 0 = negative control).

This is the definitive temporal detection curve: does the VLM discriminate
positive from negative patients at X hours before first ICD-10-coded admission?

Outputs (in --output-dir):
    icd10_auc_vs_hours.csv             — one row per (disease, anchor_hour)
    icd10_auc_vs_hours.png             — AUC curve plot
    icd10_auc_summary.json             — cohort counts + AUC highlights

Example::

    python code/study2/scripts/compute_icd10_auc.py \\
        --inference-csv  data/MIMIC-CXR/csv/study2_results_icd10/study2_inference_results.csv \\
        --cohort-csv     data/MIMIC-CXR/csv/study2_cohort_icd10/study2_icd10_cohort.csv \\
        --output-dir     data/MIMIC-CXR/csv/study2_results_icd10/auc_analysis
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

_CODE_DIR = Path(__file__).resolve().parents[2]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from study_logging import configure_study_logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data"

# Hours before admittime at which to compute AUC landmarks
ANCHOR_HOURS = [
    336, 312, 288, 264, 240, 216, 192, 168,
    144, 120, 96, 72, 48, 36, 24, 18, 12, 8, 6, 3,
]


def _pick_landmark_score(grp: pd.DataFrame, anchor_h: float) -> float | None:
    """Study score closest to ``anchor_h`` hours before admittime, preferring studies ≥ anchor_h."""
    grp = grp.copy()
    grp["hours_before_diagnosis"] = pd.to_numeric(grp["hours_before_diagnosis"], errors="coerce")
    grp = grp.dropna(subset=["hours_before_diagnosis", "disease_score"])
    if grp.empty:
        return None
    eligible = grp[grp["hours_before_diagnosis"] >= anchor_h - 1e-9]
    pool = eligible if not eligible.empty else grp
    dist = (pool["hours_before_diagnosis"] - anchor_h).abs()
    return float(pool.loc[dist.idxmin(), "disease_score"])


def compute_auc_vs_hours(
    inference_df: pd.DataFrame,
    *,
    anchor_hours: list[float],
    min_n_per_class: int = 5,
) -> pd.DataFrame:
    """One AUC per (disease_type, anchor_hour) using positive/negative labels."""
    if "label" not in inference_df.columns:
        raise ValueError("inference_df must have a 'label' column (1=positive, 0=negative).")

    keys = ["subject_id", "hadm_id", "disease_type"]
    # Patient-level label (constant per event)
    patient_labels = (
        inference_df[keys + ["label"]]
        .drop_duplicates(keys)
        .reset_index(drop=True)
    )

    rows: list[dict[str, object]] = []
    for disease, dis_inf in inference_df.groupby("disease_type"):
        labels_dis = patient_labels[patient_labels["disease_type"].astype(str) == str(disease)]
        n_pos = int((labels_dis["label"] == 1).sum())
        n_neg = int((labels_dis["label"] == 0).sum())
        logger.info("%s: %d positives, %d negatives in inference", disease, n_pos, n_neg)

        if n_pos < min_n_per_class or n_neg < min_n_per_class:
            logger.warning(
                "%s: insufficient class balance for AUC (pos=%d neg=%d min=%d)",
                disease, n_pos, n_neg, min_n_per_class,
            )
            continue

        for anchor_h in anchor_hours:
            score_rows: list[dict[str, object]] = []
            for key, grp in dis_inf.groupby(keys, sort=False):
                s = _pick_landmark_score(grp, float(anchor_h))
                if s is not None:
                    sid, hid, dis_t = key
                    score_rows.append({"subject_id": int(sid), "hadm_id": int(hid), "disease_type": str(dis_t), "score": s})

            if not score_rows:
                continue
            scores_df = pd.DataFrame(score_rows)
            m = scores_df.merge(labels_dis[keys + ["label"]], on=keys, how="inner")
            if m["label"].nunique() < 2 or len(m) < 2 * min_n_per_class:
                continue

            auc = float(roc_auc_score(m["label"].astype(int), m["score"]))
            rows.append({
                "disease_type": str(disease),
                "hours_before_admittime": float(anchor_h),
                "roc_auc": auc,
                "n_positive": int((m["label"] == 1).sum()),
                "n_negative": int((m["label"] == 0).sum()),
            })

    return pd.DataFrame(rows)


def plot_auc_curve(auc_df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"heart_failure": "#1f77b4", "sepsis": "#d62728"}

    for disease, grp in auc_df.groupby("disease_type"):
        g = grp.sort_values("hours_before_admittime", ascending=False)
        col = colors.get(str(disease), None)
        ax.plot(
            g["hours_before_admittime"],
            g["roc_auc"],
            marker="o",
            markersize=4,
            label=str(disease).replace("_", " ").title(),
            color=col,
        )

    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, label="Chance (AUC=0.5)")
    ax.set_xlabel("Hours before first ICD-10-coded admission (larger = further in advance)")
    ax.set_ylabel("ROC AUC (BioViL-T score: positive vs negative patients)")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("BioViL-T early detection: AUC vs hours before ICD-10 diagnosis")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(48))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved AUC plot: %s", out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute AUC vs hours-before-admittime for ICD-10-anchored cohort."
    )
    parser.add_argument(
        "--inference-csv",
        type=Path,
        default=DATA_ROOT / "MIMIC-CXR/csv/study2_results_icd10/study2_inference_results.csv",
        help="Output of run_inference.py on the ICD-10 cohort.",
    )
    parser.add_argument(
        "--cohort-csv",
        type=Path,
        default=DATA_ROOT / "MIMIC-CXR/csv/study2_cohort_icd10/study2_icd10_cohort.csv",
        help="Cohort CSV from build_index_cohort_icd10.py (must have 'label' column).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_ROOT / "MIMIC-CXR/csv/study2_results_icd10/auc_analysis",
    )
    parser.add_argument(
        "--min-n-per-class",
        type=int,
        default=5,
        help="Minimum patients per class (positive/negative) for AUC computation.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    configure_study_logging(study_name="study2_icd10_auc", level=getattr(logging, args.log_level))

    for name, p in [("inference_csv", args.inference_csv), ("cohort_csv", args.cohort_csv)]:
        if not p.exists():
            logger.error("Required file missing — %s: %s", name, p)
            sys.exit(1)

    logger.info("Loading inference results: %s", args.inference_csv)
    inference = pd.read_csv(args.inference_csv)
    logger.info("Inference rows: %d", len(inference))

    if "label" not in inference.columns:
        # Join label from cohort CSV
        logger.info("'label' not in inference CSV — joining from cohort: %s", args.cohort_csv)
        cohort = pd.read_csv(
            args.cohort_csv,
            usecols=["subject_id", "hadm_id", "disease_type", "label"],
            dtype={"subject_id": "int64", "hadm_id": "int64"},
        )
        inference["subject_id"] = pd.to_numeric(inference["subject_id"], errors="coerce").astype("Int64")
        inference["hadm_id"] = pd.to_numeric(inference["hadm_id"], errors="coerce").astype("Int64")
        cohort["subject_id"] = cohort["subject_id"].astype("Int64")
        cohort["hadm_id"] = cohort["hadm_id"].astype("Int64")
        inference = inference.merge(
            cohort,
            on=["subject_id", "hadm_id", "disease_type"],
            how="left",
        )
        n_missing = inference["label"].isna().sum()
        if n_missing:
            logger.warning("Dropping %d inference rows with no matching label in cohort.", n_missing)
        inference = inference.dropna(subset=["label"]).copy()
        inference["label"] = inference["label"].astype(int)
        logger.info("Inference rows after label join: %d", len(inference))

    auc_df = compute_auc_vs_hours(
        inference,
        anchor_hours=ANCHOR_HOURS,
        min_n_per_class=args.min_n_per_class,
    )

    if auc_df.empty:
        logger.error("AUC computation produced no rows — check cohort label balance.")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    auc_csv = args.output_dir / "icd10_auc_vs_hours.csv"
    auc_df.to_csv(auc_csv, index=False)
    logger.info("Wrote AUC CSV: %s", auc_csv)

    plot_auc_curve(auc_df, args.output_dir / "icd10_auc_vs_hours.png")

    # Summary stats
    summary: dict[str, object] = {}
    for disease, g in auc_df.groupby("disease_type"):
        g_sorted = g.sort_values("hours_before_admittime")
        max_row = g.loc[g["roc_auc"].idxmax()]
        close_row = g.loc[(g["hours_before_admittime"] - 24).abs().idxmin()]
        summary[str(disease)] = {
            "n_positive": int(g["n_positive"].iloc[0]),
            "n_negative": int(g["n_negative"].iloc[0]),
            "auc_at_3h": float(g.loc[g["hours_before_admittime"] == 3.0, "roc_auc"].values[0])
            if 3.0 in g["hours_before_admittime"].values else None,
            "auc_at_24h": float(close_row["roc_auc"]),
            "auc_at_24h_anchor": float(close_row["hours_before_admittime"]),
            "peak_auc": float(max_row["roc_auc"]),
            "peak_auc_at_hours": float(max_row["hours_before_admittime"]),
        }

    summary_path = args.output_dir / "icd10_auc_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote summary: %s", summary_path)

    print("\n=== ICD-10 AUC vs Hours Before Admittime ===")
    for disease, s in summary.items():
        print(f"\n{disease.replace('_', ' ').upper()}")
        print(f"  N positive / negative: {s['n_positive']} / {s['n_negative']}")
        if s["auc_at_3h"] is not None:
            print(f"  AUC at  3h:  {s['auc_at_3h']:.3f}")
        print(f"  AUC at ~24h: {s['auc_at_24h']:.3f} (anchor={s['auc_at_24h_anchor']:.0f}h)")
        print(f"  Peak AUC:    {s['peak_auc']:.3f} at {s['peak_auc_at_hours']:.0f}h")

    print(f"\nResults: {args.output_dir}")


if __name__ == "__main__":
    run()
