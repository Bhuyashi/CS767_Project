"""Study 2: Compare BioViL-T disease scores between last and 2nd-last CXR scans.

For each cohort event (subject, hadm, disease_type) with ≥2 scans in the
pre-diagnosis window, extracts:
  - last scan:     highest seq_index (closest to diagnosis)
  - 2nd-last scan: seq_index = max - 1

Uses CheXpert labels as per-scan ground truth:
  - Heart failure : Edema = 1 → positive, = 0 → negative
  - Sepsis        : Pneumonia = 1 OR Consolidation = 1 → positive; both 0 → negative

Uncertain (-1) and blank (NaN) CheXpert values are dropped from AUC calculations.

Outputs (under --output-dir)
----------------------------
  last_vs_prior_scan_data.csv          — merged scan data with CheXpert labels
  last_vs_prior_auc_summary.csv        — AUC, n, mean score per (disease, position)
  last_vs_prior_paired_analysis.csv    — per-patient paired score comparison
  figures/last_vs_prior_roc_curves.png       — overlaid ROC curves
  figures/last_vs_prior_score_distributions.png — boxplots by position and CheXpert label

Run from repo root:
    python code/study2/scripts/last_vs_prior_scan_auc.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_CODE_DIR = Path(__file__).resolve().parents[2]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from study_logging import configure_study_logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data"


# ---------------------------------------------------------------------------
# CheXpert label helpers
# ---------------------------------------------------------------------------

def _load_chexpert(chexpert_csv: Path) -> pd.DataFrame:
    cols = ["study_id", "Edema", "Pneumonia", "Consolidation"]
    df = pd.read_csv(chexpert_csv, usecols=cols, dtype={"study_id": "int64"})
    return df


def _chexpert_label_hf(edema) -> float:
    """1=positive, 0=negative, NaN=uncertain/blank."""
    if pd.isna(edema):
        return np.nan
    v = float(edema)
    if v == 1.0:
        return 1.0
    if v == 0.0:
        return 0.0
    return np.nan  # -1 (uncertain)


def _chexpert_label_sepsis(pneumonia, consolidation) -> float:
    """1 if either Pneumonia or Consolidation is positive, 0 if both are negative."""
    p = float(pneumonia) if not pd.isna(pneumonia) else np.nan
    c = float(consolidation) if not pd.isna(consolidation) else np.nan
    if p == 1.0 or c == 1.0:
        return 1.0
    if (p == 0.0 or pd.isna(p)) and (c == 0.0 or pd.isna(c)):
        if p == 0.0 or c == 0.0:  # at least one explicit 0
            return 0.0
    return np.nan


def _add_chexpert_label(df: pd.DataFrame) -> pd.DataFrame:
    labels: list[float] = []
    for _, row in df.iterrows():
        if str(row["disease_type"]) == "heart_failure":
            labels.append(_chexpert_label_hf(row.get("Edema")))
        else:
            labels.append(_chexpert_label_sepsis(row.get("Pneumonia"), row.get("Consolidation")))
    out = df.copy()
    out["chexpert_label"] = labels
    return out


# ---------------------------------------------------------------------------
# Scan extraction
# ---------------------------------------------------------------------------

def extract_last_and_prior_scans(
    inference_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For each cohort event with ≥2 scans return (last_df, prior_df).

    last_df  : one row per event — the scan with highest seq_index (closest to dx).
    prior_df : one row per event — the scan at seq_index = max - 1.
    """
    keys = ["subject_id", "hadm_id", "disease_type"]
    last_rows: list[pd.Series] = []
    prior_rows: list[pd.Series] = []

    for _, grp in inference_df.groupby(keys, sort=False):
        grp = grp.sort_values("seq_index")
        if len(grp) < 2:
            continue
        last = grp.iloc[-1].copy()
        prior = grp.iloc[-2].copy()
        last["scan_position"] = "last"
        prior["scan_position"] = "2nd_last"
        last_rows.append(last)
        prior_rows.append(prior)

    last_df = pd.DataFrame(last_rows).reset_index(drop=True)
    prior_df = pd.DataFrame(prior_rows).reset_index(drop=True)
    logger.info(
        "Extracted %d cohort events with ≥2 scans (last + 2nd-last pairs).", len(last_df)
    )
    return last_df, prior_df


# ---------------------------------------------------------------------------
# AUC / ROC helpers
# ---------------------------------------------------------------------------

def _compute_auc(labels: np.ndarray, scores: np.ndarray, min_n: int = 4) -> float:
    from sklearn.metrics import roc_auc_score

    valid = ~(np.isnan(labels) | np.isnan(scores))
    y = labels[valid].astype(int)
    s = scores[valid]
    if len(y) < min_n or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def _compute_roc_curve(
    labels: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray | None, np.ndarray | None, float]:
    from sklearn.metrics import roc_curve

    valid = ~(np.isnan(labels) | np.isnan(scores))
    y = labels[valid].astype(int)
    s = scores[valid]
    if len(np.unique(y)) < 2:
        return None, None, float("nan")
    fpr, tpr, _ = roc_curve(y, s)
    auc = _compute_auc(labels, scores)
    return fpr, tpr, auc


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def build_auc_summary(last_df: pd.DataFrame, prior_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for df, position in [(last_df, "last"), (prior_df, "2nd_last")]:
        for disease, grp in df.groupby("disease_type"):
            valid = grp.dropna(subset=["chexpert_label", "disease_score"])
            y = valid["chexpert_label"].to_numpy(dtype=float)
            s = valid["disease_score"].to_numpy(dtype=float)
            auc = _compute_auc(y, s)
            rows.append(
                {
                    "disease_type": str(disease),
                    "scan_position": position,
                    "n_scans_total": int(len(grp)),
                    "n_with_chexpert_label": int(len(valid)),
                    "n_chexpert_positive": int((y == 1).sum()),
                    "n_chexpert_negative": int((y == 0).sum()),
                    "mean_disease_score_all": round(float(grp["disease_score"].mean()), 4),
                    "mean_disease_score_chexpert_pos": round(
                        float(valid[valid["chexpert_label"] == 1.0]["disease_score"].mean()), 4
                    ) if (y == 1).any() else np.nan,
                    "mean_disease_score_chexpert_neg": round(
                        float(valid[valid["chexpert_label"] == 0.0]["disease_score"].mean()), 4
                    ) if (y == 0).any() else np.nan,
                    "roc_auc": round(auc, 4) if np.isfinite(auc) else np.nan,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Paired comparison
# ---------------------------------------------------------------------------

def build_paired_comparison(last_df: pd.DataFrame, prior_df: pd.DataFrame) -> pd.DataFrame:
    keys = ["subject_id", "hadm_id", "disease_type"]
    merged = last_df[keys + ["disease_score", "chexpert_label", "hours_before_diagnosis"]].merge(
        prior_df[keys + ["disease_score", "chexpert_label", "hours_before_diagnosis"]],
        on=keys,
        suffixes=("_last", "_prior"),
    )
    merged["score_delta"] = merged["disease_score_last"] - merged["disease_score_prior"]
    merged["last_gt_prior"] = (merged["score_delta"] > 0).astype(int)
    return merged.reset_index(drop=True)


def _wilcoxon_summary(paired_df: pd.DataFrame, disease: str) -> dict:
    sub = paired_df[paired_df["disease_type"].astype(str) == disease].dropna(subset=["score_delta"])
    n = int(len(sub))
    result: dict = {
        "disease_type": disease,
        "n_pairs": n,
        "median_score_last": round(float(sub["disease_score_last"].median()), 4) if n else np.nan,
        "median_score_prior": round(float(sub["disease_score_prior"].median()), 4) if n else np.nan,
        "median_score_delta": round(float(sub["score_delta"].median()), 4) if n else np.nan,
        "pct_last_gt_prior": round(float(sub["last_gt_prior"].mean() * 100), 1) if n else np.nan,
    }
    if n >= 5:
        try:
            from scipy.stats import wilcoxon

            stat, p = wilcoxon(sub["score_delta"].to_numpy(), alternative="greater")
            result["wilcoxon_statistic"] = round(float(stat), 4)
            result["wilcoxon_p_value"] = round(float(p), 6)
            result["wilcoxon_note"] = "H1: last score > prior score"
        except ImportError:
            result["wilcoxon_note"] = "scipy not installed; skipped"
        except Exception as e:
            result["wilcoxon_note"] = f"error: {e}"
    else:
        result["wilcoxon_note"] = "insufficient_n"
    return result


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_roc_curves(
    last_df: pd.DataFrame, prior_df: pd.DataFrame, out_path: Path
) -> None:
    import matplotlib.pyplot as plt

    diseases = sorted(
        set(last_df["disease_type"].astype(str).unique())
        | set(prior_df["disease_type"].astype(str).unique())
    )
    n = max(len(diseases), 1)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)

    for col, disease in enumerate(diseases):
        ax = axes[0][col]
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)

        for df, label, color in [
            (last_df, "Last scan", "steelblue"),
            (prior_df, "2nd-last scan", "darkorange"),
        ]:
            sub = df[df["disease_type"].astype(str) == disease].dropna(
                subset=["chexpert_label", "disease_score"]
            )
            y = sub["chexpert_label"].to_numpy(dtype=float)
            s = sub["disease_score"].to_numpy(dtype=float)
            fpr, tpr, auc = _compute_roc_curve(y, s)
            if fpr is None:
                ax.text(
                    0.5, 0.5,
                    f"{label}: insufficient data\n(n={len(sub)}, "
                    f"pos={int((y==1).sum())}, neg={int((y==0).sum())})",
                    ha="center", va="center", transform=ax.transAxes, fontsize=8,
                )
            else:
                ax.plot(
                    fpr, tpr,
                    color=color,
                    linewidth=2,
                    label=f"{label} (AUC={auc:.3f}, n={len(sub)})",
                )

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        title = disease.replace("_", " ").title()
        gt_note = "Edema" if disease == "heart_failure" else "Pneumonia / Consolidation"
        ax.set_title(f"{title}\nGround truth: CheXpert {gt_note}")
        ax.legend(loc="lower right", fontsize=9)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Last vs 2nd-Last CXR Scan: BioViL-T Disease Score ROC", fontsize=13, y=1.02
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("ROC curves saved: %s", out_path)


def plot_score_distributions(
    last_df: pd.DataFrame, prior_df: pd.DataFrame, out_path: Path
) -> None:
    import matplotlib.pyplot as plt

    diseases = sorted(
        set(last_df["disease_type"].astype(str).unique())
        | set(prior_df["disease_type"].astype(str).unique())
    )
    n = max(len(diseases), 1)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), squeeze=False)
    palette = ["#5b9bd5", "#aecde8", "#ed7d31", "#f5c09a"]

    for col, disease in enumerate(diseases):
        ax = axes[0][col]
        series: list[np.ndarray] = []
        tick_labels: list[str] = []

        for df, pos_name in [(last_df, "Last"), (prior_df, "2nd-last")]:
            sub = df[df["disease_type"].astype(str) == disease]
            for lbl, lname in [(1.0, "CheXpert+"), (0.0, "CheXpert−")]:
                s = sub[sub["chexpert_label"] == lbl]["disease_score"].dropna().to_numpy()
                series.append(s)
                tick_labels.append(f"{pos_name}\n{lname}\n(n={len(s)})")

        valid_series = [s for s in series if len(s) > 0]
        valid_labels = [l for s, l in zip(series, tick_labels) if len(s) > 0]
        valid_positions = [i + 1 for i, s in enumerate(series) if len(s) > 0]

        if valid_series:
            bp = ax.boxplot(
                valid_series,
                positions=valid_positions,
                tick_labels=valid_labels,
                patch_artist=True,
                showfliers=False,
                widths=0.55,
            )
            for patch, color in zip(bp["boxes"], palette):
                patch.set_facecolor(color)
                patch.set_alpha(0.75)

        ax.set_ylabel("disease_score")
        ax.set_ylim([0, 1])
        ax.set_title(f"{disease.replace('_', ' ').title()}\nScore by Scan Position & CheXpert Label")
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "BioViL-T Disease Score: Last vs 2nd-Last Scan", fontsize=13, y=1.02
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Score distributions saved: %s", out_path)


def plot_paired_scatter(paired_df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    diseases = sorted(paired_df["disease_type"].astype(str).unique())
    n = max(len(diseases), 1)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)

    for col, disease in enumerate(diseases):
        ax = axes[0][col]
        sub = paired_df[paired_df["disease_type"].astype(str) == disease].dropna(
            subset=["disease_score_last", "disease_score_prior"]
        )
        if sub.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        colors = sub["chexpert_label_last"].map(
            {1.0: "steelblue", 0.0: "darkorange"}
        ).fillna("gray")

        ax.scatter(
            sub["disease_score_prior"],
            sub["disease_score_last"],
            c=colors,
            alpha=0.7,
            edgecolors="none",
            s=50,
        )
        lims = [0, 1]
        ax.plot(lims, lims, "k--", alpha=0.4, linewidth=1)
        ax.set_xlabel("2nd-last scan score")
        ax.set_ylabel("Last scan score")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        pct = float(sub["last_gt_prior"].mean() * 100)
        ax.set_title(
            f"{disease.replace('_', ' ').title()}\n"
            f"Last > 2nd-last: {pct:.0f}% of patients"
        )
        ax.grid(True, alpha=0.3)
        from matplotlib.patches import Patch
        legend_handles = [
            Patch(facecolor="steelblue", label="CheXpert+ (last scan)"),
            Patch(facecolor="darkorange", label="CheXpert− (last scan)"),
            Patch(facecolor="gray", label="CheXpert unknown"),
        ]
        ax.legend(handles=legend_handles, fontsize=8, loc="lower right")

    fig.suptitle("Paired Score: Last vs 2nd-Last Scan per Patient", fontsize=12, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Paired scatter saved: %s", out_path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_analysis(
    inference_csv: Path,
    chexpert_csv: Path,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading inference results: %s", inference_csv)
    inference = pd.read_csv(inference_csv)
    inference["study_datetime"] = pd.to_datetime(inference["study_datetime"], errors="coerce")
    logger.info("Inference rows: %d  (cohort events: %d)", len(inference),
                inference[["subject_id", "hadm_id", "disease_type"]].drop_duplicates().shape[0])

    logger.info("Loading CheXpert: %s", chexpert_csv)
    chexpert = _load_chexpert(chexpert_csv)

    # Extract scan positions
    last_df, prior_df = extract_last_and_prior_scans(inference)
    if last_df.empty:
        logger.error("No cohort events with ≥2 scans found. Cannot proceed.")
        return {}

    # Join CheXpert labels (study_id is the current scan's study_id)
    cx_cols = ["study_id", "Edema", "Pneumonia", "Consolidation"]
    last_df = last_df.merge(chexpert[cx_cols], on="study_id", how="left")
    prior_df = prior_df.merge(chexpert[cx_cols], on="study_id", how="left")

    last_df = _add_chexpert_label(last_df)
    prior_df = _add_chexpert_label(prior_df)

    n_last_labeled = int(last_df["chexpert_label"].notna().sum())
    n_prior_labeled = int(prior_df["chexpert_label"].notna().sum())
    logger.info(
        "CheXpert labels resolved — last: %d/%d  2nd-last: %d/%d",
        n_last_labeled, len(last_df), n_prior_labeled, len(prior_df),
    )

    # --- Debug CSV ---
    debug_cols = [
        "subject_id", "hadm_id", "disease_type", "study_id", "seq_index",
        "hours_before_diagnosis", "disease_score", "scan_position",
        "Edema", "Pneumonia", "Consolidation", "chexpert_label",
    ]
    combined = pd.concat(
        [
            last_df[[c for c in debug_cols if c in last_df.columns]],
            prior_df[[c for c in debug_cols if c in prior_df.columns]],
        ],
        ignore_index=True,
    ).sort_values(["subject_id", "hadm_id", "disease_type", "seq_index"])
    scan_data_path = output_dir / "last_vs_prior_scan_data.csv"
    combined.to_csv(scan_data_path, index=False)

    # --- AUC summary ---
    summary = build_auc_summary(last_df, prior_df)
    summary_path = output_dir / "last_vs_prior_auc_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info("AUC summary:\n%s", summary.to_string(index=False))

    # --- Paired comparison ---
    paired = build_paired_comparison(last_df, prior_df)
    paired_path = output_dir / "last_vs_prior_paired_analysis.csv"
    paired.to_csv(paired_path, index=False)

    wilcoxon_rows = []
    for disease in sorted(paired["disease_type"].astype(str).unique()):
        w = _wilcoxon_summary(paired, disease)
        wilcoxon_rows.append(w)
        logger.info("Wilcoxon [%s]: %s", disease, w)
    wilcoxon_path = output_dir / "last_vs_prior_wilcoxon.csv"
    pd.DataFrame(wilcoxon_rows).to_csv(wilcoxon_path, index=False)

    # --- Figures ---
    fig_dir = output_dir / "figures"
    roc_path = fig_dir / "last_vs_prior_roc_curves.png"
    dist_path = fig_dir / "last_vs_prior_score_distributions.png"
    scatter_path = fig_dir / "last_vs_prior_paired_scatter.png"

    plot_roc_curves(last_df, prior_df, roc_path)
    plot_score_distributions(last_df, prior_df, dist_path)
    plot_paired_scatter(paired, scatter_path)

    logger.info("Analysis complete. Outputs in %s", output_dir)
    return {
        "scan_data": scan_data_path,
        "auc_summary": summary_path,
        "paired_analysis": paired_path,
        "wilcoxon": wilcoxon_path,
        "roc_curves": roc_path,
        "score_distributions": dist_path,
        "paired_scatter": scatter_path,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare BioViL-T disease scores: last vs 2nd-last CXR scan."
    )
    p.add_argument(
        "--inference-csv",
        type=Path,
        default=DATA_ROOT / "MIMIC-CXR/csv/study2_results/study2_inference_results.csv",
        help="Path to study2_inference_results.csv from run_inference.py.",
    )
    p.add_argument(
        "--chexpert-csv",
        type=Path,
        default=DATA_ROOT / "MIMIC-CXR/csv/mimic-cxr-2.0.0-chexpert.csv",
        help="MIMIC-CXR CheXpert labels CSV.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_ROOT / "MIMIC-CXR/csv/study2_last_vs_prior_auc",
        help="Output directory for CSVs and figures.",
    )
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return p.parse_args()


def run() -> None:
    args = parse_args()
    configure_study_logging(
        study_name="study2_last_vs_prior", level=getattr(logging, args.log_level)
    )

    missing = [
        name
        for name, p in [("inference-csv", args.inference_csv), ("chexpert-csv", args.chexpert_csv)]
        if not p.exists()
    ]
    if missing:
        for m in missing:
            logger.error("File not found: --%s = %s", m, getattr(args, m.replace("-", "_")))
        sys.exit(1)

    run_analysis(
        inference_csv=args.inference_csv,
        chexpert_csv=args.chexpert_csv,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    run()
