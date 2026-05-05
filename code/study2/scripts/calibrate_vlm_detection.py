"""Choose disease-specific VLM score cutoffs from inference CSV (held-out patients).

Reads ``study2_inference_results.csv``-style output, holds out a fraction of
patients for cutoff selection only, sweeps thresholds, then applies the locked
cutoffs to the remaining patients for downstream lead-time tables.

Proxy labels (late vs early in the pre-diagnosis window) define study-level
positives and negatives for F1 / Youden optimisation when no separate control
cohort is available — see ``study2.core.detection_calibration``.

Example::

    python code/study2/scripts/calibrate_vlm_detection.py \\
        --inference-csv data/MIMIC-CXR/csv/study2_results/study2_inference_results.csv \\
        --output-dir data/MIMIC-CXR/csv/study2_detection_calibration
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional plotting dependency
    plt = None

_CODE_DIR = Path(__file__).resolve().parents[2]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from study_logging import configure_study_logging
from study2.core.detection_calibration import Criterion, run_detection_threshold_calibration

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data"


def _json_safe(obj: object) -> object:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat() if pd.notna(obj) else None
    if isinstance(obj, float) and (np.isnan(obj) or np.isposinf(obj) or np.isneginf(obj)):
        return None
    return obj


def _safe_slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "disease"


def _write_roc_plot(roc_df: pd.DataFrame, auc: float, disease: str, out_path: Path) -> None:
    if plt is None:
        logger.warning("matplotlib unavailable; skipping ROC plot for %s", disease)
        return
    if roc_df.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(roc_df["fpr"], roc_df["tpr"], marker="o", linewidth=1.5, label=f"ROC (AUC={auc:.3f})")
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.0, color="gray", label="Chance")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve - {disease}")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Patient-level holdout, threshold sweep, and locked detection events "
            "from Study 2 inference CSV."
        )
    )
    p.add_argument(
        "--inference-csv",
        type=Path,
        default=DATA_ROOT / "MIMIC-CXR/csv/study2_results/study2_inference_results.csv",
        help="Output of run_inference.py (one row per study).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_ROOT / "MIMIC-CXR/csv/study2_detection_calibration",
        help="Directory for JSON summary, sweep CSVs, and detection-event tables.",
    )
    p.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of unique subject_id values in the calibration holdout.",
    )
    p.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="RNG seed for patient shuffle.",
    )
    p.add_argument(
        "--criterion",
        type=str,
        default="f1_then_youden",
        choices=["f1", "youden", "f1_then_youden"],
        help="How to pick the cutoff from the validation sweep.",
    )
    p.add_argument(
        "--positive-within-hours",
        type=float,
        default=48.0,
        help="Studies with hours_before_diagnosis <= this are proxy-positive (late window).",
    )
    p.add_argument(
        "--negative-at-least-hours",
        type=float,
        default=288.0,
        help="Studies with hours_before_diagnosis >= this are proxy-negative (early window).",
    )
    p.add_argument(
        "--threshold-start",
        type=float,
        default=0.1,
    )
    p.add_argument(
        "--threshold-end",
        type=float,
        default=0.9,
    )
    p.add_argument(
        "--threshold-step",
        type=float,
        default=0.05,
    )
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    p.add_argument(
        "--strict-validation-pool",
        action="store_true",
        help=(
            "Do not fall back to the full disease cohort when the validation patient "
            "fold has no proxy-positive and proxy-negative studies (default: allow fallback)."
        ),
    )
    return p.parse_args()


def run() -> None:
    args = parse_args()
    configure_study_logging(study_name="study2_detection_cal", level=getattr(logging, args.log_level))

    if not args.inference_csv.exists():
        logger.error("Inference CSV not found: %s", args.inference_csv)
        sys.exit(1)

    inf = pd.read_csv(args.inference_csv)
    if inf.empty:
        logger.error("Inference CSV is empty.")
        sys.exit(1)

    thresholds = np.arange(
        args.threshold_start,
        args.threshold_end + args.threshold_step / 2,
        args.threshold_step,
    )
    criterion: Criterion = args.criterion  # type: ignore[assignment]

    out = run_detection_threshold_calibration(
        inf,
        val_fraction=args.val_fraction,
        random_state=args.random_state,
        thresholds=thresholds,
        positive_within_hours=args.positive_within_hours,
        negative_at_least_hours=args.negative_at_least_hours,
        criterion=criterion,
        allow_full_cohort_fallback=not args.strict_validation_pool,
    )

    if not out["thresholds_by_disease"]:
        logger.error("No disease-specific thresholds were chosen — check proxy labels and data.")
        sys.exit(2)

    df = out["inference_table"]
    val_mask = out["val_mask"]
    val_subjects = sorted(df.loc[val_mask, "subject_id"].unique().astype(int).tolist())
    rep_subjects = sorted(df.loc[out["train_mask"], "subject_id"].unique().astype(int).tolist())

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for d, sw in out["sweeps_by_disease"].items():
        path = args.output_dir / f"vlm_threshold_sweep_{d}.csv"
        if isinstance(sw, pd.DataFrame) and not sw.empty:
            sw.to_csv(path, index=False)
            logger.info("Wrote %s", path)
    for d, roc_df in out["roc_by_disease"].items():
        if not isinstance(roc_df, pd.DataFrame) or roc_df.empty:
            continue
        slug = _safe_slug(str(d))
        roc_csv = args.output_dir / f"vlm_roc_curve_{slug}.csv"
        roc_df.to_csv(roc_csv, index=False)
        logger.info("Wrote %s", roc_csv)
        auc = float(out["per_disease"].get(str(d), {}).get("roc_auc", float("nan")))
        if np.isfinite(auc):
            roc_png = args.output_dir / f"vlm_roc_curve_{slug}.png"
            _write_roc_plot(roc_df, auc, str(d), roc_png)
            if roc_png.exists():
                logger.info("Wrote %s", roc_png)

    events_rep = out["events_reporting"]
    events_val = out["events_validation"]
    if not events_rep.empty:
        p_rep = args.output_dir / "vlm_detection_events_reporting_split.csv"
        events_rep.to_csv(p_rep, index=False)
        logger.info("Wrote %s (%d rows)", p_rep, len(events_rep))
    if not events_val.empty:
        p_val = args.output_dir / "vlm_detection_events_calibration_split.csv"
        events_val.to_csv(p_val, index=False)
        logger.info("Wrote %s (%d rows)", p_val, len(events_val))

    summary = {
        "thresholds_by_disease": out["thresholds_by_disease"],
        "per_disease": out["per_disease"],
        "val_fraction": out["val_fraction"],
        "random_state": out["random_state"],
        "criterion": out["criterion"],
        "n_unique_subjects": out["n_unique_subjects"],
        "n_calibration_split_subjects": out["n_validation_subjects"],
        "n_reporting_split_subjects": out["n_reporting_subjects"],
        "calibration_split_subject_ids": val_subjects,
        "reporting_split_subject_ids": rep_subjects,
        "positive_within_hours": out["positive_within_hours"],
        "negative_at_least_hours": out["negative_at_least_hours"],
        "allow_full_cohort_fallback": out["allow_full_cohort_fallback"],
        "threshold_grid": np.asarray(out["thresholds_array"]).tolist(),
        "inference_csv": str(args.inference_csv.resolve()),
    }
    summary = _json_safe(summary)
    json_path = args.output_dir / "vlm_detection_calibration_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote %s", json_path)
    logger.info("Locked thresholds: %s", out["thresholds_by_disease"])


if __name__ == "__main__":
    run()
