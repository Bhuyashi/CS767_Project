"""Lead-time outcome analysis: survival-style timing, Cox model, KM, and AUC vs time.

Reads the reporting-split detection events table and full inference CSV, optionally
joins MIMIC-IV SOFA, then writes survival/cohort tables, model summaries, and figures.

Example::

    python code/study2/scripts/analyze_lead_time_outcomes.py \\
        --events-csv data/MIMIC-CXR/csv/study2_detection_calibration/vlm_detection_events_reporting_split.csv \\
        --inference-csv data/MIMIC-CXR/csv/study2_results/study2_inference_results.csv \\
        --output-dir data/MIMIC-CXR/csv/study2_lead_time_outcomes \\
        --sofa-csv data/MIMIC-IV/derived/sofa.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parents[2]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from study_logging import configure_study_logging
from study2.core.detection_calibration import (
    DEFAULT_NEGATIVE_AT_LEAST_HOURS,
    DEFAULT_POSITIVE_WITHIN_HOURS,
)
from study2.core.lead_time_outcomes import (
    DEFAULT_ANCHOR_HOURS_BEFORE_DX,
    run_lead_time_outcome_analysis,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data"

_DEFAULT_ANCHOR_HOURS_STR = ",".join(str(int(h)) if float(h).is_integer() else str(h) for h in DEFAULT_ANCHOR_HOURS_BEFORE_DX)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Kaplan–Meier, log-rank, Cox PH, and time-anchored AUC from Study 2 "
            "inference and locked detection events."
        )
    )
    p.add_argument(
        "--events-csv",
        type=Path,
        default=DATA_ROOT / "MIMIC-CXR/csv/study2_detection_calibration/vlm_detection_events_reporting_split.csv",
        help="Reporting-split detection events (one row per cohort event).",
    )
    p.add_argument(
        "--inference-csv",
        type=Path,
        default=DATA_ROOT / "MIMIC-CXR/csv/study2_results/study2_inference_results.csv",
        help="Full per-study inference table (all patients, for imaging frequency).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_ROOT / "MIMIC-CXR/csv/study2_lead_time_outcomes",
        help="Directory for CSV/JSON outputs and figures/.",
    )
    p.add_argument(
        "--sofa-csv",
        type=Path,
        default=DATA_ROOT / "MIMIC-IV/derived/sofa.csv",
        help="MIMIC-IV derived SOFA table (optional; Cox omits if missing).",
    )
    p.add_argument(
        "--no-sofa",
        action="store_true",
        help="Do not load SOFA (pass None even if default path exists).",
    )
    p.add_argument(
        "--window-days",
        type=float,
        default=14.0,
        help="Pre-diagnosis window length (for imaging_frequency = n_studies / window_days).",
    )
    p.add_argument(
        "--anchor-hours",
        type=str,
        default=_DEFAULT_ANCHOR_HOURS_STR,
        help=(
            "Comma-separated hours-before-diagnosis anchors for AUC. "
            "Default spans the 14-day window; narrow 3–24 h grids often tie to one study per patient."
        ),
    )
    p.add_argument(
        "--cox-penalizer",
        type=float,
        default=0.1,
        help="L2 penalizer for CoxPHFitter (helps with small N or collinearity).",
    )
    p.add_argument(
        "--auc-proxy-positive-within-hours",
        type=float,
        default=DEFAULT_POSITIVE_WITHIN_HOURS,
        help="Late-window proxy cutoff (hours before diagnosis); must match calibration if comparable.",
    )
    p.add_argument(
        "--auc-proxy-negative-at-least-hours",
        type=float,
        default=DEFAULT_NEGATIVE_AT_LEAST_HOURS,
        help="Early-window proxy cutoff (hours before diagnosis).",
    )
    p.add_argument(
        "--skip-cox",
        action="store_true",
        help="Skip Cox proportional hazards fit.",
    )
    p.add_argument(
        "--skip-ph-check",
        action="store_true",
        help="Skip Schoenfeld / proportional hazards checks (faster).",
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
    configure_study_logging(study_name="study2_lead_time_outcomes", level=getattr(logging, args.log_level))

    if not args.events_csv.exists():
        logger.error("Events CSV not found: %s", args.events_csv)
        sys.exit(1)
    if not args.inference_csv.exists():
        logger.error("Inference CSV not found: %s", args.inference_csv)
        sys.exit(1)

    try:
        anchors = [float(x.strip()) for x in args.anchor_hours.split(",") if x.strip()]
    except ValueError:
        logger.error("Invalid --anchor-hours: %r", args.anchor_hours)
        sys.exit(2)

    sofa_path = None if args.no_sofa else args.sofa_csv

    out = run_lead_time_outcome_analysis(
        events_csv=args.events_csv,
        inference_csv=args.inference_csv,
        output_dir=args.output_dir,
        sofa_csv=sofa_path,
        window_days=args.window_days,
        anchor_hours=anchors,
        penalizer=args.cox_penalizer,
        skip_cox=args.skip_cox,
        skip_ph_check=args.skip_ph_check,
        auc_proxy_positive_within_hours=args.auc_proxy_positive_within_hours,
        auc_proxy_negative_at_least_hours=args.auc_proxy_negative_at_least_hours,
    )
    logger.info("Wrote survival outcomes: %s", out["survival_outcomes_path"])
    logger.info("KM figure: %s", out["km_plot"])
    logger.info("AUC figure: %s", out["auc_plot"])


if __name__ == "__main__":
    run()
