from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parents[2]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from study_logging import configure_study_logging
from study1.core.analysis import FEATURE_COLUMNS, run_stats_results
from study1.core.data_io import validate_required_paths

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Study 1 pipeline (statistics + outputs).")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DATA_ROOT / "MIMIC-CXR/csv/study1_features.csv",
        help="Feature extraction output CSV with language features and clusters.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DATA_ROOT / "MIMIC-CXR/csv/study1_results",
        help="Output directory for tables and figures.",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="Family-wise alpha for Bonferroni threshold.")
    parser.add_argument(
        "--force-three-comparisons",
        action="store_true",
        help="Force Bonferroni denominator to 5 features x 3 comparisons = 15 tests.",
    )
    parser.add_argument(
        "--feature-columns",
        type=str,
        default=None,
        help=(
            "Comma-separated feature column names to analyse. "
            "Defaults to the original 5 lexical features. "
            "Use this when running on LLM-scored metrics: "
            "specificity,abbreviation_usage,hedge_rate,urgency_signaling,actionable_recommendation_rate"
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity level.",
    )
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    configure_study_logging(study_name="study1", level=getattr(logging, args.log_level))
    validate_required_paths({"input_csv": args.input_csv})

    feature_columns = (
        [c.strip() for c in args.feature_columns.split(",") if c.strip()]
        if args.feature_columns
        else FEATURE_COLUMNS
    )

    outputs = run_stats_results(
        input_csv=args.input_csv,
        out_dir=args.out_dir,
        alpha=args.alpha,
        force_three_comparisons=args.force_three_comparisons,
        feature_columns=feature_columns,
    )

    logger.info("Stats modeling and results generation complete.")
    for key, path in outputs.items():
        logger.info("Saved %s: %s", key, path)


if __name__ == "__main__":
    run()
