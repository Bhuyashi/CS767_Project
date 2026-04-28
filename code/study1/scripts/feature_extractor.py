from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

_CODE_DIR = Path(__file__).resolve().parents[2]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from study_logging import configure_study_logging

from study1.core.data_io import validate_required_paths
from study1.core.pipeline import build_dataset

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Study 1 feature extraction pipeline (data prep, features, and clustering)."
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=DATA_ROOT / "MIMIC-CXR/csv/mimic-cxr-2.0.0-metadata.csv",
    )
    parser.add_argument(
        "--chexpert-csv",
        type=Path,
        default=DATA_ROOT / "MIMIC-CXR/csv/mimic-cxr-2.0.0-chexpert.csv",
    )
    parser.add_argument(
        "--reports-root",
        type=Path,
        default=DATA_ROOT / "MIMIC-CXR/mimic-cxr-reports/files",
    )
    parser.add_argument(
        "--mimic-iv-patients-csv",
        type=Path,
        default=DATA_ROOT / "MIMIC-IV/csv/patients.csv",
    )
    parser.add_argument(
        "--output-csv", type=Path, default=DATA_ROOT / "MIMIC-CXR/csv/study1_features.csv"
    )
    parser.add_argument(
        "--qc-csv", type=Path, default=DATA_ROOT / "MIMIC-CXR/csv/study1_feature_qc.csv"
    )
    parser.add_argument("--k-clusters", type=int, default=30, help="KMeans clusters for proxy radiologist IDs.")
    parser.add_argument("--max-reports", type=int, default=None, help="Optional cap for faster test runs.")
    parser.add_argument("--random-state", type=int, default=42)
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
    required = {
        "metadata_csv": args.metadata_csv,
        "chexpert_csv": args.chexpert_csv,
        "reports_root": args.reports_root,
        "mimic_iv_patients_csv": args.mimic_iv_patients_csv,
    }
    validate_required_paths(required)

    result, qc = build_dataset(
        metadata_csv=args.metadata_csv,
        chexpert_csv=args.chexpert_csv,
        reports_root=args.reports_root,
        mimic_iv_patients_csv=args.mimic_iv_patients_csv,
        k_clusters=args.k_clusters,
        max_reports=args.max_reports,
        random_state=args.random_state,
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.qc_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False)
    pd.DataFrame([qc]).to_csv(args.qc_csv, index=False)

    logger.info("Feature extraction pipeline complete.")
    logger.info("Output rows: %s", qc["n_reports"])
    logger.info("Timestamp granularity detected: %s", qc["time_granularity"])
    logger.info("Circadian mode used: %s", qc["circadian_mode_used"])
    logger.info("Saved: %s", args.output_csv)
    logger.info("Saved: %s", args.qc_csv)

if __name__ == "__main__":
    run()