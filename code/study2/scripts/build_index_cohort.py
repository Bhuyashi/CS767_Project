"""Study 2: sepsis vs heart-failure index cohort from MIMIC-IV discharge CSV + CXR window.

Example::

    python code/study2/scripts/build_index_cohort.py \\
        --discharge-csv data/MIMIC-IV/csv/discharge.csv \\
        --admissions-csv data/MIMIC-IV/csv/admissions.csv \\
        --cxr-metadata-csv data/MIMIC-CXR/csv/mimic-cxr-2.0.0-metadata.csv \\
        --output-dir data/MIMIC-CXR/csv/study2_cohort
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parents[2]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from study_logging import configure_study_logging
from study2.core.cohort import (
    build_index_cohort,
    serialise_qc,
)
from study2.core.data_io import validate_required_paths

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Study 2: index cohort from discharge-note mentions (NegEx) "
            "and ≥3 CXRs in the pre-diagnosis window."
        )
    )
    parser.add_argument(
        "--discharge-csv",
        type=Path,
        default=DATA_ROOT / "MIMIC-IV" / "csv" / "discharge.csv",
        help="MIMIC-IV csv/discharge.csv (subject_id, hadm_id, charttime, text).",
    )
    parser.add_argument(
        "--admissions-csv",
        type=Path,
        default=DATA_ROOT / "MIMIC-IV" / "csv" / "admissions.csv",
        help="MIMIC-IV csv/admissions.csv",
    )
    parser.add_argument(
        "--cxr-metadata-csv",
        type=Path,
        default=DATA_ROOT / "MIMIC-CXR" / "csv" / "mimic-cxr-2.0.0-metadata.csv",
        help="MIMIC-CXR mimic-cxr-2.0.0-metadata.csv",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=14,
        help="Pre-diagnosis window [diagnosis_time - window, diagnosis_time] for CXR counts.",
    )
    parser.add_argument(
        "--min-cxr-studies",
        type=int,
        default=3,
        help="Minimum distinct CXR study_ids required in the window.",
    )
    parser.add_argument(
        "--discharge-chunksize",
        type=int,
        default=50_000,
        help="Rows per chunk when reading discharge.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_ROOT / "MIMIC-CXR" / "csv" / "study2_cohort",
        help="Directory for cohort CSV(s), QC JSON, and summary.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="study2_index_cohort",
        help="Filename prefix for outputs.",
    )
    parser.add_argument(
        "--split-by-disease",
        action="store_true",
        help="Also write separate CSVs for sepsis and heart_failure.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    configure_study_logging(study_name="study2_index", level=getattr(logging, args.log_level))

    validate_required_paths(
        {
            "discharge": args.discharge_csv,
            "admissions": args.admissions_csv,
            "cxr_metadata": args.cxr_metadata_csv,
        }
    )

    cohort_df, qc = build_index_cohort(
        args.discharge_csv,
        args.admissions_csv,
        args.cxr_metadata_csv,
        window_days=args.window_days,
        min_cxr_studies=args.min_cxr_studies,
        discharge_chunksize=args.discharge_chunksize,
    )

    summary = qc.get("cohort_summary") or {}
    if isinstance(summary, dict):
        print(f"  cohort total rows:                {summary.get('total_rows', len(cohort_df))}")
        print(f"  cohort unique patients:           {summary.get('total_unique_patients', '—')}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    main_csv = args.output_dir / f"{args.prefix}.csv"
    qc_json = args.output_dir / f"{args.prefix}_qc.json"

    cohort_df.to_csv(main_csv, index=False)
    logger.info("Wrote cohort: %s (%d rows)", main_csv, len(cohort_df))

    qc_serial = serialise_qc(qc)
    qc_json.write_text(json.dumps(qc_serial, indent=2), encoding="utf-8")
    logger.info("Wrote QC: %s", qc_json)

    summary_path = args.output_dir / f"{args.prefix}_summary.json"
    summary_path.write_text(json.dumps(qc_serial["cohort_summary"], indent=2), encoding="utf-8")
    logger.info("Wrote summary: %s", summary_path)

    if args.split_by_disease and not cohort_df.empty:
        for disease, short in (("sepsis", "sepsis"), ("heart_failure", "heart_failure")):
            sub = cohort_df[cohort_df["disease_type"] == disease]
            out = args.output_dir / f"{args.prefix}_{short}.csv"
            sub.to_csv(out, index=False)
            logger.info("Wrote %s: %d rows", out.name, len(sub))

    if cohort_df.empty:
        print("[build_index_cohort] ERROR: cohort is empty — check discharge mentions and filters.")
        logger.error("Cohort is empty — check discharge mentions and filters.")
        sys.exit(2)

    print("[build_index_cohort] Building index cohort complete.")
    logger.info("Building index cohort complete.")


if __name__ == "__main__":
    run()
