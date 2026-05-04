"""Study 2 Phase 2: BioViL-T sequence inference in the pre-diagnosis CXR window.

For each cohort admission event, loads frontal studies in chronological order within
``[diagnosis_time - window_days, diagnosis_time]``, scores each study with a disease-specific
``[0, 1]`` mapping (see ``study2.core.model.cosine_similarity_to_unit_interval``), and writes
one row per study.

Temporal encoding uses BioViL-T's **two-frame** prior conditioning (``previous_image``): each
study is encoded with the immediately preceding study in the window as context; the first
study in the window uses single-image mode. This matches the public ``hi-ml-multimodal``
API rather than a single forward pass over an arbitrary-length sequence.

Example::

    python code/study2/scripts/run_inference.py \\
        --metadata-csv data/MIMIC-CXR/csv/mimic-cxr-2.0.0-metadata.csv \\
        --images-root data/MIMIC-CXR-JPG/files \\
        --cohort-csv data/MIMIC-CXR/csv/study2_cohort/study2_index_cohort.csv \\
        --output-dir data/MIMIC-CXR/csv/study2_results
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_CODE_DIR = Path(__file__).resolve().parents[2]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from study_logging import configure_study_logging
from study2.core.data_io import (
    build_cohort_study_sequence_table,
    load_metadata_frontal,
    validate_required_paths,
)
from study2.core.pipeline import run_sequence_inference

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data"


def _resolve_device(arg: str) -> str:
    if arg == "auto":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return arg


def _load_index_cohort(cohort_csv: Path) -> pd.DataFrame:
    cohort = pd.read_csv(
        cohort_csv,
        dtype={
            "subject_id": "int64",
            "hadm_id": "int64",
            "disease_type": "string",
            "window_days": "int64",
        },
    )
    cohort["diagnosis_time"] = pd.to_datetime(cohort["diagnosis_time"], errors="coerce")
    cohort = cohort.dropna(subset=["diagnosis_time"]).copy()
    if "window_days" not in cohort.columns:
        cohort["window_days"] = 14
    cohort["window_start"] = cohort["diagnosis_time"] - pd.to_timedelta(
        cohort["window_days"], unit="D"
    )
    return cohort


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Study 2 Phase 2: BioViL-T inference over each patient's ordered CXR studies "
            "in the pre-diagnosis window (disease-specific scores in [0,1])."
        )
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=DATA_ROOT / "MIMIC-CXR/csv/mimic-cxr-2.0.0-metadata.csv",
        help="MIMIC-CXR metadata CSV.",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=DATA_ROOT / "MIMIC-CXR-JPG/files",
        help="Root of MIMIC-CXR-JPG tree (p<xx>/p<subject_id>/s<study_id>/<dicom_id>.jpg).",
    )
    parser.add_argument(
        "--cohort-csv",
        type=Path,
        default=DATA_ROOT / "MIMIC-CXR/csv/study2_cohort/study2_index_cohort.csv",
        help="Phase 1 cohort CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_ROOT / "MIMIC-CXR/csv/study2_results",
        help="Output directory for CSV, optional sequence manifest, embeddings, QC JSON.",
    )
    parser.add_argument(
        "--disease-filter",
        type=str,
        default="all",
        choices=["all", "heart_failure", "sepsis"],
        help='Process only this disease cohort, or "all" (heart failure events first, then sepsis).',
    )
    parser.add_argument(
        "--min-resolved-studies",
        type=int,
        default=3,
        help="Minimum frontal JPG-backed studies per event (should match Phase 1 window count intent).",
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        default=None,
        help="Cap the number of cohort events (subject_id, hadm_id, disease_type) to process.",
    )
    parser.add_argument(
        "--no-prior",
        action="store_true",
        help="Single-image BioViL-T for every study (ignore in-window prior conditioning).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help='Torch device ("auto" prefers CUDA when available).',
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument(
        "--write-sequence-manifest",
        action="store_true",
        help="Also write study2_sequence_table.csv (pre-inference study list + paths).",
    )
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    configure_study_logging(study_name="study2", level=getattr(logging, args.log_level))

    validate_required_paths(
        {
            "metadata_csv": args.metadata_csv,
            "images_root": args.images_root,
            "cohort_csv": args.cohort_csv,
        }
    )

    cohort = _load_index_cohort(args.cohort_csv)
    if cohort.empty:
        logger.error("Cohort CSV has no valid diagnosis_time rows: %s", args.cohort_csv)
        sys.exit(1)

    if args.disease_filter != "all":
        cohort = cohort[cohort["disease_type"].astype(str) == args.disease_filter].reset_index(
            drop=True
        )
        if cohort.empty:
            logger.error("No cohort rows after --disease-filter=%s", args.disease_filter)
            sys.exit(1)

    logger.info("Loading frontal metadata (single PA-preferred row per study)")
    frontal = load_metadata_frontal(args.metadata_csv)
    frontal = frontal[frontal["subject_id"].isin(cohort["subject_id"].unique())].reset_index(
        drop=True
    )
    if frontal.empty:
        logger.error("No frontal metadata for cohort subject_ids.")
        sys.exit(1)

    seq_df, seq_qc = build_cohort_study_sequence_table(
        cohort,
        frontal,
        args.images_root,
        min_resolved_studies=args.min_resolved_studies,
    )
    if seq_df.empty:
        logger.error(
            "Empty study sequence table — check cohort overlap with metadata and on-disk JPGs. QC=%s",
            seq_qc,
        )
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.write_sequence_manifest:
        manifest_csv = args.output_dir / "study2_sequence_table.csv"
        seq_df.to_csv(manifest_csv, index=False)
        logger.info("Wrote sequence manifest: %s", manifest_csv)

    disease_order: tuple[str, ...]
    if args.disease_filter == "all":
        disease_order = ("heart_failure", "sepsis")
    else:
        disease_order = (args.disease_filter,)

    use_prior = not args.no_prior
    device = _resolve_device(args.device)
    logger.info("Inference device: %s", device)

    results_df, emb_arr, run_qc = run_sequence_inference(
        seq_df,
        device=device,
        max_patients=args.max_patients,
        disease_process_order=disease_order,
        use_prior=use_prior,
    )

    if results_df.empty:
        logger.error("Inference produced no rows.")
        sys.exit(1)

    results_csv = args.output_dir / "study2_inference_results.csv"
    embeddings_npz = args.output_dir / "study2_embeddings.npz"
    qc_json = args.output_dir / "study2_qc.json"

    results_df.to_csv(results_csv, index=False)
    logger.info("Saved results CSV: %s", results_csv)

    np.savez_compressed(
        embeddings_npz,
        study_embeddings=emb_arr,
        subject_ids=results_df["subject_id"].to_numpy(),
        study_ids=results_df["study_id"].to_numpy(),
    )
    logger.info("Saved embeddings: %s", embeddings_npz)

    qc = {
        "sequence_table_qc": seq_qc,
        "inference_qc": run_qc,
        "cohort_csv": str(args.cohort_csv),
        "metadata_csv": str(args.metadata_csv),
        "images_root": str(args.images_root),
        "disease_filter": args.disease_filter,
        "min_resolved_studies": args.min_resolved_studies,
        "max_patients": args.max_patients,
    }
    qc_json.write_text(json.dumps(qc, indent=2), encoding="utf-8")
    logger.info("Saved QC: %s", qc_json)
    logger.info("Study 2 sequence inference complete (%d study rows).", len(results_df))


if __name__ == "__main__":
    run()
