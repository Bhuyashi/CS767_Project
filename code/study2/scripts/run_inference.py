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
from study2.core.constants import DEFAULT_PROMPT_COLUMNS, DEFAULT_TEXT_PROMPTS
from study2.core.data_io import (
    build_temporal_pairs,
    load_metadata_frontal,
    resolve_pair_image_paths,
    validate_required_paths,
)
from study2.core.pipeline import run_inference

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
        usecols=["subject_id", "hadm_id", "disease_type", "diagnosis_time", "window_days"],
        dtype={
            "subject_id": "int64",
            "hadm_id": "int64",
            "disease_type": "string",
            "window_days": "int64",
        },
    )
    cohort["diagnosis_time"] = pd.to_datetime(cohort["diagnosis_time"], errors="coerce")
    cohort = cohort.dropna(subset=["diagnosis_time"]).copy()
    cohort["window_start"] = cohort["diagnosis_time"] - pd.to_timedelta(cohort["window_days"], unit="D")
    return cohort


def _link_pairs_to_index_cohort(pairs: pd.DataFrame, cohort: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Attach diagnosis metadata from Phase 1 and keep pairs inside the diagnosis window."""
    total_pairs = len(pairs)
    merged = pairs.merge(
        cohort[
            ["subject_id", "hadm_id", "disease_type", "diagnosis_time", "window_start"]
        ],
        on="subject_id",
        how="inner",
    )
    if merged.empty:
        return merged, {
            "n_pairs_before_cohort_link": int(total_pairs),
            "n_pairs_after_subject_join": 0,
            "n_pairs_after_window_filter": 0,
        }

    in_window = merged[
        (merged["current_datetime"] >= merged["window_start"])
        & (merged["current_datetime"] <= merged["diagnosis_time"])
    ].copy()
    if in_window.empty:
        return in_window, {
            "n_pairs_before_cohort_link": int(total_pairs),
            "n_pairs_after_subject_join": int(len(merged)),
            "n_pairs_after_window_filter": 0,
        }

    in_window["hours_before_diagnosis"] = (
        (in_window["diagnosis_time"] - in_window["current_datetime"]).dt.total_seconds() / 3600.0
    )
    # If a pair matches multiple cohort rows for the subject, pick the closest upcoming diagnosis.
    in_window = (
        in_window.sort_values(["subject_id", "current_study_id", "hours_before_diagnosis"])
        .drop_duplicates(subset=["subject_id", "current_study_id"], keep="first")
        .reset_index(drop=True)
    )
    return in_window, {
        "n_pairs_before_cohort_link": int(total_pairs),
        "n_pairs_after_subject_join": int(len(merged)),
        "n_pairs_after_window_filter": int(len(in_window)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Study 2: BioViL-T inference pipeline for MIMIC-CXR temporal image pairs."
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=DATA_ROOT / "MIMIC-CXR/csv/mimic-cxr-2.0.0-metadata.csv",
        help="MIMIC-CXR metadata CSV (dicom_id, subject_id, study_id, ViewPosition, StudyDate, StudyTime).",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=DATA_ROOT / "MIMIC-CXR-JPG/files",
        help="Root of MIMIC-CXR-JPG image tree (p<xx>/p<subject_id>/s<study_id>/<dicom_id>.jpg).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_ROOT / "MIMIC-CXR/csv/study2_results",
        help="Directory for output CSV, embeddings .npz, and QC JSON.",
    )
    parser.add_argument(
        "--cohort-csv",
        type=Path,
        default=DATA_ROOT / "MIMIC-CXR/csv/study2_cohort/study2_index_cohort.csv",
        help="Phase 1 cohort CSV used to link inference to diagnosis windows.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Process at most this many pairs (useful for dry runs).",
    )
    parser.add_argument(
        "--no-prior",
        action="store_true",
        help="Run single-image BioViL-T inference; ignore prior image conditioning.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help='Torch device for inference ("auto" uses CUDA when available).',
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
    configure_study_logging(study_name="study2", level=getattr(logging, args.log_level))

    validate_required_paths(
        {
            "metadata_csv": args.metadata_csv,
            "images_root": args.images_root,
            "cohort_csv": args.cohort_csv,
        }
    )

    metadata = load_metadata_frontal(args.metadata_csv)
    pairs = build_temporal_pairs(metadata)

    if pairs.empty:
        logger.error("No temporal pairs found. Check that the metadata CSV contains subjects with ≥2 studies.")
        sys.exit(1)

    cohort = _load_index_cohort(args.cohort_csv)
    if cohort.empty:
        logger.error("Cohort CSV has no valid diagnosis_time rows: %s", args.cohort_csv)
        sys.exit(1)

    pairs, cohort_link_qc = _link_pairs_to_index_cohort(pairs, cohort)
    if pairs.empty:
        logger.error(
            "No temporal pairs matched the Phase 1 cohort diagnosis windows. "
            "QC: %s",
            cohort_link_qc,
        )
        if cohort_link_qc.get("n_pairs_after_subject_join", 0) == 0:
            logger.error(
                "No overlap between pair subject_ids and cohort CSV subject_ids — "
                "confirm both pipelines use the same MIMIC release and cohort path."
            )
        elif cohort_link_qc.get("n_pairs_after_window_filter", 0) == 0:
            logger.error(
                "Pairs existed for cohort subjects but no (prior, current) pair had "
                "current_datetime inside [window_start, diagnosis_time]. "
                "Check diagnosis_time / window_days in the cohort CSV."
            )
        sys.exit(1)

    pairs, path_qc = resolve_pair_image_paths(pairs, args.images_root)

    if pairs.empty:
        logger.error(
            "No cohort-matched pairs had both JPGs on disk under %s. "
            "Download MIMIC-CXR-JPG for these subjects (see "
            "code/study2/scripts/download_cohort_mimic_jpg.py) or fix --images-root.",
            args.images_root,
        )
        sys.exit(1)

    use_prior = not args.no_prior
    device = _resolve_device(args.device)
    logger.info("Inference device: %s", device)
    results_df, current_embs, prior_embs = run_inference(
        pairs=pairs,
        text_prompts=DEFAULT_TEXT_PROMPTS,
        prompt_columns=DEFAULT_PROMPT_COLUMNS,
        use_prior=use_prior,
        max_pairs=args.max_pairs,
        device=device,
    )

    if results_df.empty:
        logger.error("Inference produced no results. Check model installation and image paths.")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    results_csv = args.output_dir / "study2_inference_results.csv"
    embeddings_npz = args.output_dir / "study2_embeddings.npz"
    qc_json = args.output_dir / "study2_qc.json"

    results_df.to_csv(results_csv, index=False)
    logger.info("Saved results CSV: %s", results_csv)

    np.savez_compressed(
        embeddings_npz,
        current_embeddings=current_embs,
        prior_embeddings=prior_embs,
        subject_ids=results_df["subject_id"].to_numpy(),
        current_study_ids=results_df["current_study_id"].to_numpy(),
        prior_study_ids=results_df["prior_study_id"].to_numpy(),
    )
    logger.info("Saved embeddings: %s", embeddings_npz)

    qc = {
        **path_qc,
        **cohort_link_qc,
        "n_pairs_processed": len(results_df),
        "use_prior_conditioning": use_prior,
        "embedding_dim": int(current_embs.shape[1]) if current_embs.ndim == 2 else None,
        "cohort_csv": str(args.cohort_csv),
        "text_prompts": DEFAULT_TEXT_PROMPTS,
        "prompt_columns": DEFAULT_PROMPT_COLUMNS,
    }
    qc_json.write_text(json.dumps(qc, indent=2))
    logger.info("Saved QC summary: %s", qc_json)

    logger.info("Study 2 inference pipeline complete.")
    logger.info("Pairs processed: %d", len(results_df))
    logger.info("Embedding array shape: %s", current_embs.shape)


if __name__ == "__main__":
    run()
