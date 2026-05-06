"""Study 1 — LLM metric extraction for radiology reports.

Reads report texts from MIMIC-CXR, joins metadata from the existing
feature CSV (circadian_bin, radiologist_cluster, severity), scores each
report with an LLM, and writes a new CSV ready for stats_modelling.py.

Checkpointing: already-scored study_ids in the output CSV are skipped,
so interrupted runs resume safely.

Examples
--------
Gemini (recommended for speed):
    python code/study1/scripts/llm_feature_extractor.py \\
        --backend gemini \\
        --api-key $GEMINI_API_KEY \\
        --features-csv data/MIMIC-CXR/csv/study1_features.csv \\
        --reports-dir data/MIMIC-CXR/mimic-cxr-reports/files \\
        --output-csv data/MIMIC-CXR/csv/study1_llm_features.csv

Llama via Ollama (local, no API key needed):
    ollama pull llama3   # one-time
    python code/study1/scripts/llm_feature_extractor.py \\
        --backend ollama --model llama3 \\
        --features-csv data/MIMIC-CXR/csv/study1_features.csv \\
        --reports-dir data/MIMIC-CXR/mimic-cxr-reports/files \\
        --output-csv data/MIMIC-CXR/csv/study1_llm_features.csv

Llama via Together AI (hosted, fast):
    python code/study1/scripts/llm_feature_extractor.py \\
        --backend openai_compat \\
        --api-key $TOGETHER_API_KEY \\
        --base-url https://api.together.xyz/v1 \\
        --model meta-llama/Llama-3-8b-chat-hf \\
        --features-csv data/MIMIC-CXR/csv/study1_features.csv \\
        --reports-dir data/MIMIC-CXR/mimic-cxr-reports/files \\
        --output-csv data/MIMIC-CXR/csv/study1_llm_features.csv

After this script, run stats analysis:
    python code/study1/scripts/stats_modelling.py \\
        --input-csv data/MIMIC-CXR/csv/study1_llm_features.csv \\
        --feature-columns specificity,abbreviation_usage,hedge_rate,urgency_signaling,actionable_recommendation_rate \\
        --out-dir data/MIMIC-CXR/csv/study1_llm_results
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd

_CODE_DIR = Path(__file__).resolve().parents[2]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from study_logging import configure_study_logging
from study1.core.llm_features import LLM_METRIC_COLUMNS, BackendType, build_scorer

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data"

METADATA_COLUMNS = [
    "study_id",
    "subject_id",
    "circadian_bin",
    "radiologist_cluster",
    "severity",
]


def _find_report(reports_dir: Path, subject_id: int, study_id: int) -> Path | None:
    sid_str = str(subject_id)
    prefix = f"p{sid_str[:2]}"
    path = reports_dir / prefix / f"p{subject_id}" / f"s{study_id}.txt"
    return path if path.exists() else None


def _load_existing(output_csv: Path) -> set[int]:
    if not output_csv.exists():
        return set()
    try:
        df = pd.read_csv(output_csv, usecols=["study_id"])
        return set(df["study_id"].dropna().astype(int).tolist())
    except Exception:
        return set()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Score radiology reports with an LLM and write metrics for Study 1 stats analysis."
    )
    p.add_argument("--backend", type=str, required=True, choices=["gemini", "ollama", "openai_compat", "hf"],
                   help="LLM backend to use. 'hf' runs a local HuggingFace model on GPU (no API key needed).")
    p.add_argument("--api-key", type=str, default=os.environ.get("LLM_API_KEY", ""),
                   help="API key for Gemini or OpenAI-compatible endpoint (or set LLM_API_KEY env var).")
    p.add_argument("--model", type=str, default="",
                   help="Model name (e.g. gemini-1.5-flash, llama3, meta-llama/Llama-3-8b-chat-hf).")
    p.add_argument("--base-url", type=str, default="",
                   help="Base URL for openai_compat or ollama backends.")
    p.add_argument("--features-csv", type=Path,
                   default=DATA_ROOT / "MIMIC-CXR/csv/study1_features.csv",
                   help="Existing Study 1 feature CSV (provides metadata columns).")
    p.add_argument("--reports-dir", type=Path,
                   default=DATA_ROOT / "MIMIC-CXR/mimic-cxr-reports/files",
                   help="Root of MIMIC-CXR report directory (p<xx>/p<subject_id>/s<study_id>.txt).")
    p.add_argument("--output-csv", type=Path,
                   default=DATA_ROOT / "MIMIC-CXR/csv/study1_llm_features.csv",
                   help="Output CSV path.")
    p.add_argument("--max-reports", type=int, default=None,
                   help="Cap number of reports to score (for testing).")
    p.add_argument("--rate-limit-delay", type=float, default=0.5,
                   help="Seconds to sleep between API calls (default 0.5). Increase for strict rate limits.")
    p.add_argument("--retries", type=int, default=3,
                   help="Retries per report on API/parse failure.")
    p.add_argument("--log-level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    return p.parse_args()


def run() -> None:
    args = parse_args()
    configure_study_logging(study_name="study1_llm", level=getattr(logging, args.log_level))

    if not args.features_csv.exists():
        logger.error("Features CSV not found: %s", args.features_csv)
        sys.exit(1)
    if not args.reports_dir.exists():
        logger.error("Reports directory not found: %s", args.reports_dir)
        sys.exit(1)

    features = pd.read_csv(args.features_csv, usecols=METADATA_COLUMNS + ["study_id"])
    features["study_id"] = features["study_id"].astype(int)
    features["subject_id"] = features["subject_id"].astype(int)
    logger.info("Loaded %d rows from features CSV", len(features))

    already_done = _load_existing(args.output_csv)
    if already_done:
        logger.info("Resuming: %d study_ids already scored, skipping.", len(already_done))
    pending = features[~features["study_id"].isin(already_done)].reset_index(drop=True)

    if args.max_reports is not None:
        pending = pending.head(args.max_reports)
    logger.info("Reports to score: %d", len(pending))

    scorer = build_scorer(
        args.backend,  # type: ignore[arg-type]
        api_key=args.api_key,
        model=args.model,
        base_url=args.base_url,
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.output_csv.exists() or args.output_csv.stat().st_size == 0

    n_ok = n_missing_report = n_failed = 0

    with args.output_csv.open("a", encoding="utf-8", newline="") as fout:
        import csv
        all_cols = METADATA_COLUMNS + LLM_METRIC_COLUMNS
        writer = csv.DictWriter(fout, fieldnames=all_cols)
        if write_header:
            writer.writeheader()

        for i, row in pending.iterrows():
            study_id = int(row["study_id"])
            subject_id = int(row["subject_id"])

            report_path = _find_report(args.reports_dir, subject_id, study_id)
            if report_path is None:
                logger.debug("Report not found: subject=%s study=%s", subject_id, study_id)
                n_missing_report += 1
                continue

            report_text = report_path.read_text(encoding="utf-8", errors="replace").strip()
            if not report_text:
                n_missing_report += 1
                continue

            scores = scorer.score(report_text, retries=args.retries)
            if scores is None:
                logger.warning("Scoring failed for study_id=%s", study_id)
                n_failed += 1
                continue

            out_row = {col: row[col] for col in METADATA_COLUMNS}
            out_row.update(scores)
            writer.writerow(out_row)
            fout.flush()
            n_ok += 1

            if n_ok % 50 == 0:
                logger.info("Progress: scored=%d missing=%d failed=%d / %d total",
                            n_ok, n_missing_report, n_failed, len(pending))

            if args.rate_limit_delay > 0:
                time.sleep(args.rate_limit_delay)

    logger.info(
        "Done: scored=%d missing_report=%d failed=%d — output: %s",
        n_ok, n_missing_report, n_failed, args.output_csv,
    )


if __name__ == "__main__":
    run()
