from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from .constants import CHEXPERT_LABELS
from .text_processing import parse_report_sections

logger = logging.getLogger(__name__)


def validate_required_paths(paths: dict[str, Path]) -> None:
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        details = "\n".join(f"- {name}: {paths[name]}" for name in missing)
        raise FileNotFoundError(f"Missing required files/directories:\n{details}")


def read_reports(report_root: Path, max_reports: int | None = None) -> pd.DataFrame:
    if not report_root.exists():
        raise FileNotFoundError(f"Reports root does not exist: {report_root}")
    if not report_root.is_dir():
        raise NotADirectoryError(f"Reports root is not a directory: {report_root}")

    logger.info("Scanning report files under: %s", report_root)
    if max_reports is not None and max_reports > 0:
        logger.info("Report read limit enabled: max_reports=%d", max_reports)

    records: list[dict[str, object]] = []
    scanned = 0
    skipped_name = 0
    skipped_subject = 0
    for path in report_root.rglob("s*.txt"):
        scanned += 1
        if scanned % 5000 == 0:
            logger.info(
                "Report scan progress: scanned=%d accepted=%d skipped_name=%d skipped_subject=%d",
                scanned,
                len(records),
                skipped_name,
                skipped_subject,
            )

        study_match = re.fullmatch(r"s(\d+)\.txt", path.name)
        if not study_match:
            skipped_name += 1
            continue
        subject_match = re.fullmatch(r"p(\d+)", path.parent.name)
        if not subject_match:
            skipped_subject += 1
            continue

        study_id = int(study_match.group(1))
        subject_id = int(subject_match.group(1))
        text = path.read_text(encoding="utf-8", errors="ignore")

        records.append(
            {
                "study_id": study_id,
                "subject_id": subject_id,
                "report_text": parse_report_sections(text),
                "report_path": str(path),
            }
        )
        if max_reports is not None and max_reports > 0 and len(records) >= max_reports:
            logger.info("Reached max_reports=%d; stopping early.", max_reports)
            break

    if not records:
        raise RuntimeError(
            "No report records were read. Check whether report files follow expected pattern "
            "(e.g., p<subject_id>/s<study_id>.txt) and whether --reports-root points to the correct directory."
        )

    logger.info(
        "Finished reading reports: scanned=%d accepted=%d skipped_name=%d skipped_subject=%d",
        scanned,
        len(records),
        skipped_name,
        skipped_subject,
    )
    return pd.DataFrame.from_records(records)


def load_metadata_study_times(metadata_csv: Path) -> pd.DataFrame:
    metadata = pd.read_csv(
        metadata_csv,
        usecols=["subject_id", "study_id", "StudyDate", "StudyTime"],
        dtype={"subject_id": "int64", "study_id": "int64", "StudyDate": "string", "StudyTime": "string"},
    )
    metadata["StudyDate"] = metadata["StudyDate"].fillna("").str.replace(r"\.0$", "", regex=True)
    metadata["StudyTime"] = metadata["StudyTime"].fillna("").str.replace(r"\.0+$", "", regex=True)
    metadata["StudyTime"] = metadata["StudyTime"].str.extract(r"^(\d{1,6})", expand=False).fillna("")
    metadata["StudyTime"] = metadata["StudyTime"].str.zfill(6)
    metadata["study_datetime"] = pd.to_datetime(
        metadata["StudyDate"] + metadata["StudyTime"],
        format="%Y%m%d%H%M%S",
        errors="coerce",
    )
    return (
        metadata.sort_values("study_datetime")
        .groupby(["study_id", "subject_id"], as_index=False)["study_datetime"]
        .first()
    )


def load_chexpert_severity(chexpert_csv: Path) -> pd.DataFrame:
    chexpert = pd.read_csv(chexpert_csv)
    existing_labels = [col for col in CHEXPERT_LABELS if col in chexpert.columns]
    if len(existing_labels) != len(CHEXPERT_LABELS):
        missing = sorted(set(CHEXPERT_LABELS) - set(existing_labels))
        raise ValueError(f"Missing CheXpert label columns: {missing}")

    labels_numeric = chexpert[existing_labels].apply(pd.to_numeric, errors="coerce")
    chexpert["severity"] = (labels_numeric == 1.0).sum(axis=1).astype("int16")
    return chexpert[["study_id", "subject_id", "severity"]]


def load_mimic_iv_patient_covariates(mimic_iv_patients_csv: Path) -> pd.DataFrame:
    return pd.read_csv(mimic_iv_patients_csv, usecols=["subject_id", "gender", "anchor_age"])
