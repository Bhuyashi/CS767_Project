from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .clustering import add_radiologist_proxy_cluster
from .constants import DEFAULT_HEDGE_PHRASES
from .data_io import (
    load_chexpert_severity,
    load_metadata_study_times,
    load_mimic_iv_patient_covariates,
    read_reports,
)
from .features import add_language_features
from .text_processing import assign_circadian_bin, infer_time_granularity

logger = logging.getLogger(__name__)


def build_dataset(
    metadata_csv: Path,
    chexpert_csv: Path,
    reports_root: Path,
    mimic_iv_patients_csv: Path,
    k_clusters: int,
    max_reports: int | None,
    random_state: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    logger.info("Loading study metadata from %s", metadata_csv)
    study_times = load_metadata_study_times(metadata_csv)
    logger.info("Loaded metadata rows: %d", len(study_times))
    granularity = infer_time_granularity(study_times["study_datetime"])
    circadian_mode = "binary_day_night" if granularity == "coarse_3hour_bins" else "four_bins"
    study_times["circadian_bin"] = study_times["study_datetime"].map(
        lambda dt: assign_circadian_bin(dt, circadian_mode)
    )

    logger.info("Reading reports from %s", reports_root)
    reports = read_reports(reports_root, max_reports=max_reports)
    logger.info("Loaded report rows: %d", len(reports))
    logger.info("Loading CheXpert labels from %s", chexpert_csv)
    chexpert = load_chexpert_severity(chexpert_csv)
    logger.info("Loaded CheXpert rows: %d", len(chexpert))
    logger.info("Loading MIMIC-IV patient covariates from %s", mimic_iv_patients_csv)
    mimic_iv_patients = load_mimic_iv_patient_covariates(mimic_iv_patients_csv)
    logger.info("Loaded MIMIC-IV patient rows: %d", len(mimic_iv_patients))

    logger.info("Merging report, metadata, CheXpert, and patient tables")
    merged = reports.merge(study_times, on=["study_id", "subject_id"], how="left")
    merged = merged.merge(chexpert, on=["study_id", "subject_id"], how="left")
    merged = merged.merge(mimic_iv_patients, on="subject_id", how="left")

    logger.info("Computing language and proxy-radiologist features")
    merged = add_language_features(merged, DEFAULT_HEDGE_PHRASES)
    merged = add_radiologist_proxy_cluster(merged, k_clusters=k_clusters, random_state=random_state)

    selected_columns = [
        "study_id",
        "subject_id",
        "study_datetime",
        "circadian_bin",
        "report_text",
        "word_count",
        "hedge_rate",
        "mean_sent_length",
        "ttr",
        "certainty_score",
        "severity",
        "radiologist_cluster",
        "gender",
        "anchor_age",
    ]
    merged = merged[selected_columns]
    logger.info("Final dataset rows: %d", len(merged))

    qc = {
        "n_reports": int(len(merged)),
        "n_reports_with_timestamp": int(merged["study_datetime"].notna().sum()),
        "n_reports_with_severity": int(merged["severity"].notna().sum()),
        "time_granularity": granularity,
        "circadian_mode_used": circadian_mode,
        "circadian_counts": merged["circadian_bin"].value_counts(dropna=False).to_dict(),
    }
    return merged, qc
