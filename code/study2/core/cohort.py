"""Study 2 Phase 1: index cohort from discharge summaries + CXR window filter.

Cohort definition: NegEx-style mentions of sepsis or heart failure in MIMIC-IV
``csv/discharge.csv``, merged with admissions for LOS QC, then ≥N CXR studies in the
pre-diagnosis window.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

DiseaseType = Literal["sepsis", "heart_failure"]

WINDOW_DAYS = 14
MIN_CXR_STUDIES = 3


def load_admissions(path: Path) -> pd.DataFrame:
    """Load MIMIC-IV ``csv/admissions.csv`` (expects hadm_id, subject_id, admittime, dischtime)."""
    logger.info("Loading admissions from %s", path)
    df = pd.read_csv(
        path,
        usecols=["subject_id", "hadm_id", "admittime", "dischtime"],
        dtype={"subject_id": "int64", "hadm_id": "int64"},
        parse_dates=["admittime", "dischtime"],
    )
    n_dup = df["hadm_id"].duplicated().sum()
    if n_dup:
        logger.warning("Dropping %d duplicate hadm_id rows in admissions", int(n_dup))
        df = df.drop_duplicates("hadm_id", keep="first")
    logger.info("Admission rows: %d", len(df))
    return df


def load_discharge_table(path: Path, *, chunksize: int = 50_000) -> pd.DataFrame:
    """Load full MIMIC-IV discharge summaries from ``csv/discharge.csv`` (chunked).

    Required columns: ``subject_id``, ``hadm_id``, ``charttime``, ``text``.
    """
    logger.info("Loading discharge summaries from %s", path)
    usecols = ["subject_id", "hadm_id", "charttime", "text"]
    chunks: list[pd.DataFrame] = []
    reader = pd.read_csv(
        path,
        usecols=usecols,
        dtype={"subject_id": "int64", "hadm_id": "int64"},
        chunksize=chunksize,
    )
    with tqdm(desc="discharge.csv", unit="row", unit_scale=True) as pbar:
        for chunk in reader:
            n_read = len(chunk)
            chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
            chunk = chunk.dropna(subset=["charttime"])
            chunks.append(chunk)
            pbar.update(n_read)

    if not chunks:
        logger.warning("Discharge load produced no rows.")
        return pd.DataFrame(columns=usecols)

    out = pd.concat(chunks, ignore_index=True)
    out = out.sort_values(["subject_id", "hadm_id", "charttime"]).reset_index(drop=True)
    logger.info("Discharge rows loaded: %d", len(out))
    return out


def merge_admissions(events: pd.DataFrame, admissions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Attach admittime / dischtime; drop events with no matching admission.

    ``events`` must include ``subject_id``, ``hadm_id``, and ``diagnosis_time``.
    """
    merged = events.merge(
        admissions,
        on=["subject_id", "hadm_id"],
        how="left",
        validate="many_to_one",
    )
    n_miss = merged["admittime"].isna().sum()
    if n_miss:
        logger.warning("Events without matching admission row (dropped): %d", int(n_miss))
    merged = merged.dropna(subset=["admittime", "dischtime"]).reset_index(drop=True)

    stay_hours = (merged["dischtime"] - merged["admittime"]).dt.total_seconds() / 3600.0
    qc = {
        "median_los_hours": float(np.nanmedian(stay_hours)),
        "p25_los_hours": float(np.nanpercentile(stay_hours, 25)),
        "p75_los_hours": float(np.nanpercentile(stay_hours, 75)),
        "n_events_with_admission": float(len(merged)),
    }
    logger.info(
        "Admission merge: median LOS hours=%.1f (events=%d)",
        qc["median_los_hours"],
        len(merged),
    )
    return merged, qc


# --- NegEx-style discharge text -------------------------------------------

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def _split_sentences(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    parts = _SENT_SPLIT.split(text.strip())
    out = [p.strip() for p in parts if p and p.strip()]
    return out if out else [text.strip()]


def _sepsis_negated(sentence: str) -> bool:
    s = sentence.lower()
    patterns = (
        r"\bno\s+sepsis\b",
        r"\bno\s+evidence\s+of\s+sepsis\b",
        r"\bwithout\s+sepsis\b",
        r"\bwithout\s+evidence\s+of\s+sepsis\b",
        r"\bdenies\s+sepsis\b",
        r"\bruled\s+out\s+sepsis\b",
        r"\bno\s+clinical\s+evidence\s+of\s+sepsis\b",
    )
    return any(re.search(p, s) for p in patterns)


def _sepsis_sentence_positive(sentence: str) -> bool:
    if not re.search(r"\bsepsis\b", sentence, re.IGNORECASE):
        return False
    return not _sepsis_negated(sentence)


def _hf_negated(sentence: str) -> bool:
    s = sentence.lower()
    patterns = (
        r"\bno\s+heart\s+failure\b",
        r"\bno\s+evidence\s+of\s+heart\s+failure\b",
        r"\bwithout\s+heart\s+failure\b",
        r"\bwithout\s+evidence\s+of\s+heart\s+failure\b",
        r"\bdenies\s+heart\s+failure\b",
        r"\bruled\s+out\s+heart\s+failure\b",
        r"\bno\s+acute\s+decompensated\s+heart\s+failure\b",
        r"\bno\s+evidence\s+of\s+acute\s+decompensated\s+heart\s+failure\b",
        r"\bruled\s+out\s+acute\s+decompensated\s+heart\s+failure\b",
        r"\badhf\b\s+(?:is\s+)?ruled\s+out\b",
    )
    return any(re.search(p, s) for p in patterns)


def _hf_sentence_positive(sentence: str) -> bool:
    if re.search(r"\bacute\s+decompensated\s+heart\s+failure\b", sentence, re.IGNORECASE):
        return not _hf_negated(sentence)
    if re.search(r"\badhf\b", sentence, re.IGNORECASE):
        return not _hf_negated(sentence)
    if re.search(r"\bheart\s+failure\b", sentence, re.IGNORECASE):
        return not _hf_negated(sentence)
    return False


def _sentence_positive(sentence: str, disease: DiseaseType) -> bool:
    if disease == "sepsis":
        return _sepsis_sentence_positive(sentence)
    return _hf_sentence_positive(sentence)


def first_mention_charttime(
    notes_sorted: pd.DataFrame,
    disease: DiseaseType,
) -> pd.Timestamp | None:
    """First discharge row (by charttime) whose text has a positive non-negated mention."""
    for _, row in notes_sorted.iterrows():
        raw = row.get("text", "")
        text = "" if pd.isna(raw) else str(raw)
        for sent in _split_sentences(text):
            if _sentence_positive(sent, disease):
                return pd.Timestamp(row["charttime"])
    return None


def build_diagnosis_events_from_discharge(discharge: pd.DataFrame) -> pd.DataFrame:
    """One row per (subject_id, hadm_id, disease_type) when that disease is mentioned (NegEx)."""
    rows: list[dict[str, object]] = []
    grouped = discharge.groupby(["subject_id", "hadm_id"], sort=False)
    logger.info("Scanning %d admission discharge groups for target mentions", grouped.ngroups)

    for (sid, hid), grp in tqdm(
        grouped,
        total=grouped.ngroups,
        desc="NegEx (discharge)",
        unit="adm",
    ):
        grp = grp.sort_values("charttime")
        for disease in ("sepsis", "heart_failure"):
            t = first_mention_charttime(grp, disease)
            if t is not None:
                rows.append(
                    {
                        "subject_id": int(sid),
                        "hadm_id": int(hid),
                        "disease_type": disease,
                        "diagnosis_time": t,
                        "diagnosis_time_source": "discharge_note_first_mention",
                    }
                )

    ev = pd.DataFrame(rows)
    logger.info("Diagnosis events from discharge text (pre-admissions merge): %d", len(ev))
    return ev


# --- MIMIC-CXR study datetimes (all views; one row per study) ---------------------


def load_cxr_study_datetimes(metadata_csv: Path) -> pd.DataFrame:
    """Parse study-level datetimes from MIMIC-CXR metadata (same rules as ``data_io`` helpers)."""
    logger.info("Loading CXR metadata for study datetimes: %s", metadata_csv)
    meta = pd.read_csv(
        metadata_csv,
        usecols=["subject_id", "study_id", "StudyDate", "StudyTime"],
        dtype={"subject_id": "int64", "study_id": "int64", "StudyDate": "string", "StudyTime": "string"},
    )
    meta["StudyDate"] = meta["StudyDate"].fillna("").str.replace(r"\.0$", "", regex=True)
    meta["StudyTime"] = meta["StudyTime"].fillna("").str.replace(r"\.0+$", "", regex=True)
    meta["StudyTime"] = meta["StudyTime"].str.extract(r"^(\d{1,6})", expand=False).fillna("")
    meta["StudyTime"] = meta["StudyTime"].str.zfill(6)
    meta["study_datetime"] = pd.to_datetime(
        meta["StudyDate"] + meta["StudyTime"],
        format="%Y%m%d%H%M%S",
        errors="coerce",
    )
    meta = meta.dropna(subset=["study_datetime"])
    studies = (
        meta.groupby(["subject_id", "study_id"], as_index=False)["study_datetime"]
        .min()
        .sort_values(["subject_id", "study_datetime"])
    )
    logger.info("CXR studies with valid timestamps: %d", len(studies))
    return studies


def count_studies_in_pre_diagnosis_window(
    cohort: pd.DataFrame,
    cxr_studies: pd.DataFrame,
    *,
    window_days: int = WINDOW_DAYS,
) -> pd.DataFrame:
    """Add n_cxr_studies_in_window."""
    cohort = cohort.copy()
    delta = pd.Timedelta(days=window_days)
    counts: list[int] = []

    for _, row in cohort.iterrows():
        sid = int(row["subject_id"])
        t_end = pd.Timestamp(row["diagnosis_time"])
        t_start = t_end - delta
        sub = cxr_studies[
            (cxr_studies["subject_id"] == sid)
            & (cxr_studies["study_datetime"] >= t_start)
            & (cxr_studies["study_datetime"] <= t_end)
        ]
        counts.append(int(sub["study_id"].nunique()))

    cohort["n_cxr_studies_in_window"] = counts
    cohort["window_days"] = window_days
    return cohort


def filter_min_cxr_studies(cohort: pd.DataFrame, minimum: int = MIN_CXR_STUDIES) -> pd.DataFrame:
    before = len(cohort)
    out = cohort[cohort["n_cxr_studies_in_window"] >= minimum].reset_index(drop=True)
    logger.info("CXR filter (>=%d studies in %dd window): %d -> %d", minimum, WINDOW_DAYS, before, len(out))
    return out


def cohort_summary_table(cohort: pd.DataFrame) -> dict[str, object]:
    """Aggregates for Table 1 style reporting."""
    rows = []
    for disease, grp in cohort.groupby("disease_type"):
        n_pt = grp["subject_id"].nunique()
        n_rows = len(grp)
        studies_per = grp["n_cxr_studies_in_window"]
        rows.append(
            {
                "disease_type": disease,
                "n_patients": int(n_pt),
                "n_admission_events": int(n_rows),
                "median_cxr_studies_in_window": float(np.median(studies_per)) if len(studies_per) else None,
                "p25_cxr_studies_in_window": float(np.percentile(studies_per, 25)) if len(studies_per) else None,
                "p75_cxr_studies_in_window": float(np.percentile(studies_per, 75)) if len(studies_per) else None,
            }
        )
    return {
        "by_disease": rows,
        "total_unique_patients": int(cohort["subject_id"].nunique()),
        "total_rows": int(len(cohort)),
    }


def build_index_cohort(
    discharge_csv: Path,
    admissions_csv: Path,
    cxr_metadata_csv: Path,
    *,
    window_days: int = WINDOW_DAYS,
    min_cxr_studies: int = MIN_CXR_STUDIES,
    discharge_chunksize: int = 50_000,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Phase 1 pipeline: discharge-based events → admissions → CXR window filter."""
    discharge_df = load_discharge_table(discharge_csv, chunksize=discharge_chunksize)
    qc_all: dict[str, object] = {"n_discharge_rows_loaded": len(discharge_df)}

    events = build_diagnosis_events_from_discharge(discharge_df)
    qc_all["n_diagnosis_events_before_admissions_merge"] = len(events)

    if events.empty:
        logger.error("No sepsis/HF mentions found in discharge text — cannot build cohort.")
        qc_all["cohort_summary"] = {
            "by_disease": [],
            "total_unique_patients": 0,
            "total_rows": 0,
        }
        qc_all["diagnosis_time_source_counts"] = {}
        qc_all["admission_timestamp_qc"] = {}
        qc_all["cxr_filter"] = {
            "window_days": window_days,
            "min_studies": min_cxr_studies,
            "n_before_filter": 0,
            "n_after_filter": 0,
        }
        return pd.DataFrame(), qc_all

    admissions = load_admissions(admissions_csv)
    merged, admission_qc = merge_admissions(events, admissions)
    qc_all["admission_timestamp_qc"] = admission_qc

    cxr_studies = load_cxr_study_datetimes(cxr_metadata_csv)
    cohort = count_studies_in_pre_diagnosis_window(merged, cxr_studies, window_days=window_days)
    n_before_cxr = len(cohort)
    cohort = filter_min_cxr_studies(cohort, minimum=min_cxr_studies)
    qc_all["cxr_filter"] = {
        "window_days": window_days,
        "min_studies": min_cxr_studies,
        "n_before_filter": n_before_cxr,
        "n_after_filter": len(cohort),
    }
    qc_all["cohort_summary"] = cohort_summary_table(cohort)
    qc_all["diagnosis_time_source_counts"] = (
        cohort["diagnosis_time_source"].value_counts().to_dict() if not cohort.empty else {}
    )

    return cohort, qc_all


def serialise_qc(obj: object) -> object:
    """JSON-serialisable QC (timestamps -> str)."""
    if isinstance(obj, dict):
        return {k: serialise_qc(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [serialise_qc(x) for x in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj
