"""Study 2 revised Phase 1: ICD-10-anchored cohort with 1:2 negative controls.

Positive cohort
    ICD-10 I50.x  → heart_failure
    ICD-10 A41.x  → sepsis
    Anchor: admittime of the *first* qualifying admission per patient.
    Inclusion criterion: ≥ min_cxr_studies CXR studies in the 14-day pre-admission window
    [admittime - window_days, admittime).

Negative controls (1:2 ratio per disease type)
    Patients in MIMIC-CXR with NO I50.x or A41.x code ever.
    Anchor: admittime of a randomly selected admission that itself has ≥ min_cxr_studies
    frontal CXRs in the preceding window.  One anchor per patient per disease type.

Output CSV columns (compatible with run_inference.py):
    subject_id, hadm_id, disease_type, diagnosis_time, window_days, label,
    n_cxr_studies_in_window, diagnosis_time_source

Example::

    python code/study2/scripts/build_index_cohort_icd10.py \\
        --diagnoses-csv  data/MIMIC-IV/csv/diagnoses_icd.csv \\
        --admissions-csv data/MIMIC-IV/csv/admissions.csv \\
        --cxr-metadata-csv data/MIMIC-CXR/csv/mimic-cxr-2.0.0-metadata.csv \\
        --output-dir data/MIMIC-CXR/csv/study2_cohort_icd10

Dry-run (fast check without sampling):

    python code/study2/scripts/build_index_cohort_icd10.py --dry-run
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

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data"

# ICD-10 prefix → disease label
DISEASE_ICD10_PREFIXES: dict[str, str] = {
    "I50": "heart_failure",
    "A41": "sepsis",
}

WINDOW_DAYS = 14
MIN_CXR_STUDIES = 2
NEGATIVE_RATIO = 2
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def _load_diagnoses(path: Path) -> pd.DataFrame:
    logger.info("Loading diagnoses_icd from %s", path)
    df = pd.read_csv(
        path,
        usecols=["subject_id", "hadm_id", "icd_code", "icd_version"],
        dtype={"subject_id": "int64", "hadm_id": "int64", "icd_code": str, "icd_version": str},
    )
    # Keep ICD-10 only (icd_version == "10") and strip whitespace
    df = df[df["icd_version"].str.strip() == "10"].copy()
    df["icd_code"] = df["icd_code"].str.strip()
    logger.info("ICD-10 rows: %d", len(df))
    return df


def _load_admissions(path: Path) -> pd.DataFrame:
    logger.info("Loading admissions from %s", path)
    df = pd.read_csv(
        path,
        usecols=["subject_id", "hadm_id", "admittime", "dischtime"],
        dtype={"subject_id": "int64", "hadm_id": "int64"},
        parse_dates=["admittime", "dischtime"],
    )
    df = df.drop_duplicates("hadm_id", keep="first").reset_index(drop=True)
    logger.info("Admissions loaded: %d", len(df))
    return df


def _load_cxr_frontal_studies(path: Path) -> pd.DataFrame:
    """One row per (subject_id, study_id) — earliest study_datetime among frontal views."""
    logger.info("Loading CXR metadata from %s", path)
    meta = pd.read_csv(
        path,
        usecols=["subject_id", "study_id", "StudyDate", "StudyTime", "ViewPosition"],
        dtype={
            "subject_id": "int64",
            "study_id": "int64",
            "StudyDate": "string",
            "StudyTime": "string",
            "ViewPosition": "string",
        },
    )
    frontal = meta[meta["ViewPosition"].str.upper().isin({"PA", "AP"})].copy()
    frontal["StudyDate"] = frontal["StudyDate"].fillna("").str.replace(r"\.0$", "", regex=True)
    frontal["StudyTime"] = (
        frontal["StudyTime"]
        .fillna("")
        .str.replace(r"\.0+$", "", regex=True)
        .str.extract(r"^(\d{1,6})", expand=False)
        .fillna("")
        .str.zfill(6)
    )
    frontal["study_datetime"] = pd.to_datetime(
        frontal["StudyDate"] + frontal["StudyTime"],
        format="%Y%m%d%H%M%S",
        errors="coerce",
    )
    frontal = frontal.dropna(subset=["study_datetime"])
    studies = (
        frontal.groupby(["subject_id", "study_id"], as_index=False)["study_datetime"]
        .min()
        .sort_values(["subject_id", "study_datetime"])
        .reset_index(drop=True)
    )
    logger.info("Frontal CXR studies with valid timestamps: %d", len(studies))
    return studies


# ---------------------------------------------------------------------------
# Positive cohort
# ---------------------------------------------------------------------------


def _tag_positive_patients(diagnoses: pd.DataFrame) -> dict[str, set[int]]:
    """Return {disease_type: set_of_subject_ids} for ICD-10 prefix matches."""
    result: dict[str, set[int]] = {}
    for prefix, disease in DISEASE_ICD10_PREFIXES.items():
        mask = diagnoses["icd_code"].str.startswith(prefix)
        pids = set(diagnoses.loc[mask, "subject_id"].unique())
        result[disease] = pids
        logger.info("ICD-10 %s (%s): %d unique patients", prefix, disease, len(pids))
    return result


def _first_qualifying_admission(
    diagnoses: pd.DataFrame,
    admissions: pd.DataFrame,
    cxr_studies: pd.DataFrame,
    positive_patients: dict[str, set[int]],
    *,
    window_days: int,
    min_cxr_studies: int,
) -> pd.DataFrame:
    """One row per (subject_id, disease_type) for the earliest ICD-10-positive admission
    that has ≥ min_cxr_studies frontal CXR studies in [admittime-window, admittime)."""
    rows: list[dict[str, object]] = []
    delta = pd.Timedelta(days=window_days)

    # Build lookup: hadm_id → admittime
    adm_map = admissions.set_index("hadm_id")[["subject_id", "admittime", "dischtime"]]

    for disease, prefix in [
        ("heart_failure", "I50"),
        ("sepsis", "A41"),
    ]:
        pos_pids = positive_patients[disease]
        # All qualifying admissions for this disease
        qual_hadm = diagnoses.loc[
            diagnoses["icd_code"].str.startswith(prefix)
            & diagnoses["subject_id"].isin(pos_pids),
            ["subject_id", "hadm_id"],
        ].drop_duplicates()

        # Attach admittime
        qual = qual_hadm.merge(
            admissions[["subject_id", "hadm_id", "admittime", "dischtime"]],
            on=["subject_id", "hadm_id"],
            how="inner",
        )
        qual = qual.dropna(subset=["admittime"]).sort_values(["subject_id", "admittime"])

        # Keep first admission per patient
        first_adm = qual.drop_duplicates("subject_id", keep="first").reset_index(drop=True)
        logger.info(
            "%s: %d patients with first qualifying admission",
            disease,
            len(first_adm),
        )

        # Count CXRs in [admittime - window, admittime)
        n_ok = 0
        for _, r in first_adm.iterrows():
            sid = int(r["subject_id"])
            t_end = pd.Timestamp(r["admittime"])
            t_start = t_end - delta
            sub = cxr_studies[
                (cxr_studies["subject_id"] == sid)
                & (cxr_studies["study_datetime"] >= t_start)
                & (cxr_studies["study_datetime"] < t_end)
            ]
            n_cxr = int(sub["study_id"].nunique())
            if n_cxr < min_cxr_studies:
                continue
            rows.append(
                {
                    "subject_id": sid,
                    "hadm_id": int(r["hadm_id"]),
                    "disease_type": disease,
                    "diagnosis_time": t_end,
                    "diagnosis_time_source": "icd10_admittime",
                    "window_days": window_days,
                    "n_cxr_studies_in_window": n_cxr,
                    "label": 1,
                }
            )
            n_ok += 1
        logger.info(
            "%s positives passing CXR filter (≥%d studies): %d / %d",
            disease,
            min_cxr_studies,
            n_ok,
            len(first_adm),
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["subject_id"] = out["subject_id"].astype(np.int64)
        out["hadm_id"] = out["hadm_id"].astype(np.int64)
    return out


# ---------------------------------------------------------------------------
# Negative controls
# ---------------------------------------------------------------------------


def _build_negative_controls(
    positive_pids_all: set[int],
    admissions: pd.DataFrame,
    cxr_studies: pd.DataFrame,
    n_needed_per_disease: dict[str, int],
    *,
    window_days: int,
    min_cxr_studies: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Sample negative controls: patients never diagnosed with HF or sepsis.

    For each eligible patient, finds admissions with ≥ min_cxr_studies frontal CXR
    studies in [admittime - window_days, admittime).  Randomly picks one such admission
    as the anchor per (patient, disease_type) pair.
    """
    # Negative candidates = patients in CXR metadata that are NOT in any positive set
    all_cxr_pids = set(cxr_studies["subject_id"].unique())
    neg_pool = all_cxr_pids - positive_pids_all
    logger.info("Negative candidate pool (no HF/sepsis ICD-10): %d patients", len(neg_pool))

    # For each negative patient, find admissions with enough prior CXRs
    neg_adm = admissions[admissions["subject_id"].isin(neg_pool)].copy()
    neg_adm = neg_adm.dropna(subset=["admittime"]).sort_values(["subject_id", "admittime"])
    logger.info("Admissions for negative candidates: %d", len(neg_adm))

    delta = pd.Timedelta(days=window_days)
    valid_rows: list[dict[str, object]] = []

    for _, r in neg_adm.iterrows():
        sid = int(r["subject_id"])
        t_end = pd.Timestamp(r["admittime"])
        t_start = t_end - delta
        sub = cxr_studies[
            (cxr_studies["subject_id"] == sid)
            & (cxr_studies["study_datetime"] >= t_start)
            & (cxr_studies["study_datetime"] < t_end)
        ]
        n_cxr = int(sub["study_id"].nunique())
        if n_cxr < min_cxr_studies:
            continue
        valid_rows.append(
            {
                "subject_id": sid,
                "hadm_id": int(r["hadm_id"]),
                "admittime": t_end,
                "n_cxr": n_cxr,
            }
        )

    valid_df = pd.DataFrame(valid_rows)
    if valid_df.empty:
        logger.error("No negative candidates satisfy CXR filter — cannot sample controls.")
        return pd.DataFrame()

    # One row per patient: randomly pick one valid admission per patient
    valid_df = valid_df.sample(frac=1, random_state=int(rng.integers(1_000_000))).reset_index(drop=True)
    one_per_patient = valid_df.drop_duplicates("subject_id", keep="first")
    logger.info(
        "Negative candidates with ≥%d CXR studies in window: %d patients",
        min_cxr_studies,
        len(one_per_patient),
    )

    out_rows: list[dict[str, object]] = []
    for disease, n_needed in n_needed_per_disease.items():
        available = len(one_per_patient)
        if available < n_needed:
            logger.warning(
                "%s: requested %d negatives but only %d available; using all.",
                disease,
                n_needed,
                available,
            )
            n_needed = available

        sampled = one_per_patient.sample(n=n_needed, replace=False, random_state=int(rng.integers(1_000_000)))
        for _, r in sampled.iterrows():
            out_rows.append(
                {
                    "subject_id": int(r["subject_id"]),
                    "hadm_id": int(r["hadm_id"]),
                    "disease_type": disease,
                    "diagnosis_time": r["admittime"],
                    "diagnosis_time_source": "negative_control_admittime",
                    "window_days": window_days,
                    "n_cxr_studies_in_window": int(r["n_cxr"]),
                    "label": 0,
                }
            )
        logger.info("Sampled %d negative controls for %s", n_needed, disease)

    out = pd.DataFrame(out_rows)
    if not out.empty:
        out["subject_id"] = out["subject_id"].astype(np.int64)
        out["hadm_id"] = out["hadm_id"].astype(np.int64)
    return out


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def build_icd10_cohort(
    diagnoses_csv: Path,
    admissions_csv: Path,
    cxr_metadata_csv: Path,
    *,
    window_days: int = WINDOW_DAYS,
    min_cxr_studies: int = MIN_CXR_STUDIES,
    negative_ratio: int = NEGATIVE_RATIO,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rng = np.random.default_rng(seed)

    diagnoses = _load_diagnoses(diagnoses_csv)
    admissions = _load_admissions(admissions_csv)
    cxr_studies = _load_cxr_frontal_studies(cxr_metadata_csv)

    positive_patients = _tag_positive_patients(diagnoses)
    all_positive_pids: set[int] = set().union(*positive_patients.values())

    positives = _first_qualifying_admission(
        diagnoses,
        admissions,
        cxr_studies,
        positive_patients,
        window_days=window_days,
        min_cxr_studies=min_cxr_studies,
    )
    logger.info("Positive cohort total rows: %d", len(positives))

    # Compute negatives needed per disease
    n_needed: dict[str, int] = {}
    for disease in DISEASE_ICD10_PREFIXES.values():
        n_pos = int((positives["disease_type"] == disease).sum())
        n_needed[disease] = n_pos * negative_ratio
        logger.info(
            "%s: %d positives → %d negatives requested (ratio 1:%d)",
            disease,
            n_pos,
            n_needed[disease],
            negative_ratio,
        )

    negatives = _build_negative_controls(
        all_positive_pids,
        admissions,
        cxr_studies,
        n_needed,
        window_days=window_days,
        min_cxr_studies=min_cxr_studies,
        rng=rng,
    )
    logger.info("Negative controls total rows: %d", len(negatives))

    cohort = pd.concat([positives, negatives], ignore_index=True)
    cohort = cohort.sort_values(["disease_type", "label", "subject_id"]).reset_index(drop=True)

    qc: dict[str, object] = {
        "window_days": window_days,
        "min_cxr_studies": min_cxr_studies,
        "negative_ratio": negative_ratio,
        "seed": seed,
        "cohort_summary": _summary_table(cohort),
    }
    return cohort, qc


def _summary_table(cohort: pd.DataFrame) -> dict[str, object]:
    rows = []
    for (disease, label), grp in cohort.groupby(["disease_type", "label"]):
        rows.append(
            {
                "disease_type": str(disease),
                "label": int(label),
                "label_name": "positive" if int(label) == 1 else "negative",
                "n_patients": int(grp["subject_id"].nunique()),
                "n_rows": int(len(grp)),
                "median_cxr_studies_in_window": float(
                    np.median(grp["n_cxr_studies_in_window"].to_numpy())
                ),
            }
        )
    return {
        "by_disease_and_label": rows,
        "total_rows": int(len(cohort)),
        "total_unique_patients": int(cohort["subject_id"].nunique()),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Study 2 revised Phase 1: ICD-10-anchored cohort (HF=I50.x, Sepsis=A41.x) "
            "with 1:2 negative controls (patients with no HF/sepsis ICD-10 ever)."
        )
    )
    parser.add_argument(
        "--diagnoses-csv",
        type=Path,
        default=DATA_ROOT / "MIMIC-IV" / "csv" / "diagnoses_icd.csv",
    )
    parser.add_argument(
        "--admissions-csv",
        type=Path,
        default=DATA_ROOT / "MIMIC-IV" / "csv" / "admissions.csv",
    )
    parser.add_argument(
        "--cxr-metadata-csv",
        type=Path,
        default=DATA_ROOT / "MIMIC-CXR" / "csv" / "mimic-cxr-2.0.0-metadata.csv",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=WINDOW_DAYS,
        help="Pre-admission window length in days.",
    )
    parser.add_argument(
        "--min-cxr-studies",
        type=int,
        default=MIN_CXR_STUDIES,
        help="Minimum frontal CXR studies required in the window.",
    )
    parser.add_argument(
        "--negative-ratio",
        type=int,
        default=NEGATIVE_RATIO,
        help="Negative controls per positive (default 2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for negative sampling.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_ROOT / "MIMIC-CXR" / "csv" / "study2_cohort_icd10",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="study2_icd10_cohort",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary counts and exit without writing files.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    configure_study_logging(study_name="study2_icd10_cohort", level=getattr(logging, args.log_level))

    for name, p in [
        ("diagnoses_icd", args.diagnoses_csv),
        ("admissions", args.admissions_csv),
        ("cxr_metadata", args.cxr_metadata_csv),
    ]:
        if not p.exists():
            logger.error("Required file missing — %s: %s", name, p)
            sys.exit(1)

    cohort, qc = build_icd10_cohort(
        args.diagnoses_csv,
        args.admissions_csv,
        args.cxr_metadata_csv,
        window_days=args.window_days,
        min_cxr_studies=args.min_cxr_studies,
        negative_ratio=args.negative_ratio,
        seed=args.seed,
    )

    summary = qc.get("cohort_summary", {})
    print(f"\n{'Disease':<15} {'Label':<10} {'N rows':>8} {'N patients':>12} {'Median CXRs':>12}")
    print("-" * 60)
    for r in summary.get("by_disease_and_label", []):
        print(
            f"{r['disease_type']:<15} {r['label_name']:<10} "
            f"{r['n_rows']:>8} {r['n_patients']:>12} {r['median_cxr_studies_in_window']:>12.1f}"
        )
    print("-" * 60)
    print(f"Total rows: {summary.get('total_rows', len(cohort))}   "
          f"Unique patients: {summary.get('total_unique_patients', '—')}")

    if args.dry_run:
        print("\n[dry-run] Exiting without writing files.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    out_csv = args.output_dir / f"{args.prefix}.csv"
    cohort.to_csv(out_csv, index=False)
    logger.info("Wrote cohort CSV: %s (%d rows)", out_csv, len(cohort))

    qc_serial = _serialise(qc)
    qc_path = args.output_dir / f"{args.prefix}_qc.json"
    qc_path.write_text(json.dumps(qc_serial, indent=2), encoding="utf-8")
    logger.info("Wrote QC: %s", qc_path)

    summary_path = args.output_dir / f"{args.prefix}_summary.json"
    summary_path.write_text(json.dumps(qc_serial["cohort_summary"], indent=2), encoding="utf-8")

    # Per-disease splits
    for disease in ("heart_failure", "sepsis"):
        sub = cohort[cohort["disease_type"] == disease]
        sub.to_csv(args.output_dir / f"{args.prefix}_{disease}.csv", index=False)
        logger.info("Wrote %s split: %d rows", disease, len(sub))

    print(f"\n[build_index_cohort_icd10] Done. Output: {args.output_dir}")


def _serialise(obj: object) -> object:
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialise(x) for x in obj]
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


if __name__ == "__main__":
    run()
