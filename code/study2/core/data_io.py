from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .constants import FRONTAL_VIEW_CODES

logger = logging.getLogger(__name__)

_TEMPORAL_PAIRS_CACHE_SCHEMA = 1


def _file_fingerprint(path: Path) -> dict[str, object]:
    st = path.stat()
    return {
        "path": str(path.resolve()),
        "mtime_ns": int(st.st_mtime_ns),
        "size": int(st.st_size),
    }


def _temporal_pairs_cache_paths(cache_dir: Path, metadata_csv: Path) -> tuple[Path, Path]:
    """Return (pickle_path, meta_json_path) for this metadata file."""
    h = hashlib.sha256(str(metadata_csv.resolve()).encode()).hexdigest()[:16]
    base = cache_dir / f"temporal_pairs_{metadata_csv.stem}_{h}"
    return base.with_suffix(".pkl"), base.with_suffix(".meta.json")


def load_or_build_temporal_pairs(
    metadata_csv: Path,
    *,
    cache_dir: Path | None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load frontal metadata, build consecutive-study pairs, optionally from disk cache.

    When ``use_cache`` and ``cache_dir`` are set, skips CSV load and pair construction
    if the cache exists and matches the metadata file's size and modification time.
    """
    if not use_cache or cache_dir is None:
        metadata = load_metadata_frontal(metadata_csv)
        return build_temporal_pairs(metadata)

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    pkl_path, meta_path = _temporal_pairs_cache_paths(cache_dir, metadata_csv)
    fp = _file_fingerprint(metadata_csv)

    if pkl_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Ignoring invalid temporal pairs cache meta (%s): %s", meta_path, e)
        else:
            if (
                meta.get("schema") == _TEMPORAL_PAIRS_CACHE_SCHEMA
                and meta.get("metadata") == fp
            ):
                logger.info("Loading temporal pairs from cache %s", pkl_path)
                return pd.read_pickle(pkl_path)
            logger.info("Temporal pairs cache stale or schema mismatch; rebuilding.")

    metadata = load_metadata_frontal(metadata_csv)
    pairs = build_temporal_pairs(metadata)
    pairs.to_pickle(pkl_path)
    meta_path.write_text(
        json.dumps(
            {
                "schema": _TEMPORAL_PAIRS_CACHE_SCHEMA,
                "metadata": fp,
                "n_pairs": int(len(pairs)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Wrote temporal pairs cache (%d pairs) to %s", len(pairs), pkl_path)
    return pairs


def validate_required_paths(paths: dict[str, Path]) -> None:
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        details = "\n".join(f"  - {name}: {paths[name]}" for name in missing)
        raise FileNotFoundError(f"Missing required files/directories:\n{details}")


def load_metadata_frontal(metadata_csv: Path) -> pd.DataFrame:
    """Load MIMIC-CXR metadata and return one frontal-view row per study.

    PA is preferred over AP when both exist for the same study.
    Rows with unparseable study_datetime are dropped.
    """
    logger.info("Loading metadata from %s", metadata_csv)
    metadata = pd.read_csv(
        metadata_csv,
        usecols=["subject_id", "study_id", "dicom_id", "ViewPosition", "StudyDate", "StudyTime"],
        dtype={
            "subject_id": "int64",
            "study_id": "int64",
            "dicom_id": "string",
            "ViewPosition": "string",
            "StudyDate": "string",
            "StudyTime": "string",
        },
    )
    logger.info("Raw metadata rows: %d", len(metadata))

    frontal = metadata[metadata["ViewPosition"].isin(FRONTAL_VIEW_CODES)].copy()
    logger.info("Frontal-view rows: %d", len(frontal))

    # Prefer PA over AP for the same study
    frontal["_view_rank"] = (frontal["ViewPosition"] == "PA").astype(int)
    frontal = (
        frontal.sort_values(["study_id", "_view_rank"], ascending=[True, False])
        .drop_duplicates("study_id")
        .drop(columns="_view_rank")
        .reset_index(drop=True)
    )

    frontal["StudyDate"] = frontal["StudyDate"].fillna("").str.replace(r"\.0$", "", regex=True)
    frontal["StudyTime"] = frontal["StudyTime"].fillna("").str.replace(r"\.0+$", "", regex=True)
    frontal["StudyTime"] = frontal["StudyTime"].str.extract(r"^(\d{1,6})", expand=False).fillna("")
    frontal["StudyTime"] = frontal["StudyTime"].str.zfill(6)
    frontal["study_datetime"] = pd.to_datetime(
        frontal["StudyDate"] + frontal["StudyTime"],
        format="%Y%m%d%H%M%S",
        errors="coerce",
    )

    n_before = len(frontal)
    frontal = frontal.dropna(subset=["study_datetime"]).reset_index(drop=True)
    n_dropped = n_before - len(frontal)
    if n_dropped:
        logger.warning("Dropped %d studies with unparseable timestamp", n_dropped)

    logger.info("Unique studies with frontal image and valid timestamp: %d", len(frontal))
    return frontal[["subject_id", "study_id", "dicom_id", "ViewPosition", "study_datetime"]]


def build_temporal_pairs(metadata: pd.DataFrame) -> pd.DataFrame:
    """For each subject with ≥2 frontal studies, yield consecutive (prior, current) pairs.

    Pairs are sorted chronologically per subject; the earliest study has no prior.
    """
    rows: list[dict[str, object]] = []
    n_subjects = metadata["subject_id"].nunique()
    logger.info("Building temporal pairs across %d subjects", n_subjects)

    grouped = metadata.groupby("subject_id", sort=False)
    for subject_id, group in tqdm(
        grouped,
        total=n_subjects,
        desc="Building temporal pairs",
        unit="subject",
    ):
        studies = group.sort_values("study_datetime").reset_index(drop=True)
        if len(studies) < 2:
            continue
        for i in range(1, len(studies)):
            prior = studies.iloc[i - 1]
            current = studies.iloc[i]
            delta_days = (
                current["study_datetime"] - prior["study_datetime"]
            ).total_seconds() / 86400.0
            rows.append(
                {
                    "subject_id": int(subject_id),
                    "current_study_id": int(current["study_id"]),
                    "prior_study_id": int(prior["study_id"]),
                    "current_datetime": current["study_datetime"],
                    "prior_datetime": prior["study_datetime"],
                    "days_between": round(delta_days, 3),
                    "current_dicom_id": str(current["dicom_id"]),
                    "prior_dicom_id": str(prior["dicom_id"]),
                    "current_view": str(current["ViewPosition"]),
                    "prior_view": str(prior["ViewPosition"]),
                }
            )

    pairs = pd.DataFrame(rows)
    logger.info(
        "Built %d temporal pairs from %d eligible subjects",
        len(pairs),
        pairs["subject_id"].nunique() if not pairs.empty else 0,
    )
    return pairs


def image_path_for(
    images_root: Path, subject_id: int, study_id: int, dicom_id: str
) -> Path:
    """Filesystem path for a MIMIC-CXR-JPG file (may not exist on disk).

    Expected layout:
      <images_root>/p{subject_id[:2]}/p{subject_id}/s{study_id}/{dicom_id}.jpg
    """
    subject_prefix = f"p{str(subject_id)[:2]}"
    base = images_root / subject_prefix / f"p{subject_id}" / f"s{study_id}"
    d = str(dicom_id).strip()
    name = d if d.lower().endswith(".jpg") else f"{d}.jpg"
    return base / name


def locate_image(
    images_root: Path, subject_id: int, study_id: int, dicom_id: str
) -> Path | None:
    """Return the JPG path for a MIMIC-CXR-JPG image, or None if not found."""
    path = image_path_for(images_root, subject_id, study_id, dicom_id)
    return path if path.exists() else None


def resolve_pair_image_paths(
    pairs: pd.DataFrame, images_root: Path
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Add current_image_path / prior_image_path columns; drop rows where either is missing."""
    current_paths: list[str | None] = []
    prior_paths: list[str | None] = []

    for _, row in pairs.iterrows():
        current_paths.append(
            str(p) if (p := locate_image(images_root, row["subject_id"], row["current_study_id"], row["current_dicom_id"])) else None
        )
        prior_paths.append(
            str(p) if (p := locate_image(images_root, row["subject_id"], row["prior_study_id"], row["prior_dicom_id"])) else None
        )

    pairs = pairs.copy()
    pairs["current_image_path"] = current_paths
    pairs["prior_image_path"] = prior_paths

    n_total = len(pairs)
    missing_current = pairs["current_image_path"].isna().sum()
    missing_prior = pairs["prior_image_path"].isna().sum()

    example_current: Path | None = None
    example_prior: Path | None = None
    if n_total > 0:
        r0 = pairs.iloc[0]
        example_current = image_path_for(
            images_root, int(r0["subject_id"]), int(r0["current_study_id"]), str(r0["current_dicom_id"])
        )
        example_prior = image_path_for(
            images_root, int(r0["subject_id"]), int(r0["prior_study_id"]), str(r0["prior_dicom_id"])
        )

    pairs = pairs.dropna(subset=["current_image_path", "prior_image_path"]).reset_index(drop=True)
    n_kept = len(pairs)

    if n_kept == 0 and n_total > 0 and example_current is not None and example_prior is not None:
        logger.warning(
            "No images resolved; example paths for first pair — current=%s (exists=%s), prior=%s (exists=%s)",
            example_current,
            example_current.exists(),
            example_prior,
            example_prior.exists(),
        )

    qc = {
        "n_pairs_total": n_total,
        "n_pairs_missing_current_image": int(missing_current),
        "n_pairs_missing_prior_image": int(missing_prior),
        "n_pairs_resolved": n_kept,
    }
    logger.info(
        "Image resolution: total=%d missing_current=%d missing_prior=%d resolved=%d",
        n_total, missing_current, missing_prior, n_kept,
    )
    return pairs, qc


def build_cohort_study_sequence_table(
    cohort: pd.DataFrame,
    frontal_studies: pd.DataFrame,
    images_root: Path | None,
    *,
    min_resolved_studies: int = 3,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """One row per frontal CXR study in the pre-diagnosis window, chronologically per cohort event.

    Cohort rows must include ``subject_id``, ``hadm_id``, ``disease_type``, ``diagnosis_time``.
    If ``window_days`` is absent it defaults to 14; ``window_start`` is derived when missing.

    ``frontal_studies`` should be one frontal row per study (e.g. from :func:`load_metadata_frontal`).
    When ``images_root`` is set, rows without a resolvable JPG are dropped; groups with fewer than
    ``min_resolved_studies`` remaining rows are removed (frontal-on-disk can be stricter than Phase 1
    counts that used all metadata views).

    Returns
    -------
    table
        Columns include ``seq_index`` (0-based order in window), ``image_path``, ``hours_before_diagnosis``.
    qc
        JSON-serialisable counts for logging / QC files.
    """
    required = {"subject_id", "hadm_id", "disease_type", "diagnosis_time"}
    missing_cols = required - set(cohort.columns)
    if missing_cols:
        raise ValueError(f"Cohort missing columns: {sorted(missing_cols)}")

    cohort = cohort.copy()
    cohort["diagnosis_time"] = pd.to_datetime(cohort["diagnosis_time"], errors="coerce")
    cohort = cohort.dropna(subset=["diagnosis_time"])
    if cohort.empty:
        return pd.DataFrame(), {"error": "empty_cohort_after_diagnosis_time_parse"}

    if "window_days" not in cohort.columns:
        cohort["window_days"] = 14
    if "window_start" not in cohort.columns:
        cohort["window_start"] = cohort["diagnosis_time"] - pd.to_timedelta(
            cohort["window_days"], unit="D"
        )

    meta_cols = {"subject_id", "study_id", "dicom_id", "study_datetime"}
    if not meta_cols.issubset(frontal_studies.columns):
        raise ValueError(f"frontal_studies must contain {sorted(meta_cols)}")

    m = cohort.merge(frontal_studies, on="subject_id", how="inner")
    win = m[
        (m["study_datetime"] >= m["window_start"]) & (m["study_datetime"] <= m["diagnosis_time"])
    ].copy()

    qc_pre_dedupe = int(len(win))
    win = win.drop_duplicates(
        subset=["subject_id", "hadm_id", "disease_type", "study_id"], keep="first"
    )
    n_rows_after_study_dedupe = int(len(win))

    win = win.sort_values(
        ["subject_id", "hadm_id", "disease_type", "study_datetime"], kind="mergesort"
    ).reset_index(drop=True)

    if images_root is not None:
        roots = Path(images_root)
        paths: list[str | None] = []
        for sid, stid, did in zip(
            win["subject_id"].to_numpy(),
            win["study_id"].to_numpy(),
            win["dicom_id"].to_numpy(),
        ):
            p = locate_image(roots, int(sid), int(stid), str(did))
            paths.append(str(p) if p else None)
        win["image_path"] = paths
        win = win.dropna(subset=["image_path"]).reset_index(drop=True)
    else:
        win["image_path"] = None

    win["hours_before_diagnosis"] = (
        win["diagnosis_time"] - win["study_datetime"]
    ).dt.total_seconds() / 3600.0

    grp_cols = ["subject_id", "hadm_id", "disease_type"]
    counts = win.groupby(grp_cols, sort=False).size().reset_index(name="n_resolved_studies")
    win = win.merge(counts, on=grp_cols, how="left")
    n_event_groups_before = int(win[grp_cols].drop_duplicates().shape[0])
    win = win[win["n_resolved_studies"] >= min_resolved_studies].reset_index(drop=True)
    n_event_groups_after = int(win[grp_cols].drop_duplicates().shape[0])

    win["seq_index"] = win.groupby(grp_cols, sort=False).cumcount()

    qc: dict[str, object] = {
        "n_cohort_rows_input": int(len(cohort)),
        "n_merged_rows_time_window_pre_dedupe": qc_pre_dedupe,
        "n_rows_after_study_dedupe": n_rows_after_study_dedupe,
        "n_event_groups_before_min_resolved": n_event_groups_before,
        "n_event_groups_after_min_resolved": n_event_groups_after,
        "n_output_rows": int(len(win)),
        "min_resolved_studies": int(min_resolved_studies),
    }
    logger.info(
        "Sequence table: cohort_rows=%d event_groups=%d output_rows=%d (min_studies=%d)",
        qc["n_cohort_rows_input"],
        n_event_groups_after,
        len(win),
        min_resolved_studies,
    )
    return win, qc
