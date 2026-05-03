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
