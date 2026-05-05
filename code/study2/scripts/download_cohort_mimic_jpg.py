"""Download a small subset of MIMIC-CXR-JPG files for Study 2 cohort subjects.

Selects frontal studies that fall in each cohort event's pre-diagnosis window (same
construction as ``study2.core.data_io.build_cohort_study_sequence_table`` with
``images_root=None`` for metadata-only listing), then downloads up to ``--max-images``
unique JPG files from PhysioNet.

Prerequisites (you must do this outside this script):
  - PhysioNet account: https://physionet.org/register/
  - Complete required training and request access to *MIMIC-CXR-JPG* (and MIMIC-CXR
    metadata if you use your own copy).
  - Use your PhysioNet **username** and **password** below or via environment variables
    ``PHYSIONET_USER`` and ``PHYSIONET_PASSWORD`` (preferred for passwords).
  - **MIMIC-CXR-JPG** is a separate restricted project from **MIMIC-CXR**; you must request
    and receive access on the JPG project page or every download returns HTTP 403.

Example::

  $env:PHYSIONET_USER = "your_username"
  $env:PHYSIONET_PASSWORD= "your_password"
  python code/study2/scripts/download_cohort_mimic_jpg.py --max-images 100

If downloads fail with HTTP 403 despite valid credentials, set
``$env:PHYSIONET_HTTP_USER_AGENT`` to a browser string or ``Wget/1.21.4``, or pass
``--user-agent`` (some networks block uncommon user agents).

Images are written under ``<output-root>/files/...`` so ``--images-root`` for
``run_inference.py`` should be ``<output-root>/files`` (default matches project layout).
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterator

import pandas as pd
from tqdm import tqdm

_CODE_DIR = Path(__file__).resolve().parents[2]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from study_logging import configure_study_logging
from study2.core.data_io import build_cohort_study_sequence_table, load_metadata_frontal

logger = logging.getLogger(__name__)

_physionet_403_hint_logged = False

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data"

# Default matches current PhysioNet wget instructions (v2.1.0). Older v2.0.0 URLs may return HTTP 403
# even when your account has access to the latest release.
DEFAULT_PHYSIONET_FILES_BASE = "https://physionet.org/files/mimic-cxr-jpg/2.1.0/files"

# Wget-style default: PhysioNet documents wget; some CDNs return 403 for rare User-Agents even with valid Basic auth.
DEFAULT_PHYSIONET_HTTP_USER_AGENT = "Wget/1.21.4"


def normalize_mimic_cxr_jpg_files_base_url(base: str) -> str:
    """Map browser ``/content/...`` bases to the wget mirror and ensure a ``.../files`` suffix.

    PhysioNet serves the same tree at ``https://physionet.org/content/mimic-cxr-jpg/<ver>/files/``
    (browser) and ``https://physionet.org/files/mimic-cxr-jpg/<ver>/files/...`` (wget / HTTP).
    This script's relative paths are ``p10/...`` under that ``files`` root.
    """
    s = base.strip().rstrip("/")
    if "/content/mimic-cxr-jpg/" in s:
        s = s.replace("/content/mimic-cxr-jpg/", "/files/mimic-cxr-jpg/", 1)
    # Accept wget-style base ``.../mimic-cxr-jpg/2.1.0`` (no trailing ``files``).
    if re.search(r"/mimic-cxr-jpg/\d+\.\d+\.\d+$", s):
        s = f"{s}/files"
    return s


def _mimic_cxr_jpg_content_mirror_base(files_base_url: str) -> str:
    """Map ``/files/mimic-cxr-jpg/`` base to the browser ``/content/mimic-cxr-jpg/`` host path."""
    return files_base_url.replace("/files/mimic-cxr-jpg/", "/content/mimic-cxr-jpg/", 1)


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


def _subject_prefix(subject_id: int) -> str:
    return f"p{str(int(subject_id))[:2]}"


def _local_jpg_relpath(subject_id: int, study_id: int, dicom_id: str) -> Path:
    """Path under ``files/`` root: pXX/p<subject>/s<study>/<dicom>.jpg."""
    sid = int(subject_id)
    stid = int(study_id)
    d = str(dicom_id).strip()
    if d.lower().endswith(".jpg"):
        name = d
    else:
        name = f"{d}.jpg"
    return Path(_subject_prefix(sid)) / f"p{sid}" / f"s{stid}" / name


def _physionet_url(base: str, rel: Path) -> str:
    rel_posix = rel.as_posix().rstrip("/")
    return f"{base.rstrip('/')}/{rel_posix}"


def _build_opener() -> urllib.request.OpenerDirector:
    """Plain opener; credentials are preemptive Basic Auth on each :class:`urllib.request.Request`."""
    return urllib.request.build_opener()


def _request_headers_basic(username: str, password: str, *, user_agent: str) -> dict[str, str]:
    """Preemptive Basic Auth — PhysioNet often returns 403 without a 401 challenge."""
    token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    return {
        "User-Agent": user_agent,
        "Authorization": f"Basic {token}",
        "Accept": "*/*",
    }


# Example path from MIMIC-CXR-JPG data description (patient p10000032; same layout in v2.0.0 / v2.1.0).
_PROBE_JPG_REL = Path("p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg")


def _probe_physionet_single_file_url(
    opener: urllib.request.OpenerDirector,
    url: str,
    headers: dict[str, str],
) -> None:
    req = urllib.request.Request(url, method="HEAD", headers=headers)
    try:
        with opener.open(req, timeout=60):
            pass
    except urllib.error.HTTPError as exc:
        if exc.code in (403, 405, 501):
            if exc.code == 403:
                logger.debug("HEAD returned %s; retrying probe with GET", exc.code)
            req = urllib.request.Request(url, method="GET", headers=headers)
            with opener.open(req, timeout=120) as resp:
                resp.read(65536)
        else:
            raise


def _verify_physionet_mimic_cxr_jpg_access(
    opener: urllib.request.OpenerDirector,
    files_base_url: str,
    username: str,
    password: str,
    *,
    user_agent: str,
) -> str:
    """Probe a documented example JPG; return the base URL that worked (files or content mirror)."""
    headers = _request_headers_basic(username, password, user_agent=user_agent)
    content_base = _mimic_cxr_jpg_content_mirror_base(files_base_url)
    candidates: list[tuple[str, str]] = [
        (_physionet_url(files_base_url, _PROBE_JPG_REL), files_base_url),
    ]
    if content_base != files_base_url:
        candidates.append((_physionet_url(content_base, _PROBE_JPG_REL), content_base))

    last_http: urllib.error.HTTPError | None = None
    for probe_url, base_for_downloads in candidates:
        try:
            _probe_physionet_single_file_url(opener, probe_url, headers)
            logger.info(
                "PhysioNet MIMIC-CXR-JPG access probe succeeded (base=%s, url=%s)",
                base_for_downloads,
                probe_url,
            )
            return base_for_downloads
        except urllib.error.HTTPError as exc:
            last_http = exc
            if exc.code == 401:
                logger.error(
                    "PhysioNet returned HTTP 401 for the access probe — wrong username/password, "
                    "or use your PhysioNet login name (not necessarily your email). URL: %s",
                    probe_url,
                )
                sys.exit(1)
            logger.debug("Access probe HTTP %s for %s", exc.code, probe_url)
        except urllib.error.URLError as exc:
            logger.error("PhysioNet access probe network error: %s (%s)", exc.reason, probe_url)
            sys.exit(1)
        except OSError as exc:
            logger.error("PhysioNet access probe failed: %s (%s)", exc, probe_url)
            sys.exit(1)

    assert last_http is not None
    primary_url = candidates[0][0]
    if last_http.code == 403:
        logger.error(
            "PhysioNet returned HTTP 403 for the MIMIC-CXR-JPG access probe (tried wget + content mirrors). "
            "Last URL: %s. If your credentials work in wget/browser, try: "
            "(1) ``--user-agent Wget/1.21.4`` or set PHYSIONET_HTTP_USER_AGENT to a normal browser string "
            "(some networks block uncommon User-Agents); "
            "(2) confirm PHYSIONET_USER is your PhysioNet **username**; "
            "(3) open the URL while logged in at physionet.org. "
            "Still stuck: mimic-support@physionet.org.",
            primary_url,
        )
        sys.exit(1)
    logger.error("PhysioNet access probe failed: HTTP %s for %s", last_http.code, primary_url)
    sys.exit(1)


def _download_one(
    opener: urllib.request.OpenerDirector,
    url: str,
    dest: Path,
    *,
    physionet_user: str,
    physionet_password: str,
    user_agent: str,
) -> tuple[bool, str]:
    global _physionet_403_hint_logged
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return True, "skipped_existing"

    try:
        req = urllib.request.Request(
            url,
            headers=_request_headers_basic(physionet_user, physionet_password, user_agent=user_agent),
        )
        with opener.open(req, timeout=120) as resp:
            data = resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 403 and not _physionet_403_hint_logged:
            _physionet_403_hint_logged = True
            logger.warning(
                "HTTP 403 from PhysioNet — if downloads keep failing: (1) use your PhysioNet "
                "**username** (not email) and account password; (2) on "
                "https://physionet.org/content/mimic-cxr-jpg/ confirm you completed training "
                "and **requested access** to MIMIC-CXR-JPG (separate from MIMIC-CXR metadata); "
                "(3) try opening the same URL in a logged-in browser."
            )
        return False, f"http_{exc.code}"
    except urllib.error.URLError as exc:
        return False, f"url_{exc.reason!s}"
    except OSError as exc:
        return False, f"os_{exc}"

    dest.write_bytes(data)
    return True, "downloaded"


def iter_unique_jpg_targets_from_sequence_table(
    sequence_df: pd.DataFrame,
    *,
    max_images: int,
) -> Iterator[tuple[Path, dict[str, object]]]:
    """Yield (relative path under ``files/``, manifest row) up to ``max_images`` unique JPGs.

    Iterates studies in chronological order within each cohort event; skips files already
    counted toward ``max_images``.
    """
    seen: set[tuple[int, int, str]] = set()
    if sequence_df.empty:
        return

    ordered = sequence_df.sort_values(
        ["subject_id", "hadm_id", "disease_type", "study_datetime"], kind="mergesort"
    )
    for _, row in ordered.iterrows():
        if len(seen) >= max_images:
            break
        sid = int(row["subject_id"])
        study_id = int(row["study_id"])
        dicom_id = str(row["dicom_id"]).strip()
        key = (sid, study_id, dicom_id)
        if key in seen:
            continue
        seen.add(key)
        rel = _local_jpg_relpath(sid, study_id, dicom_id)
        yield rel, {
            "subject_id": sid,
            "hadm_id": int(row["hadm_id"]),
            "disease_type": str(row["disease_type"]),
            "study_id": study_id,
            "dicom_id": dicom_id,
            "seq_index": int(row["seq_index"]) if "seq_index" in row.index and pd.notna(row["seq_index"]) else None,
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download a capped subset of MIMIC-CXR-JPG for Phase-1 cohort subjects (PhysioNet)."
    )
    p.add_argument(
        "--cohort-csv",
        type=Path,
        default=DATA_ROOT / "MIMIC-CXR/csv/study2_cohort/study2_index_cohort.csv",
        help="Phase-1 cohort CSV (subject_id, hadm_id, disease_type, diagnosis_time, window_days).",
    )
    p.add_argument(
        "--metadata-csv",
        type=Path,
        default=DATA_ROOT / "MIMIC-CXR/csv/mimic-cxr-2.0.0-metadata.csv",
        help="MIMIC-CXR metadata CSV (same as run_inference).",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=DATA_ROOT / "MIMIC-CXR-JPG",
        help="Directory that will contain a ``files/`` tree (matches full dataset layout).",
    )
    p.add_argument(
        "--max-images",
        type=int,
        default=100,
        help="Maximum number of unique JPG files to download (default: 100).",
    )
    p.add_argument(
        "--min-resolved-studies",
        type=int,
        default=3,
        help="Minimum in-window frontal studies per cohort event (match run_inference).",
    )
    p.add_argument(
        "--physionet-user",
        type=str,
        default=os.environ.get("PHYSIONET_USER", ""),
        help="PhysioNet username (default: env PHYSIONET_USER).",
    )
    p.add_argument(
        "--physionet-password",
        type=str,
        default=os.environ.get("PHYSIONET_PASSWORD", ""),
        help="PhysioNet password (default: env PHYSIONET_PASSWORD; avoid passing on the command line).",
    )
    p.add_argument(
        "--base-url",
        type=str,
        default=DEFAULT_PHYSIONET_FILES_BASE,
        help=(
            "PhysioNet base ending in .../mimic-cxr-jpg/<version>/files (wget mirror), or the browser "
            "https://physionet.org/content/mimic-cxr-jpg/<version>/files URL (normalized automatically)."
        ),
    )
    p.add_argument(
        "--user-agent",
        type=str,
        default=os.environ.get("PHYSIONET_HTTP_USER_AGENT", DEFAULT_PHYSIONET_HTTP_USER_AGENT),
        help=(
            "HTTP User-Agent for PhysioNet (default: wget-like, or env PHYSIONET_HTTP_USER_AGENT). "
            "Use a real browser UA string if you still get HTTP 403 with valid credentials."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List selected files and exit without downloading.",
    )
    p.add_argument(
        "--skip-access-probe",
        action="store_true",
        help="Do not run the one-file HTTP probe before bulk download (not recommended).",
    )
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return p.parse_args()


def run() -> None:
    args = parse_args()
    configure_study_logging(study_name="study2", level=getattr(logging, args.log_level))

    raw_base = args.base_url
    args.base_url = normalize_mimic_cxr_jpg_files_base_url(args.base_url)
    if args.base_url != raw_base.strip().rstrip("/"):
        logger.info("Normalized --base-url %r -> %r", raw_base, args.base_url)

    if not args.cohort_csv.exists():
        logger.error("Cohort CSV not found: %s", args.cohort_csv)
        sys.exit(1)
    if not args.metadata_csv.exists():
        logger.error("Metadata CSV not found: %s", args.metadata_csv)
        sys.exit(1)
    if args.max_images < 1:
        logger.error("--max-images must be >= 1")
        sys.exit(1)

    cohort = _load_index_cohort(args.cohort_csv)
    if cohort.empty:
        logger.error("No valid cohort rows after parsing diagnosis_time: %s", args.cohort_csv)
        sys.exit(1)

    cohort_ids = cohort["subject_id"].unique()
    logger.info("Cohort subjects: %d unique patient IDs", len(cohort_ids))

    metadata = load_metadata_frontal(args.metadata_csv)
    metadata = metadata[metadata["subject_id"].isin(cohort_ids)].reset_index(drop=True)
    if metadata.empty:
        logger.error("No frontal metadata rows for cohort subjects — check metadata/cohort overlap.")
        sys.exit(1)

    seq_df, seq_qc = build_cohort_study_sequence_table(
        cohort,
        metadata,
        images_root=None,
        min_resolved_studies=args.min_resolved_studies,
    )
    if seq_df.empty:
        logger.error(
            "No in-window frontal studies for cohort after sequence filter. QC=%s",
            seq_qc,
        )
        sys.exit(1)

    files_root = args.output_root / "files"
    targets = list(
        iter_unique_jpg_targets_from_sequence_table(seq_df, max_images=args.max_images)
    )
    logger.info(
        "Selected %d unique JPGs from cohort-window study sequences (cap=%d).",
        len(targets),
        args.max_images,
    )

    manifest = {
        "cohort_csv": str(args.cohort_csv),
        "metadata_csv": str(args.metadata_csv),
        "output_root": str(args.output_root),
        "files_root": str(files_root),
        "physionet_base_url": args.base_url,
        "max_images": args.max_images,
        "sequence_table_qc": seq_qc,
        "n_sequence_rows": int(len(seq_df)),
        "n_images_selected": len(targets),
        "dry_run": args.dry_run,
        "images": [],
    }

    if args.dry_run:
        for rel, meta in targets:
            url = _physionet_url(args.base_url, rel)
            manifest["images"].append({**meta, "relative_path": rel.as_posix(), "url": url})
        manifest_path = args.output_root / "download_cohort_mimic_jpg_manifest.json"
        args.output_root.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2))
        logger.info("Dry run: wrote manifest %s", manifest_path)
        return

    pending_targets: list[tuple[Path, dict[str, object]]] = []
    n_skip = 0
    for rel, meta in targets:
        dest = files_root / rel
        url = _physionet_url(args.base_url, rel)
        if dest.exists():
            manifest["images"].append(
                {**meta, "relative_path": rel.as_posix(), "url": url, "status": "skipped_existing"}
            )
            n_skip += 1
            continue
        pending_targets.append((rel, meta))

    if not args.physionet_user or not args.physionet_password:
        if pending_targets:
            logger.error(
                "PhysioNet credentials missing. Set PHYSIONET_USER and PHYSIONET_PASSWORD, "
                "or pass --physionet-user / --physionet-password."
            )
            sys.exit(1)
        logger.info("All selected files already exist locally; no PhysioNet credentials needed.")

    if not pending_targets:
        manifest["download_summary"] = {
            "n_files_planned": len(targets),
            "downloaded_ok": 0,
            "skipped_existing": n_skip,
            "failed": 0,
        }
        args.output_root.mkdir(parents=True, exist_ok=True)
        manifest_path = args.output_root / "download_cohort_mimic_jpg_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        logger.info(
            "Done. downloaded_ok=0 skipped_existing=%d failed=0 manifest=%s",
            n_skip,
            manifest_path,
        )
        logger.info("Point run_inference --images-root at: %s", files_root)
        return

    opener = _build_opener()
    if not args.skip_access_probe:
        args.base_url = _verify_physionet_mimic_cxr_jpg_access(
            opener,
            args.base_url,
            args.physionet_user,
            args.physionet_password,
            user_agent=args.user_agent,
        )
        manifest["physionet_base_url"] = args.base_url

    n_ok = 0
    n_fail = 0

    for rel, meta in tqdm(pending_targets, desc="Downloading JPGs", unit="file"):
        url = _physionet_url(args.base_url, rel)
        dest = files_root / rel
        ok, status = _download_one(
            opener,
            url,
            dest,
            physionet_user=args.physionet_user,
            physionet_password=args.physionet_password,
            user_agent=args.user_agent,
        )
        row = {**meta, "relative_path": rel.as_posix(), "url": url, "status": status}
        manifest["images"].append(row)
        if status == "skipped_existing":
            n_skip += 1
        elif ok:
            n_ok += 1
        else:
            n_fail += 1
            logger.warning("Failed (%s): %s", status, url)

    manifest["download_summary"] = {
        "n_files_planned": len(targets),
        "downloaded_ok": n_ok,
        "skipped_existing": n_skip,
        "failed": n_fail,
    }

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "download_cohort_mimic_jpg_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    logger.info(
        "Done. downloaded_ok=%d skipped_existing=%d failed=%d manifest=%s",
        n_ok,
        n_skip,
        n_fail,
        manifest_path,
    )
    logger.info("Point run_inference --images-root at: %s", files_root)


if __name__ == "__main__":
    run()
