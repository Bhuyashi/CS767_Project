from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .model import BioVilTInferenceEngine

logger = logging.getLogger(__name__)


def run_pairwise_inference(
    pairs: pd.DataFrame,
    text_prompts: list[str],
    prompt_columns: list[str],
    use_prior: bool,
    max_pairs: int | None,
    device: str = "cpu",
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Legacy: BioViL-T over consecutive (prior, current) study pairs with generic text prompts."""
    if max_pairs is not None:
        pairs = pairs.head(max_pairs)
        logger.info("max_pairs cap applied: processing %d pairs", len(pairs))

    engine = BioVilTInferenceEngine(device=device)

    result_rows: list[dict[str, object]] = []
    current_embs: list[np.ndarray] = []
    prior_embs: list[np.ndarray] = []

    n_total = len(pairs)
    n_ok = 0
    n_failed = 0

    for idx, row in pairs.iterrows():
        if (idx + 1) % 100 == 0 or (idx + 1) == n_total:
            logger.info("Inference progress: %d / %d  (ok=%d failed=%d)", idx + 1, n_total, n_ok, n_failed)

        current_path = Path(row["current_image_path"])
        prior_path = Path(row["prior_image_path"]) if use_prior else None

        try:
            current_emb = engine.encode_image(current_path, prior_path=prior_path)
            prior_emb = engine.encode_image(Path(row["prior_image_path"])) if use_prior else current_emb
            similarities = engine.get_similarities(current_path, text_prompts, prior_path=prior_path)
        except Exception as exc:
            logger.warning(
                "Inference failed for current_study_id=%s prior_study_id=%s: %s",
                row["current_study_id"], row["prior_study_id"], exc,
            )
            n_failed += 1
            continue

        result_row: dict[str, object] = {
            "subject_id": row["subject_id"],
            "current_study_id": row["current_study_id"],
            "prior_study_id": row["prior_study_id"],
            "current_datetime": row["current_datetime"],
            "prior_datetime": row["prior_datetime"],
            "days_between": row["days_between"],
            "current_view": row["current_view"],
            "prior_view": row["prior_view"],
            "current_image_path": row["current_image_path"],
            "prior_image_path": row["prior_image_path"],
            "use_prior_conditioning": use_prior,
        }
        for optional_col in ("hadm_id", "disease_type", "diagnosis_time", "hours_before_diagnosis"):
            if optional_col in row.index:
                result_row[optional_col] = row[optional_col]
        for col, score in zip(prompt_columns, similarities):
            result_row[col] = round(score, 6)

        result_rows.append(result_row)
        current_embs.append(current_emb)
        prior_embs.append(prior_emb)
        n_ok += 1

    logger.info("Pairwise inference complete: ok=%d failed=%d", n_ok, n_failed)

    results_df = pd.DataFrame(result_rows)

    if current_embs:
        current_arr = np.stack(current_embs).astype(np.float32)
        prior_arr = np.stack(prior_embs).astype(np.float32)
    else:
        current_arr = np.empty((0,), dtype=np.float32)
        prior_arr = np.empty((0,), dtype=np.float32)

    return results_df, current_arr, prior_arr


def run_sequence_inference(
    sequence_table: pd.DataFrame,
    *,
    device: str = "cpu",
    max_patients: int | None = None,
    disease_process_order: tuple[str, ...] = ("heart_failure", "sepsis"),
    use_prior: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, object]]:
    """Phase 2: ordered studies in the pre-diagnosis window, one disease-specific score per study.

    For each cohort event (subject_id, hadm_id, disease_type), studies are processed in
    ``seq_index`` order. When ``use_prior`` is True, the immediately prior study in that
    window conditions BioViL-T (two-frame temporal encoder); the first study uses single-image mode.

    Processing order across events defaults to all heart-failure admissions first, then sepsis
    (``disease_process_order``).

    Returns
    -------
    results_df
        Columns include ``subject_id``, ``hadm_id``, ``study_id``, ``study_datetime``,
        ``disease_type``, ``disease_score`` (``[0,1]``), ``hours_before_diagnosis``, and
        disease-specific sub-scores.
    embeddings
        ``float32`` array of shape ``(N, D)`` — current-study embeddings (one row per output row).
    qc
        Counts and configuration for JSON logging.
    """
    required = {
        "subject_id",
        "hadm_id",
        "disease_type",
        "study_id",
        "study_datetime",
        "diagnosis_time",
        "image_path",
        "seq_index",
        "hours_before_diagnosis",
    }
    miss = required - set(sequence_table.columns)
    if miss:
        raise ValueError(f"sequence_table missing columns: {sorted(miss)}")

    grp_cols = ["subject_id", "hadm_id", "disease_type"]
    rank = {d: i for i, d in enumerate(disease_process_order)}
    keys = sequence_table[grp_cols].drop_duplicates()
    keys["_proc_rank"] = keys["disease_type"].map(lambda x: rank.get(str(x), 99))
    keys = keys.sort_values(["_proc_rank", "subject_id", "hadm_id"]).drop(columns="_proc_rank")
    if max_patients is not None:
        keys = keys.head(int(max_patients))
    keys = keys.reset_index(drop=True)

    engine = BioVilTInferenceEngine(device=device)

    result_rows: list[dict[str, object]] = []
    embs: list[np.ndarray] = []
    n_ok = n_skip_bad_disease = n_failed = 0

    n_events = len(keys)
    for ev_num in range(n_events):
        keyrow = keys.iloc[ev_num]
        sid = int(keyrow["subject_id"])
        hid = int(keyrow["hadm_id"])
        dis_raw = str(keyrow["disease_type"])
        if dis_raw not in ("heart_failure", "sepsis"):
            logger.warning("Skipping unknown disease_type=%r for subject_id=%s hadm_id=%s", dis_raw, sid, hid)
            n_skip_bad_disease += 1
            continue

        grp = sequence_table[
            (sequence_table["subject_id"] == sid)
            & (sequence_table["hadm_id"] == hid)
            & (sequence_table["disease_type"].astype(str) == dis_raw)
        ].sort_values("seq_index")

        prev_image: Path | None = None
        for _, row in grp.iterrows():
            cur = Path(str(row["image_path"]))
            prior = prev_image if use_prior and prev_image is not None else None
            try:
                score, detail, emb = engine.compute_disease_score(
                    cur, dis_raw, prior_path=prior  # type: ignore[arg-type]
                )
            except Exception as exc:
                logger.warning(
                    "Sequence inference failed subject_id=%s study_id=%s: %s",
                    sid,
                    row.get("study_id"),
                    exc,
                )
                n_failed += 1
                prev_image = cur
                continue

            out: dict[str, object] = {
                "subject_id": sid,
                "hadm_id": hid,
                "disease_type": dis_raw,
                "study_id": int(row["study_id"]),
                "study_datetime": row["study_datetime"],
                "diagnosis_time": row["diagnosis_time"],
                "seq_index": int(row["seq_index"]),
                "hours_before_diagnosis": float(row["hours_before_diagnosis"]),
                "disease_score": round(float(score), 6),
                "use_prior_conditioning": bool(prior is not None),
            }
            if "ViewPosition" in row.index and pd.notna(row["ViewPosition"]):
                out["ViewPosition"] = str(row["ViewPosition"])
            for k, v in detail.items():
                out[k] = round(float(v), 6) if isinstance(v, (float, int)) else v

            result_rows.append(out)
            embs.append(emb)
            n_ok += 1
            prev_image = cur

        if (ev_num + 1) % 50 == 0 or ev_num + 1 == n_events:
            logger.info(
                "Sequence events: %d / %d  (rows_ok=%d failed_studies=%d)",
                ev_num + 1,
                n_events,
                n_ok,
                n_failed,
            )

    results_df = pd.DataFrame(result_rows)
    if embs:
        emb_arr = np.stack(embs).astype(np.float32)
    else:
        emb_arr = np.empty((0,), dtype=np.float32)

    qc: dict[str, object] = {
        "n_cohort_events_scheduled": int(n_events),
        "n_output_rows": int(len(results_df)),
        "n_study_failures": int(n_failed),
        "n_events_skipped_unknown_disease": int(n_skip_bad_disease),
        "use_prior_conditioning": use_prior,
        "disease_process_order": list(disease_process_order),
        "embedding_dim": int(emb_arr.shape[1]) if emb_arr.ndim == 2 else None,
    }
    logger.info(
        "Sequence inference complete: rows=%d study_failures=%d",
        len(results_df),
        n_failed,
    )
    return results_df, emb_arr, qc
