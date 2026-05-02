from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .model import BioVilTInferenceEngine

logger = logging.getLogger(__name__)


def run_inference(
    pairs: pd.DataFrame,
    text_prompts: list[str],
    prompt_columns: list[str],
    use_prior: bool,
    max_pairs: int | None,
    device: str = "cpu",
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Run BioViL-T inference over a DataFrame of image pairs.

    Parameters
    ----------
    pairs:
        DataFrame produced by ``data_io.resolve_pair_image_paths``.
        Must have columns: subject_id, current_study_id, prior_study_id,
        current_image_path, prior_image_path (and metadata columns).
    text_prompts:
        Text strings to compute cosine similarity against.
    prompt_columns:
        Column names (index-aligned with text_prompts) used in the output CSV.
    use_prior:
        When True, condition each image encoding on its paired prior image.
        When False, run single-image inference (ignores prior_image_path).
    max_pairs:
        Hard cap on number of pairs to process; None means no cap.
    device:
        Torch device string for BioViL-T (e.g. ``"cuda"``, ``"cpu"``).

    Returns
    -------
    results_df:
        One row per processed pair. Metadata columns + one similarity column
        per prompt. Rows where inference failed are omitted.
    current_embeddings:
        float32 array of shape (N, D) — one row per result row.
    prior_embeddings:
        float32 array of shape (N, D) — prior image embeddings (or same as
        current_embeddings when use_prior=False).
    """
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
            # Prior embedding is always single-image (no further conditioning).
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
        for col, score in zip(prompt_columns, similarities):
            result_row[col] = round(score, 6)

        result_rows.append(result_row)
        current_embs.append(current_emb)
        prior_embs.append(prior_emb)
        n_ok += 1

    logger.info("Inference complete: ok=%d failed=%d", n_ok, n_failed)

    results_df = pd.DataFrame(result_rows)

    if current_embs:
        current_arr = np.stack(current_embs).astype(np.float32)
        prior_arr = np.stack(prior_embs).astype(np.float32)
    else:
        current_arr = np.empty((0,), dtype=np.float32)
        prior_arr = np.empty((0,), dtype=np.float32)

    return results_df, current_arr, prior_arr
