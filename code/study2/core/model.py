from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _to_numpy(embedding_output) -> np.ndarray:
    """Convert hi-ml embedding return (dataclass or raw tensor) to a float32 numpy array."""
    if hasattr(embedding_output, "projected_global_embedding"):
        tensor = embedding_output.projected_global_embedding
    else:
        tensor = embedding_output
    return tensor.detach().cpu().float().numpy()


class BioVilTInferenceEngine:
    """Thin wrapper around hi-ml-multimodal ImageTextInferenceEngine for BioViL-T.

    Handles:
    - Single-image and temporal (prior + current) image encoding.
    - Cosine similarity scoring against a list of text prompts.

    Install dependency:
        pip install hi-ml-multimodal
    """

    def __init__(self) -> None:
        try:
            from health_multimodal.image.model.pretrained import get_biovil_t_image_encoder
            from health_multimodal.text.utils import get_bert_inference
            from health_multimodal.vlp import ImageTextInferenceEngine
        except ImportError as exc:
            raise ImportError(
                "hi-ml-multimodal is required for BioViL-T inference.\n"
                "Install it with: pip install hi-ml-multimodal"
            ) from exc

        logger.info("Loading BioViL-T image encoder (downloads weights on first run)")
        image_engine = get_biovil_t_image_encoder()
        logger.info("Loading BERT text encoder")
        text_engine = get_bert_inference()
        self._engine = ImageTextInferenceEngine(
            image_inference_engine=image_engine,
            text_inference_engine=text_engine,
        )
        logger.info("BioViL-T inference engine ready")

    def encode_image(
        self, image_path: Path, prior_path: Path | None = None
    ) -> np.ndarray:
        """Return a projected global embedding vector for an image (or image pair).

        Parameters
        ----------
        image_path:
            Path to the current (or only) chest X-ray JPG.
        prior_path:
            Optional path to the prior chest X-ray JPG. When provided, BioViL-T
            produces a temporally-conditioned representation.

        Returns
        -------
        np.ndarray of shape (D,) with dtype float32.
        """
        raw = self._engine.get_image_embeddings_from_raw_data(
            image_path=image_path,
            image_path_for_prior=prior_path,
        )
        return _to_numpy(raw)

    def get_similarities(
        self,
        image_path: Path,
        texts: list[str],
        prior_path: Path | None = None,
    ) -> list[float]:
        """Return cosine similarity scores between the image (pair) and each text prompt.

        Parameters
        ----------
        image_path:
            Path to the current chest X-ray JPG.
        texts:
            List of text prompts to score against.
        prior_path:
            Optional prior image for temporal conditioning.

        Returns
        -------
        list[float] with one score per text, in the same order as ``texts``.
        """
        return [
            float(
                self._engine.get_similarity_score_from_raw_data(
                    image_path=image_path,
                    query_text=text,
                    image_path_for_prior=prior_path,
                )
            )
            for text in texts
        ]
