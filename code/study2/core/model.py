from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Literal

import numpy as np

from .constants import HEART_FAILURE_SCORE_PROMPTS, SEPSIS_SCORE_PROMPTS

logger = logging.getLogger(__name__)


def cosine_similarity_to_unit_interval(sim: float) -> float:
    """Map BioViL-T cosine similarity (typically in [-1, 1]) to a bounded scalar in [0, 1].

    Used as a disease-aligned **score** (not a calibrated clinical probability).
    """
    return float(max(0.0, min(1.0, (float(sim) + 1.0) / 2.0)))


DiseaseScoreKind = Literal["heart_failure", "sepsis"]


class BioVilTInferenceEngine:
    """Wrapper around hi-ml-multimodal for BioViL-T image-text inference.

    Supports:
    - Single-image global embedding and text similarity.
    - Temporal (prior-conditioned) embedding via MultiImageEncoder (current + prior).

    Install:  pip install hi-ml-multimodal
    """

    def __init__(self, device: str = "cpu") -> None:
        try:
            import torch
            from health_multimodal.image import get_image_inference
            from health_multimodal.image.utils import ImageModelType
            from health_multimodal.text.utils import BertEncoderType, get_bert_inference
            from health_multimodal.vlp import ImageTextInferenceEngine
        except ImportError as exc:
            raise ImportError(
                "hi-ml-multimodal is required.\n"
                "Install: pip install hi-ml-multimodal"
            ) from exc

        self._torch = torch
        logger.info("Loading BioViL-T image encoder (downloads weights on first run)")
        image_engine = get_image_inference(ImageModelType.BIOVIL_T)
        logger.info("Loading BioViL-T BERT text encoder")
        text_engine = get_bert_inference(BertEncoderType.BIOVIL_T_BERT)

        self._engine = ImageTextInferenceEngine(
            image_inference_engine=image_engine,
            text_inference_engine=text_engine,
        )
        self._device = torch.device(device)
        self._engine.to(self._device)
        logger.info("BioViL-T inference engine ready on device: %s", device)

    def encode_image(
        self, image_path: Path, prior_path: Path | None = None
    ) -> np.ndarray:
        """Return an L2-normalised global embedding for a (optionally prior-conditioned) image.

        Parameters
        ----------
        image_path:
            Current chest X-ray JPG or DICOM.
        prior_path:
            Prior chest X-ray for temporal conditioning. When given and the
            backbone exposes ``previous_image``, both studies are encoded jointly
            (same path as hi-ml's ``MultiImageModel``).

        Returns
        -------
        np.ndarray of shape (128,), dtype float32, L2-normalised.
        """
        import torch.nn.functional as F

        img_engine = self._engine.image_inference_engine

        if prior_path is not None:
            current_t, _ = img_engine.load_and_transform_input_image(image_path, img_engine.transform)
            prior_t, _ = img_engine.load_and_transform_input_image(prior_path, img_engine.transform)
            model = img_engine.model
            enc = model.encoder
            with self._torch.no_grad():
                # BioViL-T uses MultiImageEncoder under plain ImageModel; ImageModel.forward
                # only accepts ``x``, so paired inputs must go through the encoder + projector.
                if "previous_image" in inspect.signature(enc.forward).parameters:
                    patch_x, pooled_x = enc(
                        current_t, prior_t, return_patch_embeddings=True
                    )
                    out = model.forward_post_encoder(patch_x, pooled_x)
                else:
                    logger.warning(
                        "prior_path was set but encoder has no previous_image; using current image only"
                    )
                    out = model.forward(current_t)
                emb = F.normalize(out.projected_global_embedding, dim=-1)[0]
        else:
            emb = img_engine.get_projected_global_embedding(image_path)

        return emb.cpu().float().numpy()

    def get_similarities(
        self,
        image_path: Path,
        texts: list[str],
        prior_path: Path | None = None,
    ) -> list[float]:
        """Return cosine similarity scores between the image and each text prompt.

        Parameters
        ----------
        image_path:
            Current chest X-ray.
        texts:
            Text prompts to score.
        prior_path:
            Optional prior image for temporal conditioning.

        Returns
        -------
        list[float] in the same order as ``texts``.
        """
        img_emb = self.encode_image(image_path, prior_path=prior_path)  # (D,) float32, L2-normed
        return self._similarities_from_embedding(img_emb, texts)

    def _similarities_from_embedding(
        self, img_emb_flat: np.ndarray, texts: list[str]
    ) -> list[float]:
        """Cosine similarities given an L2-normalised image embedding (1-D)."""
        text_engine = self._engine.text_inference_engine
        txt_embs = text_engine.get_embeddings_from_prompt(
            texts, normalize=True, verbose=False
        )
        txt_np = np.asarray(txt_embs.cpu().float().numpy(), dtype=np.float32)
        img = np.asarray(img_emb_flat, dtype=np.float32).ravel()
        sims = np.dot(img, txt_np.T)
        return [float(s) for s in np.ravel(sims)]

    def disease_score_prompts(self, disease_type: DiseaseScoreKind) -> tuple[str, ...]:
        if disease_type == "heart_failure":
            return HEART_FAILURE_SCORE_PROMPTS
        if disease_type == "sepsis":
            return SEPSIS_SCORE_PROMPTS
        raise ValueError(f"Unknown disease_type: {disease_type!r}")

    def compute_disease_score(
        self,
        image_path: Path,
        disease_type: DiseaseScoreKind,
        *,
        prior_path: Path | None = None,
    ) -> tuple[float, dict[str, float], np.ndarray]:
        """Disease-specific scalar in ``[0, 1]``, component scores, and the image embedding.

        Heart failure: single prompt (pulmonary edema).
        Sepsis: max of pneumonia and consolidation prompt scores.
        """
        prompts = list(self.disease_score_prompts(disease_type))
        emb = self.encode_image(image_path, prior_path=prior_path)
        sims = self._similarities_from_embedding(emb, prompts)
        vals = [cosine_similarity_to_unit_interval(s) for s in sims]
        if disease_type == "heart_failure":
            agg = float(vals[0]) if vals else 0.0
            detail = {"disease_score_pulmonary_edema": agg}
            return agg, detail, emb
        agg = float(max(vals)) if vals else 0.0
        detail = {
            "disease_score_pneumonia": vals[0] if len(vals) > 0 else 0.0,
            "disease_score_consolidation": vals[1] if len(vals) > 1 else 0.0,
        }
        return agg, detail, emb
