from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class BioVilTInferenceEngine:
    """Wrapper around hi-ml-multimodal for BioViL-T image-text inference.

    Supports:
    - Single-image global embedding and text similarity.
    - Temporal (prior-conditioned) embedding via MultiImageModel.forward().

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
            Prior chest X-ray for temporal conditioning. When given, the
            MultiImageModel encoder processes both images jointly.

        Returns
        -------
        np.ndarray of shape (128,), dtype float32, L2-normalised.
        """
        import torch.nn.functional as F

        img_engine = self._engine.image_inference_engine

        if prior_path is not None:
            current_t, _ = img_engine.load_and_transform_input_image(image_path, img_engine.transform)
            prior_t, _ = img_engine.load_and_transform_input_image(prior_path, img_engine.transform)
            with self._torch.no_grad():
                out = img_engine.model.forward(current_t, previous_image=prior_t)
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

        text_engine = self._engine.text_inference_engine
        # Batch all prompts in one forward pass
        txt_embs = text_engine.get_embeddings_from_prompt(
            texts, normalize=True, verbose=False
        )  # (N, D), L2-normed
        txt_np = txt_embs.cpu().float().numpy()

        scores = list(img_emb @ txt_np.T)  # (N,)
        return [float(s) for s in scores]
