from __future__ import annotations

from typing import Literal

BIOVIL_T_MODEL_ID = "microsoft/BioViL-T"

FRONTAL_VIEW_CODES: frozenset[str] = frozenset({"PA", "AP"})

# Phase 2 disease-specific VLM scores: cosine similarity is mapped to [0, 1] in ``model.py``.
# Heart failure: pulmonary edema (primary imaging correlate).
HEART_FAILURE_SCORE_PROMPTS: tuple[str, ...] = ("Pulmonary edema.",)
# Sepsis: pneumonia and consolidation; aggregate = max of per-prompt mapped scores.
SEPSIS_SCORE_PROMPTS: tuple[str, ...] = (
    "Pneumonia.",
    "Lung consolidation.",
)

CohortDiseaseType = Literal["heart_failure", "sepsis"]

DEFAULT_TEXT_PROMPTS: list[str] = [
    "No acute cardiopulmonary abnormality.",
    "Pleural effusion.",
    "Pulmonary edema.",
    "Cardiomegaly.",
    "Pneumonia.",
    "Atelectasis.",
    "Pneumothorax.",
    "Worsening findings compared to prior.",
    "Stable findings compared to prior.",
    "Improving findings compared to prior.",
    "No significant change compared to prior.",
]

# Safe column name for each prompt (index-aligned with DEFAULT_TEXT_PROMPTS).
DEFAULT_PROMPT_COLUMNS: list[str] = [
    "sim_no_acute_abnormality",
    "sim_pleural_effusion",
    "sim_pulmonary_edema",
    "sim_cardiomegaly",
    "sim_pneumonia",
    "sim_atelectasis",
    "sim_pneumothorax",
    "sim_worsening",
    "sim_stable",
    "sim_improving",
    "sim_no_change",
]
