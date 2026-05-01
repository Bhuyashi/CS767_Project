from __future__ import annotations

BIOVIL_T_MODEL_ID = "microsoft/BioViL-T"

FRONTAL_VIEW_CODES: frozenset[str] = frozenset({"PA", "AP"})

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
