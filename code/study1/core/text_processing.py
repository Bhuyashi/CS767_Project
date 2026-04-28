from __future__ import annotations

import re

import pandas as pd


def tokenize_words(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s]


def parse_report_sections(report_text: str) -> str:
    findings_match = re.search(
        r"\bFINDINGS?\s*:\s*(.*?)(?=\n[A-Z][A-Z \-/]{2,}\s*:|\Z)",
        report_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    impression_match = re.search(
        r"\bIMPRESSION\s*:\s*(.*?)(?=\n[A-Z][A-Z \-/]{2,}\s*:|\Z)",
        report_text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    parts: list[str] = []
    if findings_match:
        parts.append(findings_match.group(1).strip())
    if impression_match:
        parts.append(impression_match.group(1).strip())

    if parts:
        return "\n".join(parts).strip()
    return report_text.strip()


def infer_time_granularity(study_dt: pd.Series) -> str:
    valid = study_dt.dropna()
    if valid.empty:
        return "unknown"

    minutes = valid.dt.minute
    seconds = valid.dt.second
    is_3h_grid = (
        valid.dt.hour.isin([0, 3, 6, 9, 12, 15, 18, 21]) & (minutes == 0) & (seconds == 0)
    )
    ratio_3h = is_3h_grid.mean()
    unique_minutes = minutes.nunique()

    if ratio_3h >= 0.95:
        return "coarse_3hour_bins"
    if unique_minutes > 1:
        return "minute_level_or_finer"
    return "coarse_non_3hour"


def assign_circadian_bin(dt: pd.Timestamp, mode: str) -> str:
    if pd.isna(dt):
        return "unknown"
    hour = int(dt.hour)
    if mode == "binary_day_night":
        return "day" if 6 <= hour < 18 else "night"
    if hour < 6:
        return "night"
    if hour < 12:
        return "morning"
    if hour < 18:
        return "afternoon"
    return "evening"
