from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .constants import CERTAIN_WORDS, UNCERTAIN_WORDS
from .text_processing import split_sentences, tokenize_words


def count_hedges(text: str, hedge_patterns: list[re.Pattern[str]]) -> int:
    text_lower = text.lower()
    return sum(len(pattern.findall(text_lower)) for pattern in hedge_patterns)


def add_language_features(df: pd.DataFrame, hedge_phrases: list[str]) -> pd.DataFrame:
    hedge_patterns = [re.compile(rf"\b{re.escape(phrase)}\b", flags=re.IGNORECASE) for phrase in hedge_phrases]
    words = df["report_text"].map(tokenize_words)

    df["word_count"] = words.map(len)
    df["hedge_count"] = df["report_text"].map(lambda text: count_hedges(text, hedge_patterns))
    df["hedge_rate"] = np.where(df["word_count"] > 0, df["hedge_count"] / df["word_count"], np.nan)

    sentences = df["report_text"].map(split_sentences)
    df["mean_sent_length"] = [
        np.mean([len(tokenize_words(sentence)) for sentence in report_sentences]) if report_sentences else np.nan
        for report_sentences in sentences
    ]
    df["ttr"] = words.map(lambda tokens: len(set(tokens)) / len(tokens) if tokens else np.nan)

    def certainty_score(tokens: list[str]) -> float:
        if not tokens:
            return np.nan
        certain_count = sum(1 for token in tokens if token in CERTAIN_WORDS)
        uncertain_count = sum(1 for token in tokens if token in UNCERTAIN_WORDS)
        return (certain_count - uncertain_count) / len(tokens)

    df["certainty_score"] = words.map(certainty_score)
    return df
