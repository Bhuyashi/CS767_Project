from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def add_radiologist_proxy_cluster(df: pd.DataFrame, k_clusters: int, random_state: int) -> pd.DataFrame:
    if df.empty:
        df["radiologist_cluster"] = []
        return df

    n_samples = len(df)
    k = max(2, min(k_clusters, n_samples))
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=5 if n_samples >= 1000 else 1,
        max_df=0.95,
        max_features=5000,
    )
    tfidf = vectorizer.fit_transform(df["report_text"].fillna(""))
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    df["radiologist_cluster"] = kmeans.fit_predict(tfidf).astype(int)
    return df
