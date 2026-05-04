"""Select VLM score thresholds from inference outputs using a held-out patient split.

Validation rows are preferred to choose a score cutoff; if that patient fold has
no proxy-positive and proxy-negative studies for a disease (small cohorts, strict
time windows), calibration falls back to all rows for that disease with a
warning. The chosen value is then frozen for application to the reporting split.

Study-level proxy labels (late vs early in the pre-diagnosis window) supply a
binary reference for sensitivity, specificity, F1, and Youden's J when every
patient is a case and cross-prompt negatives are unavailable.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

Criterion = Literal["f1", "youden", "f1_then_youden"]

DEFAULT_THRESHOLD_START = 0.1
DEFAULT_THRESHOLD_END = 0.9
DEFAULT_THRESHOLD_STEP = 0.05

# Studies within this many hours of diagnosis are treated as "late-window" positives.
DEFAULT_POSITIVE_WITHIN_HOURS = 48.0
# Studies at least this many hours before diagnosis are "early-window" negatives.
DEFAULT_NEGATIVE_AT_LEAST_HOURS = 288.0


def default_threshold_grid() -> np.ndarray:
    """Values from 0.1 through 0.9 inclusive in steps of 0.05."""
    return np.arange(
        DEFAULT_THRESHOLD_START,
        DEFAULT_THRESHOLD_END + DEFAULT_THRESHOLD_STEP / 2,
        DEFAULT_THRESHOLD_STEP,
    )


def assign_patient_validation_mask(
    df: pd.DataFrame,
    *,
    val_fraction: float = 0.2,
    random_state: int = 42,
    subject_col: str = "subject_id",
) -> pd.Series:
    """True for rows whose ``subject_id`` falls in the validation fold (patient-level)."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be in (0, 1)")
    # Copy: pandas may expose a read-only buffer; shuffle mutates in place.
    subjects = np.asarray(
        df[subject_col].drop_duplicates().to_numpy(), copy=True
    )
    if len(subjects) < 2:
        raise ValueError(
            "Need at least two distinct subject_id values for a patient-level holdout."
        )
    rng = np.random.default_rng(random_state)
    rng.shuffle(subjects)
    n_val = max(1, int(np.ceil(len(subjects) * val_fraction)))
    n_val = min(n_val, len(subjects) - 1)
    val_set = set(int(x) for x in subjects[:n_val])
    return df[subject_col].isin(val_set)


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """y_true, y_pred in {0,1}. Returns sens, spec, f1, youden (NaNs if undefined)."""
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    f1 = 2 * prec * sens / (prec + sens) if prec + sens > 0 else float("nan")
    youden = sens + spec - 1 if np.isfinite(sens) and np.isfinite(spec) else float("nan")
    return {
        "sensitivity": float(sens),
        "specificity": float(spec),
        "precision": float(prec),
        "f1": float(f1),
        "youden_j": float(youden),
        "tp": float(tp),
        "fn": float(fn),
        "fp": float(fp),
        "tn": float(tn),
    }


def apply_proxy_labels(
    df: pd.DataFrame,
    *,
    hours_col: str = "hours_before_diagnosis",
    positive_within_hours: float = DEFAULT_POSITIVE_WITHIN_HOURS,
    negative_at_least_hours: float = DEFAULT_NEGATIVE_AT_LEAST_HOURS,
) -> pd.Series:
    """1 = late-window proxy positive, 0 = early-window proxy negative, NaN = unused."""
    h = pd.to_numeric(df[hours_col], errors="coerce")
    label = pd.Series(np.nan, index=df.index, dtype="float64")
    label.loc[h <= positive_within_hours] = 1.0
    label.loc[h >= negative_at_least_hours] = 0.0
    return label


def sweep_thresholds_study_level(
    scores: np.ndarray,
    y_true: np.ndarray,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    """One row per threshold with confusion-derived metrics."""
    scores = np.asarray(scores, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    rows: list[dict[str, object]] = []
    for tau in thresholds:
        y_pred = (scores > float(tau)).astype(np.int64)
        m = _binary_metrics(y_true, y_pred)
        rows.append({"threshold": float(tau), **m})
    return pd.DataFrame(rows)


def pick_threshold(
    sweep: pd.DataFrame,
    criterion: Criterion,
) -> tuple[float, pd.DataFrame, int]:
    """Return best threshold, sorted sweep (by threshold), and chosen row position."""
    s = sweep.sort_values("threshold").reset_index(drop=True)
    if s.empty:
        raise ValueError("Empty threshold sweep")
    if criterion == "f1":
        best_idx = s["f1"].idxmax(skipna=True)
    elif criterion == "youden":
        best_idx = s["youden_j"].idxmax(skipna=True)
    else:
        max_f1 = s["f1"].max(skipna=True)
        if pd.isna(max_f1):
            best_idx = s["youden_j"].idxmax(skipna=True)
        else:
            tied = s[s["f1"] == max_f1]
            if len(tied) > 1:
                best_idx = tied["youden_j"].idxmax(skipna=True)
            else:
                best_idx = tied.index[0]
    best_pos = int(s.index.get_loc(best_idx))
    tau = round(float(s.iloc[best_pos]["threshold"]), 6)
    return tau, s, best_pos


def first_crossing_per_event(
    seq: pd.DataFrame,
    threshold: float,
    *,
    score_col: str = "disease_score",
    time_col: str = "study_datetime",
) -> dict[str, object]:
    """First study (chronological) with score > threshold; else censored."""
    g = seq.sort_values(time_col, kind="mergesort")
    hit = g[g[score_col] > threshold]
    if hit.empty:
        return {
            "detection_time": pd.NaT,
            "censored": True,
            "first_crossing_study_id": None,
            "max_score": float(g[score_col].max()),
        }
    row = hit.iloc[0]
    return {
        "detection_time": row[time_col],
        "censored": False,
        "first_crossing_study_id": int(row["study_id"]) if "study_id" in row.index else None,
        "max_score": float(g[score_col].max()),
    }


def _proxy_labeled_frame(
    pool: pd.DataFrame,
    *,
    positive_within_hours: float,
    negative_at_least_hours: float,
) -> tuple[pd.DataFrame, int, int]:
    """Attach proxy labels and return (labeled rows, n_pos, n_neg)."""
    p = pool.copy()
    p["proxy_label"] = apply_proxy_labels(
        p,
        positive_within_hours=positive_within_hours,
        negative_at_least_hours=negative_at_least_hours,
    )
    labeled = p.dropna(subset=["proxy_label"]).copy()
    if labeled.empty:
        return labeled, 0, 0
    labeled["proxy_label"] = labeled["proxy_label"].astype(int)
    n_pos = int((labeled["proxy_label"] == 1).sum())
    n_neg = int((labeled["proxy_label"] == 0).sum())
    return labeled, n_pos, n_neg


def calibration_for_disease(
    inference_df: pd.DataFrame,
    disease: str,
    val_mask: pd.Series,
    thresholds: np.ndarray,
    *,
    positive_within_hours: float,
    negative_at_least_hours: float,
    criterion: Criterion,
    allow_full_cohort_fallback: bool = True,
) -> dict[str, object]:
    """Run proxy-label threshold sweep for one ``disease_type`` (validation pool first)."""
    sub = inference_df[inference_df["disease_type"].astype(str) == disease].copy()
    if sub.empty:
        return {
            "disease_type": disease,
            "error": "no_rows",
            "threshold": None,
            "sweep": pd.DataFrame(),
        }

    vsub = sub[val_mask.reindex(sub.index).fillna(False)].copy()
    labeled, n_pos, n_neg = _proxy_labeled_frame(
        vsub,
        positive_within_hours=positive_within_hours,
        negative_at_least_hours=negative_at_least_hours,
    )
    calibration_pool: Literal["validation", "full_cohort"] = "validation"

    if len(labeled) == 0 or n_pos == 0 or n_neg == 0:
        if allow_full_cohort_fallback:
            fb_labeled, fb_pos, fb_neg = _proxy_labeled_frame(
                sub,
                positive_within_hours=positive_within_hours,
                negative_at_least_hours=negative_at_least_hours,
            )
            if len(fb_labeled) > 0 and fb_pos > 0 and fb_neg > 0:
                logger.warning(
                    "Disease %s: validation proxy labels unusable for sweep "
                    "(val study rows=%d labeled=%d n_pos=%d n_neg=%d); "
                    "using full-cohort fallback (labeled=%d n_pos=%d n_neg=%d).",
                    disease,
                    len(vsub),
                    len(labeled),
                    n_pos,
                    n_neg,
                    len(fb_labeled),
                    fb_pos,
                    fb_neg,
                )
                labeled, n_pos, n_neg = fb_labeled, fb_pos, fb_neg
                calibration_pool = "full_cohort"
            else:
                logger.warning(
                    "Disease %s: insufficient proxy labels (labeled=%d n_pos=%d n_neg=%d)",
                    disease,
                    len(fb_labeled),
                    fb_pos,
                    fb_neg,
                )
                return {
                    "disease_type": disease,
                    "error": "insufficient_proxy_labels",
                    "calibration_pool": calibration_pool,
                    "threshold": None,
                    "criterion": criterion,
                    "positive_within_hours": positive_within_hours,
                    "negative_at_least_hours": negative_at_least_hours,
                    "n_validation_study_rows": int(len(vsub)),
                    "n_validation_labeled_studies": int(len(fb_labeled)),
                    "n_proxy_positive": fb_pos,
                    "n_proxy_negative": fb_neg,
                    "best_row": None,
                    "sweep": pd.DataFrame(),
                }
        else:
            logger.warning(
                "Disease %s: insufficient proxy labels (labeled=%d n_pos=%d n_neg=%d)",
                disease,
                len(labeled),
                n_pos,
                n_neg,
            )
            return {
                "disease_type": disease,
                "error": "insufficient_proxy_labels",
                "calibration_pool": calibration_pool,
                "threshold": None,
                "criterion": criterion,
                "positive_within_hours": positive_within_hours,
                "negative_at_least_hours": negative_at_least_hours,
                "n_validation_study_rows": int(len(vsub)),
                "n_validation_labeled_studies": int(len(labeled)),
                "n_proxy_positive": n_pos,
                "n_proxy_negative": n_neg,
                "best_row": None,
                "sweep": pd.DataFrame(),
            }

    sweep = sweep_thresholds_study_level(
        labeled["disease_score"].to_numpy(),
        labeled["proxy_label"].to_numpy(),
        thresholds,
    )
    tau, sweep_sorted, best_pos = pick_threshold(sweep, criterion)
    best_row = sweep_sorted.iloc[best_pos].to_dict()

    return {
        "disease_type": disease,
        "threshold": tau,
        "calibration_pool": calibration_pool,
        "criterion": criterion,
        "positive_within_hours": positive_within_hours,
        "negative_at_least_hours": negative_at_least_hours,
        "n_validation_study_rows": int(len(vsub)),
        "n_validation_labeled_studies": int(len(labeled)),
        "n_proxy_positive": n_pos,
        "n_proxy_negative": n_neg,
        "best_row": best_row,
        "sweep": sweep_sorted,
    }


def build_detection_events_table(
    inference_df: pd.DataFrame,
    thresholds_by_disease: dict[str, float],
    apply_mask: pd.Series,
) -> pd.DataFrame:
    """Per (subject_id, hadm_id, disease_type): first crossing or censored."""
    sub = inference_df[apply_mask.reindex(inference_df.index).fillna(False)].copy()
    if sub.empty:
        return pd.DataFrame()

    grp_cols = ["subject_id", "hadm_id", "disease_type"]
    rows: list[dict[str, object]] = []
    for key, grp in sub.groupby(grp_cols, sort=False):
        sid, hid, dis = key
        dis_s = str(dis)
        tau = thresholds_by_disease.get(dis_s)
        if tau is None:
            continue
        dx = pd.to_datetime(grp["diagnosis_time"].iloc[0], errors="coerce")
        fc = first_crossing_per_event(grp, tau)
        det_t = fc["detection_time"]
        if fc["censored"] or pd.isna(det_t):
            lead_h = float("nan")
        else:
            lead_h = (dx - pd.Timestamp(det_t)).total_seconds() / 3600.0
        rows.append(
            {
                "subject_id": int(sid),
                "hadm_id": int(hid),
                "disease_type": dis_s,
                "diagnosis_time": dx,
                "detection_time": det_t,
                "hours_lead_time": float(lead_h) if np.isfinite(lead_h) else None,
                "censored": bool(fc["censored"]),
                "threshold": float(tau),
                "first_crossing_study_id": fc["first_crossing_study_id"],
                "max_disease_score": fc["max_score"],
            }
        )
    return pd.DataFrame(rows)


def run_detection_threshold_calibration(
    inference_df: pd.DataFrame,
    *,
    val_fraction: float = 0.2,
    random_state: int = 42,
    thresholds: np.ndarray | None = None,
    positive_within_hours: float = DEFAULT_POSITIVE_WITHIN_HOURS,
    negative_at_least_hours: float = DEFAULT_NEGATIVE_AT_LEAST_HOURS,
    criterion: Criterion = "f1_then_youden",
    allow_full_cohort_fallback: bool = True,
) -> dict[str, object]:
    """Split patients, calibrate per disease on validation proxy labels, return artifacts."""
    df = inference_df.copy()
    required = {"subject_id", "hadm_id", "disease_type", "disease_score", "study_datetime", "diagnosis_time"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"Inference table missing columns: {sorted(miss)}")

    df["study_datetime"] = pd.to_datetime(df["study_datetime"], errors="coerce")
    df["diagnosis_time"] = pd.to_datetime(df["diagnosis_time"], errors="coerce")
    df = df.dropna(subset=["study_datetime", "diagnosis_time", "disease_score"])

    if thresholds is None:
        thresholds = default_threshold_grid()

    val_mask = assign_patient_validation_mask(
        df, val_fraction=val_fraction, random_state=random_state
    )
    train_mask = ~val_mask

    diseases = sorted(df["disease_type"].astype(str).unique())
    per_disease: dict[str, object] = {}
    sweeps_by_disease: dict[str, pd.DataFrame] = {}
    thresholds_out: dict[str, float] = {}

    for d in diseases:
        out = calibration_for_disease(
            df,
            d,
            val_mask,
            thresholds,
            positive_within_hours=positive_within_hours,
            negative_at_least_hours=negative_at_least_hours,
            criterion=criterion,
            allow_full_cohort_fallback=allow_full_cohort_fallback,
        )
        sweeps_by_disease[d] = out.pop("sweep")  # type: ignore[assignment]
        per_disease[d] = out
        if out.get("threshold") is not None:
            thresholds_out[d] = float(out["threshold"])

    events_reporting = build_detection_events_table(df, thresholds_out, train_mask)
    events_validation = build_detection_events_table(df, thresholds_out, val_mask)

    n_subjects = int(df["subject_id"].nunique())
    n_val_subjects = int(df.loc[val_mask, "subject_id"].nunique())
    n_train_subjects = int(df.loc[train_mask, "subject_id"].nunique())

    return {
        "thresholds_by_disease": thresholds_out,
        "per_disease": per_disease,
        "sweeps_by_disease": sweeps_by_disease,
        "val_fraction": val_fraction,
        "random_state": random_state,
        "criterion": criterion,
        "n_unique_subjects": n_subjects,
        "n_validation_subjects": n_val_subjects,
        "n_reporting_subjects": n_train_subjects,
        "events_reporting": events_reporting,
        "events_validation": events_validation,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "inference_table": df,
        "thresholds_array": thresholds,
        "positive_within_hours": positive_within_hours,
        "negative_at_least_hours": negative_at_least_hours,
        "allow_full_cohort_fallback": allow_full_cohort_fallback,
    }
