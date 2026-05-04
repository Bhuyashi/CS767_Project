"""Lead-time outcome summaries: survival-style timing, Cox model, and time-sliced AUC.

Builds a patient-level table from locked detection events (reporting split) and
inference rows, optionally joins SOFA from MIMIC-IV ``derived/sofa.csv``, then runs
Kaplan–Meier, log-rank, Cox proportional hazards, and score-vs-time AUC curves.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .detection_calibration import (
    DEFAULT_NEGATIVE_AT_LEAST_HOURS,
    DEFAULT_POSITIVE_WITHIN_HOURS,
    apply_proxy_labels,
)

logger = logging.getLogger(__name__)

# Hours before diagnosis for AUC landmarks (spans 14 d so scores can change; 3–24 h alone often ties to one study).
DEFAULT_ANCHOR_HOURS_BEFORE_DX: tuple[float, ...] = (
    336,
    312,
    288,
    264,
    240,
    216,
    192,
    168,
    144,
    120,
    96,
    72,
    48,
    36,
    24,
    18,
    12,
    8,
    6,
    3,
)


def load_sofa_per_admission(sofa_csv: Path | None) -> pd.DataFrame:
    """Return ``subject_id``, ``hadm_id``, ``sofa_score`` (max SOFA over stay rows)."""
    if sofa_csv is None or not Path(sofa_csv).exists():
        if sofa_csv is not None:
            logger.warning("SOFA file not found (%s); covariate will be missing.", sofa_csv)
        return pd.DataFrame(columns=["subject_id", "hadm_id", "sofa_score"])

    df = pd.read_csv(sofa_csv)
    cols = {c.lower(): c for c in df.columns}
    sid_c = cols.get("subject_id")
    hid_c = cols.get("hadm_id")
    sofa_c = cols.get("sofa") or cols.get("sofa_score")
    if not sid_c or not hid_c or not sofa_c:
        raise ValueError(
            f"SOFA CSV must include subject_id, hadm_id, and sofa (or sofa_score); got columns {list(df.columns)}"
        )
    sub = df[[sid_c, hid_c, sofa_c]].copy()
    sub.columns = ["subject_id", "hadm_id", "sofa_raw"]
    sub["subject_id"] = pd.to_numeric(sub["subject_id"], errors="coerce").astype("Int64")
    sub["hadm_id"] = pd.to_numeric(sub["hadm_id"], errors="coerce").astype("Int64")
    sub["sofa_score"] = pd.to_numeric(sub["sofa_raw"], errors="coerce")
    sub = sub.dropna(subset=["subject_id", "hadm_id"])
    sub["subject_id"] = sub["subject_id"].astype(np.int64)
    sub["hadm_id"] = sub["hadm_id"].astype(np.int64)
    agg = (
        sub.groupby(["subject_id", "hadm_id"], as_index=False)["sofa_score"]
        .max()
        .rename(columns={"sofa_score": "sofa_score"})
    )
    logger.info("SOFA aggregated to %d (subject_id, hadm_id) rows", len(agg))
    return agg


def _inference_aggregates(inference_df: pd.DataFrame, *, window_days: float) -> pd.DataFrame:
    """Per cohort event: first study time, study count, imaging frequency."""
    req = {"subject_id", "hadm_id", "disease_type", "study_id", "study_datetime"}
    miss = req - set(inference_df.columns)
    if miss:
        raise ValueError(f"Inference table missing columns: {sorted(miss)}")

    gcols = ["subject_id", "hadm_id", "disease_type"]
    x = inference_df.copy()
    x["study_datetime"] = pd.to_datetime(x["study_datetime"], errors="coerce")
    x = x.dropna(subset=["study_datetime"])
    grp = x.groupby(gcols, sort=False)
    out = grp.agg(
        first_study_datetime=("study_datetime", "min"),
        n_studies_in_window=("study_id", "nunique"),
    ).reset_index()
    out["imaging_frequency"] = out["n_studies_in_window"].astype(float) / float(window_days)
    return out


def build_survival_outcomes_table(
    events_df: pd.DataFrame,
    inference_df: pd.DataFrame,
    *,
    sofa_per_adm: pd.DataFrame | None = None,
    window_days: float = 14.0,
) -> pd.DataFrame:
    """Patient-level table for lifelines (manuscript-aligned timing).

    ``time_hours`` is hours from the **pre-diagnosis window start**
    (``diagnosis_time - window_days``) to first VLM detection, or to ``diagnosis_time``
    if censored. This is the clock used by Kaplan–Meier and Cox: it stays positive
    whenever detection occurs after window open, even when the first in-window chest
    radiograph is the crossing study (where "first CXR → detection" would be 0 and
    degenerates lifelines plots).

    ``time_hours_from_first_cxr`` is the manuscript Phase 4 Step 1 duration: hours from
    the earliest in-window study to detection or to ``diagnosis_time`` if censored.

    ``time_hours_from_window_start_to_detection_or_dx`` duplicates ``time_hours`` for
    explicit naming in exports.

    ``hours_before_diagnosis_at_first_cxr`` is residual time from first in-window study
    to diagnosis.

    ``event`` is 1 when ``censored`` is false in the events table.
    """
    req_e = {
        "subject_id",
        "hadm_id",
        "disease_type",
        "diagnosis_time",
        "detection_time",
        "censored",
    }
    miss = req_e - set(events_df.columns)
    if miss:
        raise ValueError(f"Events table missing columns: {sorted(miss)}")

    ev = events_df.copy()
    ev["diagnosis_time"] = pd.to_datetime(ev["diagnosis_time"], errors="coerce")
    ev["detection_time"] = pd.to_datetime(ev["detection_time"], errors="coerce")
    ev["disease_type"] = ev["disease_type"].astype(str)
    ev = ev.dropna(subset=["diagnosis_time"])

    agg = _inference_aggregates(inference_df, window_days=window_days)
    merged = ev.merge(agg, on=["subject_id", "hadm_id", "disease_type"], how="left")

    miss_study = merged["first_study_datetime"].isna()
    if miss_study.any():
        logger.warning("Dropping %d events with no matching inference rows.", int(miss_study.sum()))
        merged = merged.loc[~miss_study].copy()

    merged["first_study_datetime"] = pd.to_datetime(merged["first_study_datetime"], errors="coerce")
    merged["event"] = (~merged["censored"].astype(bool)).astype(np.int64)

    win_delta = pd.to_timedelta(float(window_days), unit="D")
    merged["window_start"] = merged["diagnosis_time"] - win_delta

    det_ok = merged["event"].astype(bool) & merged["detection_time"].notna()
    span_to_dx = (merged["diagnosis_time"] - merged["first_study_datetime"]).dt.total_seconds() / 3600.0
    time_from_first = span_to_dx.copy()
    time_from_first.loc[det_ok] = (
        merged.loc[det_ok, "detection_time"] - merged.loc[det_ok, "first_study_datetime"]
    ).dt.total_seconds() / 3600.0
    merged["time_hours_from_first_cxr"] = time_from_first.astype(float)

    span_win_to_dx = (merged["diagnosis_time"] - merged["window_start"]).dt.total_seconds() / 3600.0
    time_from_win = span_win_to_dx.copy()
    time_from_win.loc[det_ok] = (
        merged.loc[det_ok, "detection_time"] - merged.loc[det_ok, "window_start"]
    ).dt.total_seconds() / 3600.0
    merged["time_hours_from_window_start_to_detection_or_dx"] = time_from_win.astype(float)

    merged["hours_before_diagnosis_at_first_cxr"] = span_to_dx.astype(float)

    merged["time_hours"] = merged["time_hours_from_window_start_to_detection_or_dx"].astype(
        float
    )

    bad = merged["time_hours"] < 0
    if bad.any():
        logger.warning("Clipping %d rows with negative time_hours to 0.", int(bad.sum()))
        merged.loc[bad, "time_hours"] = 0.0

    merged = merged.drop(columns=["window_start"])

    lead_h = (merged["diagnosis_time"] - merged["detection_time"]).dt.total_seconds() / 3600.0
    merged["hours_lead_time"] = np.nan
    merged.loc[det_ok, "hours_lead_time"] = lead_h.loc[det_ok].astype(float)

    if sofa_per_adm is not None and not sofa_per_adm.empty:
        merged = merged.merge(
            sofa_per_adm,
            on=["subject_id", "hadm_id"],
            how="left",
        )
    else:
        merged["sofa_score"] = np.nan

    merged["disease_sepsis"] = (merged["disease_type"].astype(str) == "sepsis").astype(np.int64)
    return merged.reset_index(drop=True)


def cohort_characteristics(inference_df: pd.DataFrame) -> dict[str, Any]:
    """Counts for Table 1-style summaries (patients, studies, median studies per patient)."""
    x = inference_df.copy()
    x["disease_type"] = x["disease_type"].astype(str)
    keys = ["subject_id", "hadm_id", "disease_type"]
    per_event = x.groupby(keys, sort=False)["study_id"].nunique().reset_index(name="n_studies")
    rows: list[dict[str, Any]] = []
    for d, g in per_event.groupby("disease_type"):
        rows.append(
            {
                "disease_type": str(d),
                "n_patients": int(len(g)),
                "n_studies_total": int(g["n_studies"].sum()),
                "median_studies_per_patient": float(np.median(g["n_studies"].to_numpy())),
            }
        )
    return {"by_disease": rows, "n_study_rows_inference": int(len(x))}


def _pick_landmark_row(grp: pd.DataFrame, anchor_h: float) -> pd.Series:
    """Study row whose score is used at ``anchor_h`` hours before diagnosis.

    Pool: studies with ``hours_before_diagnosis >= anchor_h`` (imaging at least
    ``anchor_h`` hours before diagnosis). From that pool choose the study whose
    ``hours_before_diagnosis`` is **closest** to ``anchor_h`` (minimize ``|hbfd - h|``).

    If no study satisfies ``>= anchor_h``, use the **earliest** study in the window
    (maximum ``hours_before_diagnosis``).
    """
    h = float(anchor_h)
    eligible = grp[grp["hours_before_diagnosis"] >= h - 1e-9].copy()
    if not eligible.empty:
        dist = (eligible["hours_before_diagnosis"] - h).abs()
        eligible = eligible.assign(_landmark_dist=dist)
        sort_cols = ["_landmark_dist", "hours_before_diagnosis"]
        if "study_datetime" in eligible.columns:
            sort_cols.append("study_datetime")
        if "study_id" in eligible.columns:
            sort_cols.append("study_id")
        return eligible.sort_values(sort_cols, ascending=True, kind="mergesort").iloc[0]
    fb = grp.copy()
    sc = ["hours_before_diagnosis"]
    if "study_datetime" in fb.columns:
        sc.append("study_datetime")
    if "study_id" in fb.columns:
        sc.append("study_id")
    return fb.sort_values(sc, ascending=[False] + [True] * (len(sc) - 1), kind="mergesort").iloc[0]


def _disease_score_at_landmark(grp: pd.DataFrame, anchor_h: float) -> float:
    return float(_pick_landmark_row(grp, anchor_h)["disease_score"])


def _auc_at_anchor(
    inference_df: pd.DataFrame,
    outcome_by_event: pd.DataFrame,
    anchor_h: float,
    disease: str | None,
    *,
    y_col: str = "y",
    min_n: int = 4,
) -> float:
    """AUC of ``y_col`` (0/1) from the landmark ``disease_score`` at ``anchor_h`` hours before dx."""
    from sklearn.metrics import roc_auc_score

    inf = inference_df.copy()
    if disease is not None:
        inf = inf[inf["disease_type"].astype(str) == disease]
    if inf.empty:
        return float("nan")

    inf["hours_before_diagnosis"] = pd.to_numeric(inf["hours_before_diagnosis"], errors="coerce")
    inf["disease_score"] = pd.to_numeric(inf["disease_score"], errors="coerce")
    inf["study_datetime"] = pd.to_datetime(inf["study_datetime"], errors="coerce")
    inf = inf.dropna(subset=["hours_before_diagnosis", "disease_score"])
    keys = ["subject_id", "hadm_id", "disease_type"]
    rows: list[dict[str, object]] = []
    for key, grp in inf.groupby(keys, sort=False):
        sid, hid, dis = key
        score = _disease_score_at_landmark(grp, float(anchor_h))
        rows.append(
            {
                "subject_id": int(sid),
                "hadm_id": int(hid),
                "disease_type": str(dis),
                "score": score,
            }
        )
    scores = pd.DataFrame(rows)
    need = keys + [y_col]
    miss = set(need) - set(outcome_by_event.columns)
    if miss:
        raise ValueError(f"outcome_by_event missing columns: {sorted(miss)}")
    m = scores.merge(outcome_by_event[need], on=keys, how="inner")
    if len(m) < int(min_n):
        return float("nan")
    y = m[y_col].astype(int).to_numpy()
    s = m["score"].to_numpy()
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def _auc_at_anchor_proxy(
    inference_df: pd.DataFrame,
    anchor_h: float,
    disease: str | None,
    *,
    positive_within_hours: float,
    negative_at_least_hours: float,
    min_n: int = 4,
) -> float:
    """ROC AUC of landmark score vs late-window (1) vs early-window (0) proxy on the landmark study."""
    from sklearn.metrics import roc_auc_score

    inf = inference_df.copy()
    if disease is not None:
        inf = inf[inf["disease_type"].astype(str) == disease]
    if inf.empty:
        return float("nan")

    inf["hours_before_diagnosis"] = pd.to_numeric(inf["hours_before_diagnosis"], errors="coerce")
    inf["disease_score"] = pd.to_numeric(inf["disease_score"], errors="coerce")
    inf = inf.dropna(subset=["hours_before_diagnosis", "disease_score"])
    keys = ["subject_id", "hadm_id", "disease_type"]
    scores: list[float] = []
    labels: list[int] = []
    for _, grp in inf.groupby(keys, sort=False):
        row = _pick_landmark_row(grp, float(anchor_h))
        one = pd.DataFrame([row])
        if "hours_before_diagnosis" not in one.columns:
            continue
        pl = apply_proxy_labels(
            one,
            positive_within_hours=positive_within_hours,
            negative_at_least_hours=negative_at_least_hours,
        )
        if pd.isna(pl.iloc[0]):
            continue
        scores.append(float(row["disease_score"]))
        labels.append(int(pl.iloc[0]))

    if len(scores) < int(min_n) or len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def auc_vs_hours_before_diagnosis(
    inference_df: pd.DataFrame,
    survival_outcomes: pd.DataFrame,
    *,
    anchor_hours: list[float],
    positive_within_hours: float = DEFAULT_POSITIVE_WITHIN_HOURS,
    negative_at_least_hours: float = DEFAULT_NEGATIVE_AT_LEAST_HOURS,
) -> pd.DataFrame:
    """One AUC per (disease_type, anchor_hour) using landmark ``disease_score``.

    For each anchor *h*, the score uses studies with ``hours_before_diagnosis >= h``,
    choosing the study whose timing is closest to *h* (see ``_pick_landmark_row``).

    **Primary (manuscript Phase 4, Step 5):** ROC AUC vs **imaging timing proxy** on the
    landmark study — late window (≤ ``positive_within_hours`` before diagnosis) vs early
    (≥ ``negative_at_least_hours``), aligned with threshold-calibration proxy semantics.

    If too few patients have a labeled landmark study at that anchor, falls back to
    patient-level ``event`` (detected vs censored), then to long-vs-short lead-time
    splits as in the legacy path.
    """
    keys = ["subject_id", "hadm_id", "disease_type"]
    det = survival_outcomes[keys + ["event", "hours_lead_time"]].copy()
    diseases = sorted(det["disease_type"].astype(str).unique())
    out_rows: list[dict[str, object]] = []
    for d in diseases:
        sub = det[det["disease_type"].astype(str) == d].copy()
        y_col = "y"
        evt = sub["event"].astype(int)
        if evt.nunique() >= 2:
            outcome = sub[keys].copy()
            outcome[y_col] = evt
            auc_mode = "vlm_detected_before_dx"
        else:
            hl = pd.to_numeric(sub["hours_lead_time"], errors="coerce")
            med = float(np.nanmedian(hl.to_numpy())) if hl.notna().any() else float("nan")
            outcome = sub[keys].copy()
            if np.isfinite(med):
                outcome[y_col] = (hl >= med).astype(int)
                auc_mode = "long_lead_ge_median_hours"
            else:
                outcome[y_col] = 0
                auc_mode = "long_lead_ge_median_hours"
            if outcome[y_col].nunique() < 2 and hl.notna().any():
                mu = float(np.nanmean(hl.to_numpy()))
                outcome[y_col] = (hl >= mu).astype(int)
                auc_mode = "long_lead_ge_mean_hours"
            if outcome[y_col].nunique() < 2:
                logger.warning(
                    "Disease %s: cannot form a binary AUC outcome (constant y); AUC will be NaN.",
                    d,
                )
                auc_mode = "auc_unusable_constant_y"
            else:
                logger.info(
                    "Disease %s: AUC uses outcome '%s' (all VLM-detected in this split).",
                    d,
                    auc_mode,
                )

        for h in anchor_hours:
            auc_proxy = _auc_at_anchor_proxy(
                inference_df,
                float(h),
                d,
                positive_within_hours=positive_within_hours,
                negative_at_least_hours=negative_at_least_hours,
            )
            if np.isfinite(auc_proxy):
                out_rows.append(
                    {
                        "disease_type": d,
                        "hours_before_diagnosis": float(h),
                        "roc_auc": auc_proxy,
                        "auc_outcome": "landmark_study_proxy_late_vs_early_window",
                    }
                )
                continue

            auc = _auc_at_anchor(inference_df, outcome, float(h), disease=d, y_col=y_col)
            out_rows.append(
                {
                    "disease_type": d,
                    "hours_before_diagnosis": float(h),
                    "roc_auc": auc,
                    "auc_outcome": auc_mode,
                }
            )
    return pd.DataFrame(out_rows)


def fit_kaplan_meier_and_logrank(survival_df: pd.DataFrame) -> dict[str, Any]:
    """KM / log-rank on ``time_hours`` (window start → detection or censor at diagnosis)."""
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test

    req = {"time_hours", "event", "disease_type"}
    miss = req - set(survival_df.columns)
    if miss:
        raise ValueError(f"Survival table missing columns: {sorted(miss)}")

    df = survival_df.dropna(subset=["time_hours", "event"]).copy()
    df["event"] = df["event"].astype(int)
    curves: dict[str, Any] = {}
    kmfs: dict[str, KaplanMeierFitter] = {}
    for d, g in df.groupby("disease_type"):
        label = str(d)
        kmf = KaplanMeierFitter(label=label)
        kmf.fit(g["time_hours"], g["event"], label=label)
        kmfs[label] = kmf
        ci_df = kmf.confidence_interval_survival_function_
        n_ci = int(ci_df.shape[1]) if ci_df is not None else 0
        if n_ci >= 2:
            ci_lo = ci_df.iloc[:, 0].tolist()
            ci_hi = ci_df.iloc[:, 1].tolist()
        else:
            sf0 = kmf.survival_function_.iloc[:, 0].tolist()
            ci_lo = ci_hi = sf0
        curves[label] = {
            "timeline_hours": kmf.timeline.tolist(),
            "survival": kmf.survival_function_.iloc[:, 0].tolist(),
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
        }

    logrank_summary: dict[str, Any] = {}
    dlist = sorted(df["disease_type"].astype(str).unique())
    if len(dlist) >= 2:
        a, b = dlist[0], dlist[1]
        ga = df[df["disease_type"].astype(str) == a]
        gb = df[df["disease_type"].astype(str) == b]
        if len(ga) > 0 and len(gb) > 0:
            lr = logrank_test(
                ga["time_hours"],
                gb["time_hours"],
                ga["event"],
                gb["event"],
            )
            logrank_summary = {
                "group_a": a,
                "group_b": b,
                "test_statistic": float(lr.test_statistic),
                "p_value": float(lr.p_value),
            }

    medians: dict[str, Any] = {}
    for label, kmf in kmfs.items():
        try:
            m = kmf.median_survival_time_
            medians[label] = float(m) if m is not None and pd.notna(m) else None
        except Exception:
            medians[label] = None

    lead = df[df["event"] == 1]["hours_lead_time"].dropna()
    lead_by: dict[str, Any] = {}
    for d, g in df[df["event"] == 1].groupby("disease_type"):
        hh = g["hours_lead_time"].dropna()
        lead_by[str(d)] = float(hh.median()) if len(hh) else None

    return {
        "kaplan_meier": curves,
        "median_survival_time_hours": medians,
        "median_hours_lead_time_among_detected": lead_by,
        "logrank": logrank_summary,
        "kmf_objects": kmfs,
    }


def fit_cox_proportional_hazards(
    survival_df: pd.DataFrame,
    *,
    penalizer: float = 0.1,
    check_assumptions: bool = True,
) -> dict[str, Any]:
    """Cox model on ``time_hours`` = hours from window start to detection or censor."""
    from lifelines import CoxPHFitter

    cols = ["time_hours", "event", "disease_sepsis", "sofa_score", "imaging_frequency"]
    df = survival_df.copy()
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Survival table missing {c}")
    df = df.dropna(subset=["time_hours", "event"])
    df["event"] = df["event"].astype(int)
    df["sofa_score"] = pd.to_numeric(df["sofa_score"], errors="coerce")
    df["imaging_frequency"] = pd.to_numeric(df["imaging_frequency"], errors="coerce")

    cox_df = df[["time_hours", "event", "disease_sepsis", "sofa_score", "imaging_frequency"]].copy()
    if cox_df["sofa_score"].notna().sum() < max(5, int(0.2 * len(cox_df))):
        logger.info("SOFA unavailable or sparse for most rows; fitting Cox without sofa_score.")
        cox_df = cox_df.drop(columns=["sofa_score"])
    cox_df = cox_df.dropna()

    if len(cox_df) < 10 or cox_df["event"].sum() < 3:
        return {"error": "insufficient_rows_or_events", "n_rows": int(len(cox_df))}

    cph = CoxPHFitter(penalizer=float(penalizer))
    cph.fit(cox_df, duration_col="time_hours", event_col="event")

    summary_csv = cph.summary.to_csv()
    violations: dict[str, Any] = {}
    if check_assumptions and len(cox_df) >= 20:
        try:
            res = cph.check_assumptions(cox_df, p_value_threshold=0.05, show_plots=False)
            violations["check_assumptions_return"] = str(res)
        except Exception as exc:
            violations["error"] = str(exc)

    try:
        c_index = float(cph.concordance_index_)
    except (ZeroDivisionError, ValueError, RuntimeError):
        c_index = float("nan")

    return {
        "concordance_index": c_index,
        "log_likelihood": float(cph.log_likelihood_),
        "summary_csv": summary_csv,
        "summary_dataframe": cph.summary,
        "proportional_hazard_checks": violations,
        "fitter": cph,
        "n_rows_used": int(len(cox_df)),
    }


def plot_km_by_disease(km_payload: dict[str, Any], out_path: Path, *, title: str | None = None) -> None:
    import matplotlib.pyplot as plt

    kmfs: dict[str, Any] = km_payload.get("kmf_objects") or {}
    if not kmfs:
        raise ValueError("No KaplanMeierFitter objects in payload (run fit_kaplan_meier_and_logrank first).")

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, kmf in kmfs.items():
        kmf.plot_survival_function(ax=ax, ci_show=True, label=str(label))
    ax.set_xlabel(
        "Hours from start of the 14-day pre-diagnosis window to VLM detection\n"
        "(or to diagnosis if censored); increasing time → closer to diagnosis documentation"
    )
    ax.set_ylabel("Probability (VLM not yet detected above threshold)")
    ax.set_title(title or "Kaplan–Meier: time to VLM detection (by disease cohort)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_auc_vs_hours(auc_df: pd.DataFrame, out_path: Path, *, title: str | None = None) -> None:
    import matplotlib.pyplot as plt

    if auc_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for d, g in auc_df.groupby("disease_type"):
        gg = g.sort_values("hours_before_diagnosis", ascending=False)
        aucv = pd.to_numeric(gg["roc_auc"], errors="coerce")
        mask = aucv.notna()
        if not mask.any():
            continue
        ax.plot(
            gg.loc[mask, "hours_before_diagnosis"],
            aucv.loc[mask],
            marker="o",
            label=str(d),
        )
    ax.set_xlabel("Landmark: hours before diagnosis (larger = farther from diagnosis time)")
    ax.set_ylabel(
        "ROC AUC (VLM score at landmark vs outcome; primary = late vs early imaging-window proxy)"
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title or "AUC vs hours before diagnosis")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_lead_time_outcome_analysis(
    *,
    events_csv: Path,
    inference_csv: Path,
    output_dir: Path,
    sofa_csv: Path | None = None,
    window_days: float = 14.0,
    anchor_hours: list[float] | None = None,
    penalizer: float = 0.1,
    skip_cox: bool = False,
    skip_ph_check: bool = False,
    auc_proxy_positive_within_hours: float = DEFAULT_POSITIVE_WITHIN_HOURS,
    auc_proxy_negative_at_least_hours: float = DEFAULT_NEGATIVE_AT_LEAST_HOURS,
) -> dict[str, Any]:
    """End-to-end tables, models, figures, and JSON summary under ``output_dir``."""
    if anchor_hours is None:
        anchor_hours = list(DEFAULT_ANCHOR_HOURS_BEFORE_DX)

    events = pd.read_csv(events_csv)
    inference = pd.read_csv(inference_csv)
    inference["study_datetime"] = pd.to_datetime(inference["study_datetime"], errors="coerce")

    sofa_tbl = load_sofa_per_admission(sofa_csv)
    surv = build_survival_outcomes_table(events, inference, sofa_per_adm=sofa_tbl, window_days=window_days)

    output_dir.mkdir(parents=True, exist_ok=True)
    surv_path = output_dir / "survival_outcomes_table.csv"
    surv.to_csv(surv_path, index=False)

    t1 = cohort_characteristics(inference)
    (output_dir / "cohort_characteristics.json").write_text(json.dumps(t1, indent=2), encoding="utf-8")

    km = fit_kaplan_meier_and_logrank(surv)
    km_plot_path = output_dir / "figures" / "km_vlm_detection_by_disease.png"
    plot_km_by_disease(km, km_plot_path)

    for d, curve in km["kaplan_meier"].items():
        pd.DataFrame(
            {
                "timeline_hours": curve["timeline_hours"],
                "survival": curve["survival"],
                "ci_lower": curve["ci_lower"],
                "ci_upper": curve["ci_upper"],
            }
        ).to_csv(output_dir / f"km_curve_{d}.csv", index=False)

    auc_df = auc_vs_hours_before_diagnosis(
        inference,
        surv,
        anchor_hours=anchor_hours,
        positive_within_hours=auc_proxy_positive_within_hours,
        negative_at_least_hours=auc_proxy_negative_at_least_hours,
    )
    auc_df.to_csv(output_dir / "auc_vs_hours_before_diagnosis.csv", index=False)
    plot_auc_vs_hours(auc_df, output_dir / "figures" / "auc_vs_hours_before_diagnosis.png")

    cox: dict[str, Any] = {}
    if not skip_cox:
        cox = fit_cox_proportional_hazards(surv, penalizer=penalizer, check_assumptions=not skip_ph_check)
        fitter = cox.pop("fitter", None)
        sdf = cox.pop("summary_dataframe", None)
        if sdf is not None and isinstance(sdf, pd.DataFrame):
            sdf.to_csv(output_dir / "cox_model_summary.csv")
        summary_csv = cox.get("summary_csv")
        if isinstance(summary_csv, str):
            (output_dir / "cox_model_summary_full.txt").write_text(summary_csv, encoding="utf-8")

    summary = {
        "n_survival_rows": int(len(surv)),
        "events_csv": str(events_csv.resolve()),
        "inference_csv": str(inference_csv.resolve()),
        "sofa_csv": str(sofa_csv.resolve()) if sofa_csv else None,
        "window_days": float(window_days),
        "anchor_hours": [float(h) for h in anchor_hours],
        "survival_time_hours_definition": (
            "time_hours = hours from pre-diagnosis window start (diagnosis_time − window_days) to "
            "first threshold crossing or to diagnosis_time if censored (used for KM/Cox so durations "
            "are not all zero when the first in-window study is the crossing study). "
            "time_hours_from_first_cxr = hours from earliest in-window study to the same endpoints "
            "(manuscript Phase 4 Step 1 clock)."
        ),
        "km_y_axis_definition": (
            "Kaplan–Meier survival = probability VLM has not yet crossed the calibrated threshold "
            "(manuscript Phase 4 Step 2)."
        ),
        "auc_primary_outcome": (
            "ROC AUC of disease_score at each landmark hour vs late-window (≤ "
            f"{auc_proxy_positive_within_hours:g} h before diagnosis) vs early-window "
            f"(≥ {auc_proxy_negative_at_least_hours:g} h) proxy on the landmark study; "
            "falls back to detection-vs-censor or lead-time splits when proxy labels are too sparse."
        ),
        "cohort_characteristics": t1,
        "median_survival_time_hours": km.get("median_survival_time_hours"),
        "median_hours_lead_time_among_detected": km.get("median_hours_lead_time_among_detected"),
        "logrank": km.get("logrank"),
        "cox": {k: v for k, v in cox.items() if k not in ("summary_dataframe",)},
    }
    (output_dir / "lead_time_outcome_analysis.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )

    km.pop("kmf_objects", None)
    return {
        "survival_outcomes_path": surv_path,
        "km_plot": km_plot_path,
        "auc_plot": output_dir / "figures" / "auc_vs_hours_before_diagnosis.png",
        "summary": summary,
    }
