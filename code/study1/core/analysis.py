from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - dependency availability is environment specific
    plt = None

try:
    import statsmodels.formula.api as smf
except ImportError:  # pragma: no cover - dependency availability is environment specific
    smf = None

FEATURE_COLUMNS = [
    "word_count",
    "hedge_rate",
    "mean_sent_length",
    "ttr",
    "certainty_score",
]

logger = logging.getLogger(__name__)


def _ordered_circadian_levels(values: pd.Series) -> list[str]:
    preferred_order = ["night", "morning", "afternoon", "evening", "day", "unknown"]
    present = [x for x in preferred_order if x in set(values.dropna().astype(str))]
    extras = sorted(set(values.dropna().astype(str)) - set(preferred_order))
    return present + extras


def _set_circadian_categorical(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    levels = _ordered_circadian_levels(out["circadian_bin"])
    out["circadian_bin"] = pd.Categorical(out["circadian_bin"], categories=levels, ordered=True)
    return out


def _pick_reference_bin(levels: list[str]) -> str:
    if "morning" in levels:
        return "morning"
    if "day" in levels:
        return "day"
    return levels[0]


def _cohens_d(sample_a: pd.Series, sample_b: pd.Series) -> float:
    a = pd.to_numeric(sample_a, errors="coerce").dropna()
    b = pd.to_numeric(sample_b, errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    pooled_sd = np.sqrt(((len(a) - 1) * var_a + (len(b) - 1) * var_b) / (len(a) + len(b) - 2))
    if pooled_sd == 0:
        return np.nan
    return (a.mean() - b.mean()) / pooled_sd


def _safe_mixedlm_fit(model_formula: str, data: pd.DataFrame) -> object | None:
    if smf is None:
        raise ImportError(
            "statsmodels is required for stats modeling. "
            "Install it with: pip install statsmodels"
        )
    try:
        model = smf.mixedlm(
            model_formula,
            data=data,
            groups=data["radiologist_cluster"],
            re_formula="1",
        )
        return model.fit(method="lbfgs", reml=False)
    except Exception as exc:  # pragma: no cover - model failures are data dependent
        logger.warning("MixedLM failed for formula '%s': %s", model_formula, exc)
        return None


def build_table1_descriptives(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    ordered = _set_circadian_categorical(df)
    grouped = ordered.groupby("circadian_bin", observed=False)
    rows: list[dict[str, object]] = []
    for circadian_bin, group in grouped:
        row: dict[str, object] = {
            "circadian_bin": circadian_bin,
            "n_reports": int(len(group)),
        }
        for feature in feature_columns:
            vals = pd.to_numeric(group[feature], errors="coerce")
            mean = vals.mean()
            sd = vals.std(ddof=1)
            row[f"{feature}_mean"] = mean
            row[f"{feature}_sd"] = sd
            row[f"{feature}_mean_sd"] = (
                f"{mean:.4f} ± {sd:.4f}" if pd.notna(mean) and pd.notna(sd) else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_mixedlm_and_effects(
    df: pd.DataFrame,
    feature_columns: list[str],
    alpha: float = 0.05,
    force_three_comparisons: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = _set_circadian_categorical(df)
    levels = list(ordered["circadian_bin"].cat.categories)
    reference = _pick_reference_bin(levels)
    n_features = len(feature_columns)
    n_comparisons = (3 if force_three_comparisons else max(0, len(levels) - 1)) * n_features
    bonferroni_threshold = alpha / n_comparisons if n_comparisons > 0 else np.nan

    fixed_rows: list[dict[str, object]] = []
    icc_rows: list[dict[str, object]] = []

    for feature in feature_columns:
        model_df = ordered[["circadian_bin", "severity", "radiologist_cluster", feature]].copy()
        model_df[feature] = pd.to_numeric(model_df[feature], errors="coerce")
        model_df["severity"] = pd.to_numeric(model_df["severity"], errors="coerce")
        model_df = model_df.dropna(subset=[feature, "severity", "radiologist_cluster", "circadian_bin"])
        if model_df.empty:
            continue

        formula = f"{feature} ~ C(circadian_bin, Treatment(reference='{reference}')) + severity"
        fit_result = _safe_mixedlm_fit(formula, model_df)
        if fit_result is None:
            continue

        pvalues = fit_result.pvalues.to_dict()
        params = fit_result.params.to_dict()
        std_err = fit_result.bse.to_dict()

        ref_values = model_df.loc[model_df["circadian_bin"] == reference, feature]
        compared_levels = [lv for lv in levels if lv != reference]
        for level in compared_levels:
            contrast_name = f"C(circadian_bin, Treatment(reference='{reference}'))[T.{level}]"
            pval = float(pvalues.get(contrast_name, np.nan))
            coef = float(params.get(contrast_name, np.nan))
            se = float(std_err.get(contrast_name, np.nan))
            comp_values = model_df.loc[model_df["circadian_bin"] == level, feature]
            d_val = _cohens_d(comp_values, ref_values)
            fixed_rows.append(
                {
                    "feature": feature,
                    "reference_bin": reference,
                    "comparison_bin": level,
                    "coef": coef,
                    "std_err": se,
                    "p_value": pval,
                    "p_value_bonferroni": min(1.0, pval * n_comparisons) if pd.notna(pval) else np.nan,
                    "significant_bonferroni": bool(pval < bonferroni_threshold) if pd.notna(pval) and pd.notna(bonferroni_threshold) else False,
                    "cohens_d": d_val,
                    "n_reference": int(ref_values.notna().sum()),
                    "n_comparison": int(comp_values.notna().sum()),
                    "bonferroni_threshold": bonferroni_threshold,
                    "n_tests": n_comparisons,
                }
            )

        icc_model_based = np.nan
        if fit_result.cov_re is not None and getattr(fit_result.cov_re, "size", 0) > 0:
            random_intercept_var = float(fit_result.cov_re.iloc[0, 0])
            resid_var = float(fit_result.scale)
            denom = random_intercept_var + resid_var
            icc_model_based = (random_intercept_var / denom) if denom > 0 else np.nan

        icc_row = {
            "feature": feature,
            "icc": icc_model_based,
            "icc_method": "mixedlm_variance_ratio",
            "random_intercept_var": float(fit_result.cov_re.iloc[0, 0]) if fit_result.cov_re is not None else np.nan,
            "residual_var": float(fit_result.scale),
        }

        try:
            import pingouin as pg

            icc_input = model_df.rename(
                columns={
                    "radiologist_cluster": "targets",
                    "circadian_bin": "raters",
                    feature: "ratings",
                }
            )[["targets", "raters", "ratings"]]
            icc_out = pg.intraclass_corr(data=icc_input, targets="targets", raters="raters", ratings="ratings")
            icc2 = icc_out.loc[icc_out["Type"] == "ICC2", "ICC"]
            if not icc2.empty:
                icc_row["icc"] = float(icc2.iloc[0])
                icc_row["icc_method"] = "pingouin_icc2"
        except Exception:  # pragma: no cover - optional dependency and data-shape dependent
            pass

        icc_rows.append(icc_row)

    return pd.DataFrame(fixed_rows), pd.DataFrame(icc_rows)


def save_feature_plots(df: pd.DataFrame, feature_columns: list[str], out_path: Path) -> None:
    if plt is None:
        raise ImportError(
            "matplotlib is required for results plotting. "
            "Install it with: pip install matplotlib"
        )
    ordered = _set_circadian_categorical(df)
    bins = list(ordered["circadian_bin"].cat.categories)
    n_features = len(feature_columns)
    fig, axes = plt.subplots(n_features, 1, figsize=(10, 3.3 * n_features), constrained_layout=True)
    if n_features == 1:
        axes = [axes]

    for ax, feature in zip(axes, feature_columns):
        series_by_bin = [
            pd.to_numeric(ordered.loc[ordered["circadian_bin"] == bin_name, feature], errors="coerce").dropna()
            for bin_name in bins
        ]
        valid_series = [s for s in series_by_bin if not s.empty]
        valid_positions = [idx for idx, s in enumerate(series_by_bin, start=1) if not s.empty]
        if valid_series:
            parts = ax.violinplot(valid_series, positions=valid_positions, showmeans=True, showextrema=True)
            for body in parts["bodies"]:
                body.set_alpha(0.5)
        ax.boxplot(
            series_by_bin,
            labels=bins,
            showfliers=False,
            widths=0.35,
            patch_artist=False,
        )
        ax.set_title(feature)
        ax.set_xlabel("circadian_bin")
        ax.set_ylabel(feature)
        ax.tick_params(axis="x", rotation=20)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle("Study 1 Results: Language Features by Circadian Bin")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def run_stats_results(
    input_csv: Path,
    out_dir: Path,
    alpha: float = 0.05,
    force_three_comparisons: bool = False,
    feature_columns: list[str] | None = None,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_csv)

    cols = feature_columns if feature_columns is not None else FEATURE_COLUMNS
    required_cols = {"study_id", "subject_id", "circadian_bin", "severity", "radiologist_cluster", *cols}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in feature extraction CSV: {missing}")

    table1 = build_table1_descriptives(df, cols)
    table2, icc_table = build_mixedlm_and_effects(
        df,
        cols,
        alpha=alpha,
        force_three_comparisons=force_three_comparisons,
    )

    table1_path = out_dir / "study1_table1_descriptives.csv"
    table2_path = out_dir / "study1_table2_mixedlm_effects.csv"
    icc_path = out_dir / "study1_icc_summary.csv"
    figure_path = out_dir / "study1_feature_distributions.png"

    table1.to_csv(table1_path, index=False)
    table2.to_csv(table2_path, index=False)
    icc_table.to_csv(icc_path, index=False)
    save_feature_plots(df, cols, figure_path)

    return {
        "table1": table1_path,
        "table2": table2_path,
        "icc": icc_path,
        "figure": figure_path,
    }
