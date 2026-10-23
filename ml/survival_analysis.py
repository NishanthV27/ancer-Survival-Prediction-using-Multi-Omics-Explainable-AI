"""
Kaplan-Meier Survival Analysis and Hypothesis Testing Module.

Computes Kaplan-Meier survival curves, confidence intervals, median survival times,
and log-rank hypothesis tests across risk tiers, clinical stages, and cohorts
using lifelines (with a pure mathematical fallback).
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# Lifelines import with fallback handling
try:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test, multivariate_logrank_test
    LIFELINES_AVAILABLE = True
    LIFELINES_IMPORT_ERROR = None
except ImportError as err:
    KaplanMeierFitter = None
    logrank_test = None
    multivariate_logrank_test = None
    LIFELINES_AVAILABLE = False
    LIFELINES_IMPORT_ERROR = str(err)


def _compute_km_fallback(
    durations: np.ndarray,
    events: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute Kaplan-Meier survival curve using the standard product-limit estimator
    and Greenwood's formula for 95% confidence intervals when lifelines is absent.
    """
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=bool)

    # Sort by time
    order = np.argsort(durations)
    durations_sorted = durations[order]
    events_sorted = events[order]

    unique_times = np.unique(durations_sorted)
    # Start at time 0
    timeline = [0.0]
    survival = [1.0]
    var_sum = 0.0
    variances = [0.0]

    n_at_risk = len(durations)
    s_curr = 1.0

    for t in unique_times:
        mask = durations_sorted == t
        d = int(np.sum(events_sorted[mask]))  # Events at t
        c = int(np.sum(~events_sorted[mask]))  # Censored at t

        if n_at_risk > 0 and d > 0:
            s_curr *= (1.0 - d / n_at_risk)
            if n_at_risk > d:
                var_sum += d / (n_at_risk * (n_at_risk - d))

        timeline.append(float(t))
        survival.append(float(s_curr))
        variances.append(float((s_curr ** 2) * var_sum))

        n_at_risk -= (d + c)

    # 95% Confidence intervals (normal approximation clamped [0, 1])
    ci_lower = []
    ci_upper = []
    for s_val, var_val in zip(survival, variances):
        se = np.sqrt(max(0.0, var_val))
        ci_lower.append(float(max(0.0, s_val - 1.96 * se)))
        ci_upper.append(float(min(1.0, s_val + 1.96 * se)))

    # Median survival: first time where S(t) <= 0.5
    median_survival = None
    for t_val, s_val in zip(timeline, survival):
        if s_val <= 0.5:
            median_survival = float(t_val)
            break

    return {
        "timeline": [round(t, 2) for t in timeline],
        "survival_probability": [round(s, 4) for s in survival],
        "ci_lower": [round(cl, 4) for cl in ci_lower],
        "ci_upper": [round(cu, 4) for cu in ci_upper],
        "median_survival": median_survival,
        "n_patients": len(durations),
        "n_events": int(np.sum(events)),
    }


def _compute_logrank_fallback(
    durations_a: np.ndarray,
    events_a: np.ndarray,
    durations_b: np.ndarray,
    events_b: np.ndarray,
) -> Tuple[float, float]:
    """
    Standard two-sample Mantel-Cox log-rank test fallback.
    """
    all_times = np.unique(np.concatenate([durations_a, durations_b]))
    all_times.sort()

    o1_total = 0.0
    e1_total = 0.0
    v_total = 0.0

    n1 = len(durations_a)
    n2 = len(durations_b)

    for t in all_times:
        d1 = int(np.sum((durations_a == t) & events_a))
        d2 = int(np.sum((durations_b == t) & events_b))
        d = d1 + d2

        y1 = int(np.sum(durations_a >= t))
        y2 = int(np.sum(durations_b >= t))
        y = y1 + y2

        if y > 1 and d > 0:
            e1 = d * (y1 / y)
            o1_total += d1
            e1_total += e1
            v = (y1 * y2 * d * (y - d)) / ((y ** 2) * (y - 1))
            v_total += v

    if v_total > 0:
        stat = ((o1_total - e1_total) ** 2) / v_total
        # Approximate 1-df p-value from chi2
        # P(X >= stat) for chi2(1)
        z = np.sqrt(stat)
        # Standard normal tail approximation
        p_val = float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / np.sqrt(2.0)))))
    else:
        stat = 0.0
        p_val = 1.0

    return float(stat), float(max(0.0, min(1.0, p_val)))


def compute_km_curves(
    df: pd.DataFrame,
    risk_scores: Union[np.ndarray, List[float], pd.Series],
    threshold: float,
) -> Dict[str, Any]:
    """
    Split patients into High Risk and Low Risk strata using a threshold,
    fit Kaplan-Meier survival curves for both groups, and calculate log-rank significance.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'survival_months' and 'event' columns.
    risk_scores : array-like
        Predicted risk score for each patient in df.
    threshold : float
        Decision threshold for risk stratification (e.g. median training risk score).

    Returns
    -------
    dict
        Dictionary containing curve data (timeline, probabilities, 95% CIs)
        for High and Low Risk groups, along with log-rank statistical test results.
    """
    scores = np.asarray(risk_scores, dtype=float)
    if len(scores) != len(df):
        raise ValueError(f"Length mismatch: df has {len(df)} rows, but risk_scores has {len(scores)} elements.")

    high_mask = scores >= threshold
    low_mask = ~high_mask

    df_high = df[high_mask]
    df_low = df[low_mask]

    result: Dict[str, Any] = {}

    if LIFELINES_AVAILABLE and KaplanMeierFitter is not None:
        # High Risk fit
        kmf_high = KaplanMeierFitter()
        kmf_high.fit(
            durations=df_high["survival_months"],
            event_observed=df_high["event"].astype(bool),
            label="High Risk",
        )
        # Low Risk fit
        kmf_low = KaplanMeierFitter()
        kmf_low.fit(
            durations=df_low["survival_months"],
            event_observed=df_low["event"].astype(bool),
            label="Low Risk",
        )

        # Extract timeline and probabilities
        high_curve = {
            "timeline": [round(float(t), 2) for t in kmf_high.timeline],
            "survival_probability": [round(float(p), 4) for p in kmf_high.survival_function_["High Risk"]],
            "ci_lower": [round(float(ci), 4) for ci in kmf_high.confidence_interval_["High Risk_lower_0.95"]],
            "ci_upper": [round(float(ci), 4) for ci in kmf_high.confidence_interval_["High Risk_upper_0.95"]],
            "median_survival": None if np.isinf(kmf_high.median_survival_time_) else round(float(kmf_high.median_survival_time_), 2),
            "n_patients": int(len(df_high)),
            "n_events": int(df_high["event"].sum()),
        }

        low_curve = {
            "timeline": [round(float(t), 2) for t in kmf_low.timeline],
            "survival_probability": [round(float(p), 4) for p in kmf_low.survival_function_["Low Risk"]],
            "ci_lower": [round(float(ci), 4) for ci in kmf_low.confidence_interval_["Low Risk_lower_0.95"]],
            "ci_upper": [round(float(ci), 4) for ci in kmf_low.confidence_interval_["Low Risk_upper_0.95"]],
            "median_survival": None if np.isinf(kmf_low.median_survival_time_) else round(float(kmf_low.median_survival_time_), 2),
            "n_patients": int(len(df_low)),
            "n_events": int(df_low["event"].sum()),
        }

        # Log-rank test
        lr = logrank_test(
            df_high["survival_months"],
            df_low["survival_months"],
            df_high["event"].astype(bool),
            df_low["event"].astype(bool),
        )
        test_stat = float(lr.test_statistic)
        p_val = float(lr.p_value)

    else:
        # Fallback implementation
        high_curve = _compute_km_fallback(df_high["survival_months"].values, df_high["event"].values)
        low_curve = _compute_km_fallback(df_low["survival_months"].values, df_low["event"].values)
        test_stat, p_val = _compute_logrank_fallback(
            df_high["survival_months"].values,
            df_high["event"].values,
            df_low["survival_months"].values,
            df_low["event"].values,
        )

    p_val_fmt = "< 0.0001" if p_val < 0.0001 else f"{p_val:.4f}"

    result["high_risk"] = high_curve
    result["low_risk"] = low_curve
    result["logrank_test"] = {
        "test_statistic": round(test_stat, 4),
        "p_value": p_val,
        "p_value_formatted": p_val_fmt,
        "is_significant": bool(p_val < 0.05),
        "threshold_used": round(float(threshold), 4),
    }
    result["logrank_p"] = p_val
    result["groups"] = {
        "High Risk": {
            "n": high_curve.get("n_patients", len(df_high)),
            "timeline": high_curve.get("timeline", []),
            "survival_prob": high_curve.get("survival_probability", []),
            "ci_lower": high_curve.get("ci_lower", []),
            "ci_upper": high_curve.get("ci_upper", []),
            "median_survival": high_curve.get("median_survival"),
        },
        "Low Risk": {
            "n": low_curve.get("n_patients", len(df_low)),
            "timeline": low_curve.get("timeline", []),
            "survival_prob": low_curve.get("survival_probability", []),
            "ci_lower": low_curve.get("ci_lower", []),
            "ci_upper": low_curve.get("ci_upper", []),
            "median_survival": low_curve.get("median_survival"),
        },
    }

    return result


def compute_km_by_feature(
    df: pd.DataFrame,
    feature: str = "tumor_stage",
    groups: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute Kaplan-Meier survival curves stratified across categories of a clinical feature.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the feature column, 'survival_months', and 'event'.
    feature : str, default='tumor_stage'
        Categorical feature column name.
    groups : list of str, optional
        Specific ordered categories to evaluate. If None, uses unique values in df.

    Returns
    -------
    dict
        Stratified KM curves for each category, plus multi-group log-rank significance test.
    """
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' not found in dataframe columns: {list(df.columns)}")

    if groups is None:
        groups = sorted(df[feature].dropna().unique().tolist())

    curves: Dict[str, Any] = {}

    for grp in groups:
        subset = df[df[feature] == grp]
        if len(subset) == 0:
            continue

        if LIFELINES_AVAILABLE and KaplanMeierFitter is not None:
            kmf = KaplanMeierFitter()
            kmf.fit(
                durations=subset["survival_months"],
                event_observed=subset["event"].astype(bool),
                label=str(grp),
            )
            curves[str(grp)] = {
                "timeline": [round(float(t), 2) for t in kmf.timeline],
                "survival_probability": [round(float(p), 4) for p in kmf.survival_function_[str(grp)]],
                "ci_lower": [round(float(ci), 4) for ci in kmf.confidence_interval_[f"{grp}_lower_0.95"]],
                "ci_upper": [round(float(ci), 4) for ci in kmf.confidence_interval_[f"{grp}_upper_0.95"]],
                "median_survival": None if np.isinf(kmf.median_survival_time_) else round(float(kmf.median_survival_time_), 2),
                "n_patients": int(len(subset)),
                "n_events": int(subset["event"].sum()),
            }
        else:
            curves[str(grp)] = _compute_km_fallback(
                subset["survival_months"].values,
                subset["event"].values,
            )

    # Multi-group logrank test
    logrank_res = logrank_test_groups(df, group_column=feature)

    formatted_groups = {}
    for grp, c in curves.items():
        formatted_groups[grp] = {
            "n": c.get("n_patients", len(df[df[feature] == grp])),
            "timeline": c.get("timeline", []),
            "survival_prob": c.get("survival_probability", []),
            "ci_lower": c.get("ci_lower", []),
            "ci_upper": c.get("ci_upper", []),
            "median_survival": c.get("median_survival"),
        }

    return {
        "feature": feature,
        "curves": curves,
        "groups": formatted_groups,
        "logrank_test": logrank_res,
        "logrank_p": logrank_res.get("p_value", 1.0),
    }


def logrank_test_groups(
    df: pd.DataFrame,
    group_column: str,
    time_col: str = "survival_months",
    event_col: str = "event",
) -> Dict[str, Any]:
    """
    Perform hypothesis testing (log-rank test) to compare survival distributions between groups.

    Uses two-sample log-rank test for 2 categories, or multivariate log-rank test
    for 3 or more categories.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing group, time, and event columns.
    group_column : str
        Categorical grouping column name.
    time_col : str, default='survival_months'
        Survival time column name.
    event_col : str, default='event'
        Event indicator column name.

    Returns
    -------
    dict
        Statistical test results including test statistic, p-value, and significance flag.
    """
    clean_df = df.dropna(subset=[group_column, time_col, event_col])
    groups = clean_df[group_column].unique()

    if len(groups) < 2:
        return {
            "test_statistic": 0.0,
            "p_value": 1.0,
            "p_value_formatted": "1.0000",
            "is_significant": False,
            "note": "At least 2 distinct groups are required for a log-rank test.",
        }

    if LIFELINES_AVAILABLE and multivariate_logrank_test is not None:
        if len(groups) == 2:
            g1, g2 = groups[0], groups[1]
            sub1 = clean_df[clean_df[group_column] == g1]
            sub2 = clean_df[clean_df[group_column] == g2]
            res = logrank_test(
                sub1[time_col],
                sub2[time_col],
                sub1[event_col].astype(bool),
                sub2[event_col].astype(bool),
            )
        else:
            res = multivariate_logrank_test(
                clean_df[time_col],
                clean_df[group_column],
                clean_df[event_col].astype(bool),
            )

        stat = float(res.test_statistic)
        p_val = float(res.p_value)

    else:
        # Fallback between first two groups
        g1, g2 = groups[0], groups[1]
        sub1 = clean_df[clean_df[group_column] == g1]
        sub2 = clean_df[clean_df[group_column] == g2]
        stat, p_val = _compute_logrank_fallback(
            sub1[time_col].values,
            sub1[event_col].values,
            sub2[time_col].values,
            sub2[event_col].values,
        )

    p_val_fmt = "< 0.0001" if p_val < 0.0001 else f"{p_val:.4f}"

    return {
        "test_statistic": round(stat, 4),
        "p_value": p_val,
        "p_value_formatted": p_val_fmt,
        "is_significant": bool(p_val < 0.05),
        "groups": [str(g) for g in groups],
        "degrees_of_freedom": len(groups) - 1,
    }


def get_km_plot_data(df: pd.DataFrame, models: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main analytics function that generates all Kaplan-Meier data required by the web application:
    - High vs. Low Risk KM curves (using best model predictions and established threshold).
    - Tumor stage stratified KM curves.
    - Overall cohort survival curve.
    - Summary milestone statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Clinical cancer dataset.
    models : dict
        Loaded models dictionary containing 'preprocessor_info', 'metadata',
        and trained survival models.

    Returns
    -------
    dict
        Complete analytical package formatted for front-end visualization.
    """
    preprocessor_info = models["preprocessor_info"]
    metadata = models.get("metadata", {})

    best_model_name = metadata.get("best_model", "rsf")
    if best_model_name not in models:
        best_model_name = "rsf" if "rsf" in models else "cox_ph"

    model = models[best_model_name]

    # Transform entire cohort
    col_transformer = preprocessor_info["column_transformer"]
    feature_cols = preprocessor_info.get("numeric_features", []) + preprocessor_info.get("categorical_features", [])
    X_all = col_transformer.transform(df[feature_cols])

    # Predict risk scores
    risk_scores = np.asarray(model.predict(X_all), dtype=float)

    # Threshold
    thresholds = metadata.get("thresholds", {})
    threshold = float(thresholds.get(best_model_name, metadata.get("risk_threshold", np.median(risk_scores))))

    # 1. High vs Low Risk curves
    risk_curves = compute_km_curves(df, risk_scores, threshold)

    # 2. Stratified by Tumor Stage
    stage_curves = compute_km_by_feature(df, feature="tumor_stage")

    # 3. Overall cohort curve
    if LIFELINES_AVAILABLE and KaplanMeierFitter is not None:
        kmf_overall = KaplanMeierFitter()
        kmf_overall.fit(
            durations=df["survival_months"],
            event_observed=df["event"].astype(bool),
            label="Overall Cohort",
        )
        overall_curve = {
            "timeline": [round(float(t), 2) for t in kmf_overall.timeline],
            "survival_probability": [round(float(p), 4) for p in kmf_overall.survival_function_["Overall Cohort"]],
            "ci_lower": [round(float(ci), 4) for ci in kmf_overall.confidence_interval_["Overall Cohort_lower_0.95"]],
            "ci_upper": [round(float(ci), 4) for ci in kmf_overall.confidence_interval_["Overall Cohort_upper_0.95"]],
            "median_survival": None if np.isinf(kmf_overall.median_survival_time_) else round(float(kmf_overall.median_survival_time_), 2),
            "n_patients": int(len(df)),
            "n_events": int(df["event"].sum()),
        }
    else:
        overall_curve = _compute_km_fallback(df["survival_months"].values, df["event"].values)

    return {
        "status": "success",
        "model_used": best_model_name,
        "threshold": round(threshold, 4),
        "groups": risk_curves.get("groups", {}),
        "logrank_p": risk_curves.get("logrank_p", 1.0),
        "risk_stratified_curves": risk_curves,
        "stage_stratified_curves": stage_curves,
        "overall_curve": overall_curve,
        "cohort_summary": {
            "total_patients": int(len(df)),
            "total_events": int(df["event"].sum()),
            "event_rate_pct": round(float(df["event"].mean() * 100), 2),
            "median_followup_months": round(float(df["survival_months"].median()), 1),
            "max_followup_months": round(float(df["survival_months"].max()), 1),
        },
    }
