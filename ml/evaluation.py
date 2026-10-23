"""
Model evaluation module for survival analysis.

Computes Harrell's Concordance Index (C-index), pair concordances,
and comparative metrics for survival models using scikit-survival.
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

# Try importing sksurv metric
try:
    from sksurv.metrics import concordance_index_censored
    SKSURV_AVAILABLE = True
except ImportError:
    concordance_index_censored = None
    SKSURV_AVAILABLE = False


def _compute_c_index_fallback(
    event_indicator: np.ndarray,
    event_time: np.ndarray,
    estimate: np.ndarray,
) -> Tuple[float, int, int, int, int]:
    """
    Pure Python/NumPy fallback implementation of Harrell's Concordance Index (C-index).

    Used if scikit-survival is not available in the current Python environment.

    Parameters
    ----------
    event_indicator : np.ndarray
        1D boolean array indicating whether the event occurred.
    event_time : np.ndarray
        1D numeric array of observed survival / follow-up times.
    estimate : np.ndarray
        1D numeric array of predicted risk scores (higher score = shorter survival expected).

    Returns
    -------
    c_index : float
        Concordance index in [0, 1].
    concordant : int
        Number of concordant pairs.
    discordant : int
        Number of discordant pairs.
    tied_risk : int
        Number of pairs with tied risk scores.
    tied_time : int
        Number of pairs with tied survival times.
    """
    n_samples = len(event_time)
    concordant = 0
    discordant = 0
    tied_risk = 0
    tied_time = 0

    for i in range(n_samples):
        # Only evaluate pairs where subject i experienced the event
        if not event_indicator[i]:
            continue

        for j in range(n_samples):
            if i == j:
                continue

            time_i = event_time[i]
            time_j = event_time[j]

            # Case: time_i < time_j (subject i died before subject j was censored or died)
            if time_i < time_j:
                risk_i = estimate[i]
                risk_j = estimate[j]

                if risk_i > risk_j:
                    concordant += 1
                elif risk_i < risk_j:
                    discordant += 1
                else:
                    tied_risk += 1

            elif time_i == time_j and not event_indicator[j]:
                # time_i == time_j, but subject j was censored at that time
                risk_i = estimate[i]
                risk_j = estimate[j]
                if risk_i > risk_j:
                    concordant += 1
                elif risk_i < risk_j:
                    discordant += 1
                else:
                    tied_risk += 1

            elif time_i == time_j and event_indicator[j] and i < j:
                # Both experienced event at the same time
                tied_time += 1

    total_evaluable = concordant + discordant + tied_risk
    if total_evaluable == 0:
        c_index = 0.5
    else:
        c_index = (concordant + 0.5 * tied_risk) / total_evaluable

    return float(c_index), concordant, discordant, tied_risk, tied_time


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "SurvivalModel",
) -> Dict[str, Any]:
    """
    Evaluate a survival model on a test set using Harrell's Concordance Index.

    Parameters
    ----------
    model : object
        Fitted survival analysis model with a `.predict(X)` method returning risk scores.
    X_test : np.ndarray
        2D test feature matrix.
    y_test : np.ndarray
        Structured array with fields ('event', bool) and ('survival_months', float),
        or 2D array / tuple of (events, times).
    model_name : str, default='SurvivalModel'
        Descriptive name of the model being evaluated.

    Returns
    -------
    dict
        Evaluation results containing:
        - 'model_name': name of the model
        - 'c_index': Concordance index (float)
        - 'concordant_pairs': count of concordant patient pairs
        - 'discordant_pairs': count of discordant patient pairs
        - 'tied_risk_pairs': count of tied risk pairs
        - 'tied_time_pairs': count of tied time pairs
        - 'total_evaluable_pairs': total comparable pairs
        - 'n_samples': number of test samples
        - 'interpretation': clinical / statistical interpretation sentence
    """
    # Generate risk scores (higher score = higher predicted risk)
    risk_scores = np.asarray(model.predict(X_test), dtype=float)

    # Extract event indicator and survival times from y_test
    if isinstance(y_test, np.ndarray) and y_test.dtype.names is not None:
        event_col = "event" if "event" in y_test.dtype.names else y_test.dtype.names[0]
        time_col = "survival_months" if "survival_months" in y_test.dtype.names else y_test.dtype.names[1]
        events = y_test[event_col].astype(bool)
        times = y_test[time_col].astype(float)
    elif isinstance(y_test, (tuple, list)) and len(y_test) == 2:
        events = np.asarray(y_test[0], dtype=bool)
        times = np.asarray(y_test[1], dtype=float)
    else:
        raise ValueError(
            "y_test must be a structured numpy array with 'event' and 'survival_months' fields."
        )

    # Compute C-index
    if SKSURV_AVAILABLE and concordance_index_censored is not None:
        res = concordance_index_censored(events, times, risk_scores)
        c_index = float(res[0])
        concordant = int(res[1])
        discordant = int(res[2])
        tied_risk = int(res[3])
        tied_time = int(res[4])
    else:
        c_index, concordant, discordant, tied_risk, tied_time = _compute_c_index_fallback(
            events, times, risk_scores
        )

    total_evaluable = concordant + discordant + tied_risk

    # Clinical interpretation text
    if c_index >= 0.8:
        quality = "excellent"
    elif c_index >= 0.7:
        quality = "good"
    elif c_index >= 0.6:
        quality = "moderate"
    else:
        quality = "poor / near random"

    interpretation = (
        f"The {model_name} model achieved a C-index of {c_index:.4f} ({quality} discrimination), "
        f"correctly ranking {c_index * 100:.1f}% of comparable patient survival pairs."
    )

    return {
        "model_name": model_name,
        "c_index": round(c_index, 4),
        "concordant_pairs": concordant,
        "discordant_pairs": discordant,
        "tied_risk_pairs": tied_risk,
        "tied_time_pairs": tied_time,
        "total_evaluable_pairs": total_evaluable,
        "n_samples": int(len(X_test)),
        "discrimination_quality": quality,
        "interpretation": interpretation,
    }


def compare_models(
    models_dict: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """
    Evaluate and compare multiple survival models on the test set.

    Parameters
    ----------
    models_dict : dict
        Mapping of model identifier (e.g., 'rsf', 'cox_ph') to trained model instance.
    X_test : np.ndarray
        Test feature matrix.
    y_test : np.ndarray
        Test structured target array.

    Returns
    -------
    dict
        Comparative report containing individual model metrics, ranking,
        the best performing model name, and metric interpretation explanation.
    """
    results: Dict[str, Dict[str, Any]] = {}
    best_model_name: Optional[str] = None
    best_c_index = -1.0

    for name, model in models_dict.items():
        metrics = evaluate_model(model, X_test, y_test, model_name=name)
        results[name] = metrics
        if metrics["c_index"] > best_c_index:
            best_c_index = metrics["c_index"]
            best_model_name = name

    c_index_explanation = (
        "Harrell's Concordance Index (C-index) is the gold-standard metric for evaluating "
        "survival model discrimination under right-censored data. It represents the probability "
        "that, for a randomly selected pair of patients where one experienced an event earlier "
        "than the other, the model assigned a higher risk score to the earlier-event patient. "
        "A C-index of 0.5 indicates random chance, 0.70-0.79 represents acceptable/good discrimination, "
        "0.80-0.89 represents excellent discrimination, and 1.0 represents perfect discrimination."
    )

    return {
        "models": results,
        "best_model": best_model_name,
        "best_c_index": round(best_c_index, 4),
        "c_index_explanation": c_index_explanation,
    }
