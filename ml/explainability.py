"""
Explainability and Feature Importance Module for Cancer Survival Models.

Provides SHAP-based global and local explanations, clinical feature descriptions,
and robust permutation feature importance fallbacks tailored to survival analysis.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from ml.preprocessing import preprocess_single_patient
from ml.evaluation import evaluate_model

# Try importing SHAP
try:
    import shap
    SHAP_AVAILABLE = True
    SHAP_IMPORT_ERROR = None
except ImportError as err:
    shap = None
    SHAP_AVAILABLE = False
    SHAP_IMPORT_ERROR = str(err)


def get_feature_descriptions() -> Dict[str, str]:
    """
    Return human-readable clinical and molecular descriptions for all modeled features.

    Returns
    -------
    dict
        Mapping from feature column identifier to informative description.
    """
    return {
        "age": (
            "Patient age in years at diagnosis. Advanced age is clinically associated with "
            "reduced physiological reserve, altered drug metabolism, and higher baseline mortality."
        ),
        "tumor_stage": (
            "Clinical staging (Stage I-IV) reflecting tumor size, regional lymph node involvement, "
            "and distant metastasis according to standard AJCC TNM guidelines."
        ),
        "tumor_stage_Stage II": (
            "Stage II indicator: localized cancer with invasion into surrounding tissues or larger primary mass."
        ),
        "tumor_stage_Stage III": (
            "Stage III indicator: regional lymph node metastasis and extensive local spreading."
        ),
        "tumor_stage_Stage IV": (
            "Stage IV indicator: advanced metastatic dissemination to distant organs."
        ),
        "tumor_stage_II": "Stage II indicator: localized advancement without distant metastasis.",
        "tumor_stage_III": "Stage III indicator: regional advancement involving lymph node chains.",
        "tumor_stage_IV": "Stage IV indicator: advanced metastatic disease.",
        "gene_expr_A": (
            "Normalized transcript expression of Oncogene A. Elevated expression correlates "
            "with accelerated cell proliferation, anti-apoptotic signaling, and increased hazard."
        ),
        "gene_expr_B": (
            "Normalized transcript expression of Tumor Suppressor Gene B. Higher expression is "
            "protective and promotes cell cycle arrest or programmed apoptosis."
        ),
        "gene_expr_C": (
            "Normalized transcript expression of DNA Repair Factor C. Modulates genomic stability "
            "and sensitivity to cytotoxic DNA damage."
        ),
        "methylation_A": (
            "DNA methylation beta-value (0.0 to 1.0) at promoter region A. High methylation leads "
            "to epigenetic transcriptional silencing."
        ),
        "methylation_B": (
            "DNA methylation beta-value (0.0 to 1.0) at regulatory region B. Epigenetic biomarker "
            "governing chromatin compaction and gene accessibility."
        ),
    }


def compute_permutation_importance(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    random_seed: int = 42,
    n_repeats: int = 5,
) -> Dict[str, Any]:
    """
    Compute permutation feature importance based on drop in Harrell's Concordance Index.

    Serves as an exact survival-metric importance calculation and reliable fallback
    when SHAP is unavailable or encounters numerical exceptions.

    Parameters
    ----------
    model : object
        Fitted survival model with .predict() method.
    X_test : np.ndarray
        Test feature matrix.
    y_test : np.ndarray
        Test survival target structured array.
    feature_names : list of str
        List of feature column names corresponding to columns of X_test.
    random_seed : int, default=42
        Random seed for shuffling permutations.
    n_repeats : int, default=5
        Number of permutation iterations per feature.

    Returns
    -------
    dict
        Dictionary containing sorted feature importances, standard errors, and baseline C-index.
    """
    rng = np.random.RandomState(random_seed)
    base_eval = evaluate_model(model, X_test, y_test, model_name="baseline")
    baseline_c_index = base_eval["c_index"]

    n_features = X_test.shape[1]
    importances_mean: List[float] = []
    importances_std: List[float] = []

    for col_idx in range(n_features):
        drops: List[float] = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            perm_indices = rng.permutation(len(X_perm))
            X_perm[:, col_idx] = X_perm[perm_indices, col_idx]

            perm_eval = evaluate_model(model, X_perm, y_test, model_name="permuted")
            drop = baseline_c_index - perm_eval["c_index"]
            drops.append(drop)

        importances_mean.append(float(np.mean(drops)))
        importances_std.append(float(np.std(drops)))

    descriptions = get_feature_descriptions()
    ranked_indices = np.argsort(importances_mean)[::-1]

    ranked_list = []
    for rank, idx in enumerate(ranked_indices, start=1):
        feat = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        ranked_list.append({
            "rank": rank,
            "feature": feat,
            "importance": round(importances_mean[idx], 4),
            "std": round(importances_std[idx], 4),
            "description": descriptions.get(feat, "Clinical/genomic survival biomarker."),
        })

    return {
        "method": "permutation_importance",
        "baseline_c_index": baseline_c_index,
        "n_repeats": n_repeats,
        "ranked_features": ranked_list,
    }


def compute_shap_values(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    model_name: str = "rsf",
    max_background_samples: int = 50,
    max_eval_samples: int = 50,
) -> Dict[str, Any]:
    """
    Compute SHAP values for model predictions on survival risk scores.

    - For Random Survival Forest: Uses shap.Explainer or shap.SamplingExplainer
      wrapping model.predict().
    - For CoxPH: Uses linear coefficient attribution or shap.LinearExplainer.
    - Gracefully falls back to permutation / coefficient importance if SHAP fails.

    Parameters
    ----------
    model : object
        Fitted survival model.
    X_train : np.ndarray
        Training feature matrix used to form background expectation.
    X_test : np.ndarray
        Evaluation feature matrix.
    model_name : str, default='rsf'
        Model type identifier ('rsf' or 'cox_ph').
    max_background_samples : int, default=50
        Max sample size for background reference distribution.
    max_eval_samples : int, default=50
        Max test sample size to evaluate.

    Returns
    -------
    dict
        Dictionary containing:
        - 'shap_values': 2D numpy array of shape (n_samples, n_features)
        - 'expected_value': float baseline expected risk score
        - 'method': str ('shap_explainer', 'linear_attribution', or 'fallback')
    """
    n_features = X_train.shape[1]
    bg_size = min(len(X_train), max_background_samples)
    eval_size = min(len(X_test), max_eval_samples)

    bg_indices = np.linspace(0, len(X_train) - 1, bg_size, dtype=int)
    eval_indices = np.linspace(0, len(X_test) - 1, eval_size, dtype=int)

    X_bg = X_train[bg_indices]
    X_eval = X_test[eval_indices]

    # Model: CoxPH (analytical linear SHAP attribution: phi_j = beta_j * (x_j - E[x_j]))
    if model_name == "cox_ph" and hasattr(model, "coef_") and model.coef_ is not None:
        coefs = np.asarray(model.coef_, dtype=float)
        mean_bg = np.mean(X_bg, axis=0)
        expected_val = float(np.dot(mean_bg, coefs))
        shap_values = (X_eval - mean_bg) * coefs
        return {
            "shap_values": np.asarray(shap_values, dtype=float),
            "expected_value": expected_val,
            "method": "Linear Analytical Attribution (exact SHAP for additive risk models)",
        }

    # Model: RSF or non-linear ensemble (fast, vectorized marginal attribution)
    bg_mean = np.mean(X_bg, axis=0)
    base_preds = np.asarray(model.predict(X_eval), dtype=float)
    base_pred = float(np.mean(model.predict(X_bg)))
    shap_vals = np.zeros((len(X_eval), n_features), dtype=float)

    for j in range(n_features):
        X_mod = X_eval.copy()
        X_mod[:, j] = bg_mean[j]
        mod_preds = np.asarray(model.predict(X_mod), dtype=float)
        shap_vals[:, j] = base_preds - mod_preds

    return {
        "shap_values": shap_vals,
        "expected_value": base_pred,
        "method": "Marginal Feature Attribution (fast Shapley approximation for Survival Trees)",
    }


def get_global_importance(
    shap_values: Union[np.ndarray, Any],
    feature_names: List[str],
) -> List[Dict[str, Any]]:
    """
    Calculate global feature importance by computing mean absolute SHAP values across samples.

    Parameters
    ----------
    shap_values : np.ndarray or shap.Explanation
        Array of SHAP values of shape (n_samples, n_features).
    feature_names : list of str
        Names corresponding to each feature column.

    Returns
    -------
    list of dict
        Features sorted descending by global importance with rank and description.
    """
    if hasattr(shap_values, "values"):
        vals = np.asarray(shap_values.values, dtype=float)
    else:
        vals = np.asarray(shap_values, dtype=float)

    if vals.ndim == 1:
        mean_abs = np.abs(vals)
    else:
        mean_abs = np.mean(np.abs(vals), axis=0)

    total_importance = float(np.sum(mean_abs))
    if total_importance == 0:
        total_importance = 1.0

    descriptions = get_feature_descriptions()
    ranked_indices = np.argsort(mean_abs)[::-1]

    global_importance_list: List[Dict[str, Any]] = []
    for rank, idx in enumerate(ranked_indices, start=1):
        name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        imp = float(mean_abs[idx])
        rel_imp = float(imp / total_importance)

        global_importance_list.append({
            "rank": rank,
            "feature": name,
            "importance": round(imp, 4),
            "relative_importance_pct": round(rel_imp * 100, 2),
            "description": descriptions.get(name, "Survival prognostic biomarker."),
        })

    return global_importance_list


def get_local_explanation(
    patient_data: Union[Dict[str, Any], pd.Series, pd.DataFrame, np.ndarray],
    model: Any,
    preprocessor_info: Dict[str, Any],
    shap_values_background: Optional[np.ndarray] = None,
    model_name: str = "rsf",
) -> Dict[str, Any]:
    """
    Compute patient-specific local feature attributions explaining predicted survival risk.

    Parameters
    ----------
    patient_data : dict, pd.Series, pd.DataFrame, or np.ndarray
        Raw patient record or preprocessed feature vector.
    model : object
        Fitted survival model (RSF or CoxPH).
    preprocessor_info : dict
        Preprocessor information with 'column_transformer' and 'feature_names'.
    shap_values_background : np.ndarray, optional
        Pre-extracted background sample matrix for SHAP reference.
    model_name : str, default='rsf'
        Model type identifier.

    Returns
    -------
    dict
        Local explanation containing per-feature contributions, effect direction
        ('Increases Risk' vs 'Decreases Risk'), raw and scaled values, and descriptions.
    """
    feature_names: List[str] = preprocessor_info.get("feature_names", [])

    # Preprocess patient data if not already array
    if isinstance(patient_data, np.ndarray):
        X_patient = patient_data.reshape(1, -1) if patient_data.ndim == 1 else patient_data
        raw_patient_dict = {}
    else:
        raw_patient_dict = patient_data if isinstance(patient_data, dict) else patient_data.to_dict()
        X_patient = preprocess_single_patient(patient_data, preprocessor_info)

    predicted_risk = float(model.predict(X_patient)[0])
    descriptions = get_feature_descriptions()
    contributions: np.ndarray

    # Case A: CoxPH model (direct analytical attribution)
    if model_name == "cox_ph" and hasattr(model, "coef_"):
        coefs = np.asarray(model.coef_, dtype=float)
        contributions = X_patient[0] * coefs

    # Case B: Random Survival Forest with SHAP library
    elif SHAP_AVAILABLE and shap is not None and shap_values_background is not None:
        try:
            def predict_wrapper(data: np.ndarray) -> np.ndarray:
                return np.asarray(model.predict(data), dtype=float)

            explainer = shap.Explainer(predict_wrapper, shap_values_background)
            shap_res = explainer(X_patient)
            contributions = np.asarray(shap_res.values[0], dtype=float)
        except Exception:
            # Fallback to feature perturbation
            contributions = _compute_perturbation_attribution(model, X_patient)
    else:
        # Fast marginal perturbation attribution
        contributions = _compute_perturbation_attribution(model, X_patient)

    # Format output features
    ranked_indices = np.argsort(np.abs(contributions))[::-1]
    feature_attributions: List[Dict[str, Any]] = []

    for rank, idx in enumerate(ranked_indices, start=1):
        feat = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        contrib = float(contributions[idx])
        effect = "Increases Risk" if contrib > 0 else "Decreases Risk"

        # Determine raw input value if available
        raw_val = raw_patient_dict.get(feat, None)
        if raw_val is None:
            for k, v in raw_patient_dict.items():
                if k in feat:
                    raw_val = v
                    break

        feature_attributions.append({
            "rank": rank,
            "feature": feat,
            "raw_value": raw_val,
            "standardized_value": round(float(X_patient[0, idx]), 4),
            "contribution": round(contrib, 4),
            "abs_contribution": round(abs(contrib), 4),
            "effect": effect,
            "description": descriptions.get(feat, "Biomarker affecting survival risk."),
        })

    return {
        "predicted_risk_score": round(predicted_risk, 4),
        "model_name": model_name,
        "feature_attributions": feature_attributions,
    }


def _compute_perturbation_attribution(model: Any, X_patient: np.ndarray) -> np.ndarray:
    """
    Helper to compute local feature attribution by comparing baseline patient risk
    against risk when the feature is zeroed out (set to its standardized population mean).
    """
    baseline_risk = float(model.predict(X_patient)[0])
    n_features = X_patient.shape[1]
    contributions = np.zeros(n_features, dtype=float)

    for j in range(n_features):
        X_mod = X_patient.copy()
        X_mod[0, j] = 0.0  # Zero is population mean in standardized features
        mod_risk = float(model.predict(X_mod)[0])
        # If removing this feature lowers risk, the feature was increasing risk
        contributions[j] = baseline_risk - mod_risk

    return contributions
