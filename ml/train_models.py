"""
Model training pipeline for Cancer Survival Prediction.

Trains Cox Proportional Hazards and Random Survival Forest models,
evaluates discrimination via Harrell's C-index, establishes risk stratification thresholds,
and serializes models, preprocessors, and metadata.
Supports scikit-survival where available, with native lifelines and log-rank survival tree fallbacks.
"""

from datetime import datetime, timezone
import json
import os
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd

from ml.preprocessing import (
    load_dataset,
    preprocess_data,
    save_preprocessor,
    load_preprocessor,
)
from ml.evaluation import evaluate_model, compare_models

# Check scikit-survival availability
try:
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.ensemble import RandomSurvivalForest
    SKSURV_AVAILABLE = True
    SKSURV_IMPORT_ERROR = None
except (ImportError, Exception) as err:
    CoxPHSurvivalAnalysis = None
    RandomSurvivalForest = None
    SKSURV_AVAILABLE = False
    SKSURV_IMPORT_ERROR = str(err)

# Lifelines Cox Proportional Hazards
try:
    from lifelines import CoxPHFitter
    LIFELINES_AVAILABLE = True
except (ImportError, Exception):
    CoxPHFitter = None
    LIFELINES_AVAILABLE = False


class SurvivalStepFunction:
    """
    Step function for survival curve S(t).
    Evaluates survival probability at any time milestone t.
    """
    def __init__(self, x_times: np.ndarray, y_probs: np.ndarray):
        self.x = np.asarray(x_times, dtype=float)
        self.y = np.asarray(y_probs, dtype=float)

    def __call__(self, t: float) -> float:
        if len(self.x) == 0:
            return 0.5
        if t < self.x[0]:
            return 1.0
        idx = np.searchsorted(self.x, t, side="right") - 1
        if idx >= len(self.y):
            idx = len(self.y) - 1
        return float(self.y[idx])


class SurvivalTreeNode:
    """
    Node in a Survival Tree using log-rank splitting (Ishwaran et al., 2008).
    Computes Nelson-Aalen cumulative hazard estimators in terminal leaves.
    """
    def __init__(
        self,
        depth: int = 0,
        max_depth: int = 8,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
    ):
        self.depth = depth
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.is_leaf = False
        self.feature_idx: Optional[int] = None
        self.threshold: Optional[float] = None
        self.left: Optional["SurvivalTreeNode"] = None
        self.right: Optional["SurvivalTreeNode"] = None
        self.timeline: np.ndarray = np.array([])
        self.cum_hazard: np.ndarray = np.array([])
        self.mortality: float = 0.0

    def fit(self, X: np.ndarray, times: np.ndarray, events: np.ndarray, rng: np.random.RandomState, mtry: int) -> "SurvivalTreeNode":
        n_samples = len(times)
        n_events = int(np.sum(events))

        if (self.depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            n_events < 2):
            self._make_leaf(times, events)
            return self

        n_features = X.shape[1]
        features_to_try = rng.choice(n_features, size=min(mtry, n_features), replace=False)

        best_score = 0.0
        best_feat = None
        best_thresh = None

        for feat in features_to_try:
            x_vals = X[:, feat]
            uniques = np.unique(x_vals)
            if len(uniques) < 2:
                continue

            if len(uniques) > 10:
                percentiles = np.linspace(10, 90, 8)
                thresholds = np.percentile(x_vals, percentiles)
            else:
                thresholds = (uniques[:-1] + uniques[1:]) / 2.0

            for thresh in thresholds:
                left_mask = x_vals <= thresh
                right_mask = ~left_mask
                n_left = int(np.sum(left_mask))
                n_right = int(np.sum(right_mask))

                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                if np.sum(events[left_mask]) == 0 or np.sum(events[right_mask]) == 0:
                    continue

                t_l, e_l = times[left_mask], events[left_mask]
                t_r, e_r = times[right_mask], events[right_mask]

                all_times = np.unique(times[left_mask | right_mask])
                O1, E1, V = 0.0, 0.0, 0.0
                for t in all_times:
                    y1 = np.sum(t_l >= t)
                    y0 = np.sum(t_r >= t)
                    y = y1 + y0
                    if y <= 1:
                        continue
                    d1 = np.sum((t_l == t) & e_l)
                    d0 = np.sum((t_r == t) & e_r)
                    d = d1 + d0
                    O1 += d1
                    E1 += y1 * (d / y)
                    V += (y1 * y0 * d * (y - d)) / (y**2 * (y - 1)) if y > 1 else 0.0

                score = abs(O1 - E1) / np.sqrt(V) if V > 1e-8 else 0.0
                if score > best_score:
                    best_score = score
                    best_feat = feat
                    best_thresh = thresh

        if best_feat is None or best_score < 1e-5:
            self._make_leaf(times, events)
            return self

        self.feature_idx = best_feat
        self.threshold = float(best_thresh)
        left_mask = X[:, best_feat] <= best_thresh

        self.left = SurvivalTreeNode(self.depth + 1, self.max_depth, self.min_samples_split, self.min_samples_leaf)
        self.left.fit(X[left_mask], times[left_mask], events[left_mask], rng, mtry)

        self.right = SurvivalTreeNode(self.depth + 1, self.max_depth, self.min_samples_split, self.min_samples_leaf)
        self.right.fit(X[~left_mask], times[~left_mask], events[~left_mask], rng, mtry)
        return self

    def _make_leaf(self, times: np.ndarray, events: np.ndarray) -> None:
        self.is_leaf = True
        unique_times = np.sort(np.unique(times))
        chf = []
        cum = 0.0
        for t in unique_times:
            d = np.sum((times == t) & events)
            y = np.sum(times >= t)
            if y > 0:
                cum += d / y
            chf.append(cum)

        self.timeline = unique_times
        self.cum_hazard = np.array(chf, dtype=float)
        self.mortality = float(cum)

    def predict_leaf(self, x_single: np.ndarray) -> "SurvivalTreeNode":
        if self.is_leaf:
            return self
        if x_single[self.feature_idx] <= self.threshold:
            return self.left.predict_leaf(x_single)
        else:
            return self.right.predict_leaf(x_single)


class NativeRandomSurvivalForest:
    """
    Random Survival Forest implementation based on Ishwaran et al. (2008).
    Uses log-rank test splitting criteria and Nelson-Aalen cumulative hazard estimators.
    """
    def __init__(
        self,
        n_estimators: int = 50,
        max_depth: int = 6,
        min_samples_split: int = 8,
        min_samples_leaf: int = 4,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.trees: List[SurvivalTreeNode] = []
        self.global_times_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NativeRandomSurvivalForest":
        rng = np.random.RandomState(self.random_state)
        times = y["survival_months"].astype(float)
        events = y["event"].astype(bool)
        n_samples, n_features = X.shape
        mtry = int(np.ceil(np.sqrt(n_features)))
        self.global_times_ = np.sort(np.unique(times))

        self.trees = []
        for _ in range(self.n_estimators):
            boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
            X_b = X[boot_idx]
            t_b = times[boot_idx]
            e_b = events[boot_idx]

            tree = SurvivalTreeNode(
                depth=0,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
            )
            tree.fit(X_b, t_b, e_b, rng, mtry)
            self.trees.append(tree)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict continuous ensemble mortality risk score for each sample."""
        n_samples = len(X)
        mortalities = np.zeros(n_samples, dtype=float)
        for i in range(n_samples):
            x = X[i]
            m_sum = sum(tree.predict_leaf(x).mortality for tree in self.trees)
            mortalities[i] = m_sum / len(self.trees)
        return mortalities

    def predict_survival_function(self, X: np.ndarray) -> List[SurvivalStepFunction]:
        """Predict patient-specific survival curves S(t) = exp(-H(t))."""
        n_samples = len(X)
        funcs = []
        for i in range(n_samples):
            x = X[i]
            all_chfs = []
            for tree in self.trees:
                leaf = tree.predict_leaf(x)
                if len(leaf.timeline) == 0:
                    chf_interp = np.zeros_like(self.global_times_)
                else:
                    idx = np.searchsorted(leaf.timeline, self.global_times_, side="right") - 1
                    valid = idx >= 0
                    chf_interp = np.zeros_like(self.global_times_)
                    chf_interp[valid] = leaf.cum_hazard[idx[valid]]
                all_chfs.append(chf_interp)

            mean_chf = np.mean(all_chfs, axis=0)
            surv = np.exp(-mean_chf)
            funcs.append(SurvivalStepFunction(self.global_times_, surv))
        return funcs


class LifelinesCoxPHWrapper:
    """
    Adapter wrapping lifelines.CoxPHFitter to present a standard scikit-style API:
    - .fit(X, y)
    - .predict(X) returning log-hazard risk scores
    - .coef_ exposing regression coefficients
    - .predict_survival_function(X) returning survival step functions
    """
    def __init__(self, alpha: float = 0.1, feature_names: Optional[List[str]] = None):
        self.alpha = alpha
        self.feature_names = feature_names or []
        self.fitter: Optional[CoxPHFitter] = None
        self.coef_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> "LifelinesCoxPHWrapper":
        if feature_names is not None:
            self.feature_names = list(feature_names)
        elif not self.feature_names:
            self.feature_names = [f"f_{i}" for i in range(X.shape[1])]

        if not LIFELINES_AVAILABLE or CoxPHFitter is None:
            raise ImportError("lifelines is required for LifelinesCoxPHWrapper.")

        self.fitter = CoxPHFitter(penalizer=self.alpha)
        df = pd.DataFrame(X, columns=self.feature_names)
        df["survival_months"] = y["survival_months"].astype(float)
        df["event"] = y["event"].astype(int)

        self.fitter.fit(df, duration_col="survival_months", event_col="event")
        self.coef_ = np.asarray(self.fitter.params_.values, dtype=float)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict log-partial hazard risk score (X @ beta)."""
        df = pd.DataFrame(X, columns=self.feature_names)
        return np.asarray(self.fitter.predict_log_partial_hazard(df).values, dtype=float)

    def predict_survival_function(self, X: np.ndarray) -> List[SurvivalStepFunction]:
        """Predict survival function S(t) for each individual."""
        df = pd.DataFrame(X, columns=self.feature_names)
        surv_df = self.fitter.predict_survival_function(df)
        timeline = np.asarray(surv_df.index.values, dtype=float)
        funcs = []
        for col in surv_df.columns:
            probs = np.asarray(surv_df[col].values, dtype=float)
            funcs.append(SurvivalStepFunction(timeline, probs))
        return funcs


def train_cox_ph(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float = 0.1,
    feature_names: Optional[List[str]] = None,
) -> Any:
    """Train Cox Proportional Hazards survival analysis model."""
    if SKSURV_AVAILABLE and CoxPHSurvivalAnalysis is not None:
        try:
            model = CoxPHSurvivalAnalysis(alpha=alpha)
            model.fit(X_train, y_train)
            return model
        except Exception:
            pass

    # Use lifelines CoxPH wrapper
    if LIFELINES_AVAILABLE and CoxPHFitter is not None:
        wrapper = LifelinesCoxPHWrapper(alpha=alpha, feature_names=feature_names)
        wrapper.fit(X_train, y_train, feature_names=feature_names)
        return wrapper

    raise RuntimeError(
        "Neither scikit-survival nor lifelines is functional in this environment. "
        "Please install lifelines using: pip install lifelines"
    )


def train_rsf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_seed: int = 42,
    n_estimators: int = 50,
    min_samples_split: int = 8,
    min_samples_leaf: int = 4,
) -> Any:
    """Train Random Survival Forest model."""
    if SKSURV_AVAILABLE and RandomSurvivalForest is not None:
        try:
            model = RandomSurvivalForest(
                n_estimators=n_estimators,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=random_seed,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)
            return model
        except Exception:
            pass

    # Native implementation using log-rank survival trees
    native_rsf = NativeRandomSurvivalForest(
        n_estimators=n_estimators,
        max_depth=6,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_seed,
    )
    native_rsf.fit(X_train, y_train)
    return native_rsf


def train_and_save_all(
    data_path: str,
    models_dir: str,
    random_seed: int = 42,
    test_size: float = 0.2,
) -> Dict[str, Any]:
    """Execute end-to-end survival ML pipeline and serialize all artifacts."""
    os.makedirs(models_dir, exist_ok=True)

    # 1. Load dataset
    df = load_dataset(data_path)

    # 2. Preprocess data
    X_train, X_test, y_train, y_test, preprocessor_info = preprocess_data(
        df=df,
        random_seed=random_seed,
        test_size=test_size,
    )

    feature_names = preprocessor_info["feature_names"]

    # 3. Train models
    cox_model = train_cox_ph(X_train, y_train, alpha=0.1, feature_names=feature_names)
    rsf_model = train_rsf(
        X_train,
        y_train,
        random_seed=random_seed,
        n_estimators=50,
        min_samples_split=8,
        min_samples_leaf=4,
    )

    # 4. Evaluate models on test split
    comparison = compare_models(
        models_dict={"rsf": rsf_model, "cox_ph": cox_model},
        X_test=X_test,
        y_test=y_test,
    )

    # 5. Compute risk score distributions & stratification thresholds (median of training risk)
    rsf_train_risks = rsf_model.predict(X_train)
    cox_train_risks = cox_model.predict(X_train)

    rsf_threshold = float(np.median(rsf_train_risks))
    cox_threshold = float(np.median(cox_train_risks))

    best_model_name = comparison["best_model"] or "rsf"
    primary_threshold = rsf_threshold if best_model_name == "rsf" else cox_threshold

    # 6. Build metadata
    training_timestamp = datetime.now(timezone.utc).isoformat()

    metadata: Dict[str, Any] = {
        "training_date": training_timestamp,
        "best_model": best_model_name,
        "risk_threshold": round(primary_threshold, 6),
        "thresholds": {
            "rsf": round(rsf_threshold, 6),
            "cox_ph": round(cox_threshold, 6),
        },
        "metrics": comparison["models"],
        "best_c_index": comparison["best_c_index"],
        "feature_names": preprocessor_info["feature_names"],
        "numeric_features": preprocessor_info["numeric_features"],
        "categorical_features": preprocessor_info["categorical_features"],
        "dataset_statistics": {
            "total_samples": int(len(df)),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "overall_event_rate": round(float(df["event"].mean()), 4),
            "median_survival_months": round(float(df["survival_months"].median()), 2),
            "min_survival_months": round(float(df["survival_months"].min()), 2),
            "max_survival_months": round(float(df["survival_months"].max()), 2),
        },
        "model_hyperparameters": {
            "cox_ph": {"alpha": 0.1, "engine": "lifelines" if isinstance(cox_model, LifelinesCoxPHWrapper) else "sksurv"},
            "rsf": {
                "n_estimators": 50,
                "min_samples_split": 8,
                "min_samples_leaf": 4,
                "random_state": random_seed,
                "engine": "native_logrank" if isinstance(rsf_model, NativeRandomSurvivalForest) else "sksurv",
            },
        },
        "random_seed": random_seed,
        "disclaimer": (
            "This model was trained on synthetic cancer survival data for research and "
            "educational demonstration. It must not be used for actual medical diagnosis, "
            "clinical treatment planning, or patient prognosis."
        ),
    }

    # 7. Serialize artifacts
    cox_path = os.path.join(models_dir, "cox_ph_model.joblib")
    rsf_path = os.path.join(models_dir, "rsf_model.joblib")
    preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
    metadata_joblib_path = os.path.join(models_dir, "metadata.joblib")
    metadata_json_path = os.path.join(models_dir, "metadata.json")

    joblib.dump(cox_model, cox_path)
    joblib.dump(rsf_model, rsf_path)
    save_preprocessor(preprocessor_info, preprocessor_path)
    joblib.dump(metadata, metadata_joblib_path)

    with open(metadata_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "status": "success",
        "models_dir": models_dir,
        "best_model": best_model_name,
        "best_c_index": comparison["best_c_index"],
        "risk_threshold": primary_threshold,
        "thresholds": metadata["thresholds"],
        "metrics": comparison["models"],
        "artifacts": {
            "cox_ph": cox_path,
            "rsf": rsf_path,
            "preprocessor": preprocessor_path,
            "metadata_joblib": metadata_joblib_path,
            "metadata_json": metadata_json_path,
        },
    }


def load_models(models_dir: str) -> Dict[str, Any]:
    """Load saved models, preprocessor, and metadata from disk."""
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"Models directory does not exist: {models_dir}")

    cox_path = os.path.join(models_dir, "cox_ph_model.joblib")
    rsf_path = os.path.join(models_dir, "rsf_model.joblib")
    preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
    metadata_path = os.path.join(models_dir, "metadata.joblib")

    for path, desc in [
        (cox_path, "Cox PH model"),
        (rsf_path, "Random Survival Forest model"),
        (preprocessor_path, "Preprocessor"),
        (metadata_path, "Metadata"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required artifact '{desc}' at: {path}")

    cox_model = joblib.load(cox_path)
    rsf_model = joblib.load(rsf_path)
    preprocessor_info = load_preprocessor(preprocessor_path)
    metadata = joblib.load(metadata_path)

    return {
        "cox_ph": cox_model,
        "rsf": rsf_model,
        "preprocessor_info": preprocessor_info,
        "metadata": metadata,
        "trained_models": {"rsf": rsf_model, "cox_ph": cox_model},
    }
