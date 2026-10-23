"""
Data loading and preprocessing module for Cancer Survival Analysis.

Handles loading clinical and molecular datasets, data validation,
stratified train/test splitting, feature encoding, scaling, and
conversion to survival structured target arrays.
Designed to be robust and fully compatible without relying on blocked C extensions.
"""

import os
from typing import Any, Dict, List, Tuple, Union
import numpy as np
import pandas as pd
import joblib

# Canonical column definitions
REQUIRED_COLUMNS: List[str] = [
    "patient_id",
    "age",
    "tumor_stage",
    "gene_expr_A",
    "gene_expr_B",
    "gene_expr_C",
    "methylation_A",
    "methylation_B",
    "survival_months",
    "event",
]

NUMERIC_FEATURES: List[str] = [
    "age",
    "gene_expr_A",
    "gene_expr_B",
    "gene_expr_C",
    "methylation_A",
    "methylation_B",
]

CATEGORICAL_FEATURES: List[str] = [
    "tumor_stage",
]

TARGET_COLUMNS: List[str] = [
    "event",
    "survival_months",
]


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load a cancer survival dataset from a CSV file and perform rigorous validation.

    Checks performed:
    1. File existence.
    2. Required columns existence.
    3. Missing (NaN / null) values in required columns.
    4. Invalid survival times (survival_months <= 0).
    5. Duplicate patient IDs or identical rows.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'. Please verify the file is placed in the data directory."
        )

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise ValueError(f"Failed to parse CSV file '{filepath}': {str(e)}")

    if df.empty:
        raise ValueError("The loaded dataset is empty (0 records).")

    # Column existence
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise KeyError(
            f"Dataset is missing mandatory column(s): {missing_cols}. "
            f"Required columns: {REQUIRED_COLUMNS}"
        )

    # Missing values
    null_counts = df[REQUIRED_COLUMNS].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0].to_dict()
    if cols_with_nulls:
        raise ValueError(
            f"Dataset contains null/missing values in required columns: {cols_with_nulls}. "
            "Data imputation or cleanup is required."
        )

    # Positive survival times
    df["survival_months"] = pd.to_numeric(df["survival_months"], errors="coerce")
    if (df["survival_months"] <= 0).any():
        invalid_count = int((df["survival_months"] <= 0).sum())
        raise ValueError(
            f"Found {invalid_count} records with invalid non-positive survival_months (<= 0)."
        )

    # Event indicator validation
    valid_events = {0, 1, 0.0, 1.0, True, False, "0", "1"}
    unrecognized = set(df["event"].unique()) - valid_events
    if unrecognized:
        raise ValueError(
            f"Event column contains invalid values: {unrecognized}. Allowed: 0 (censored), 1 (event observed)."
        )

    # Check duplicates
    if df["patient_id"].duplicated().any():
        dup_count = int(df["patient_id"].duplicated().sum())
        raise ValueError(f"Dataset contains {dup_count} duplicate patient_id entries.")

    return df


def _to_bool(val: Any) -> bool:
    """Safely convert diverse representations of binary event flags to boolean."""
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (int, np.integer, float, np.floating)):
        return bool(val != 0)
    if isinstance(val, str):
        return val.strip().lower() in ("1", "true", "t", "yes", "y")
    return bool(val)


def create_survival_target(
    events: Union[pd.Series, np.ndarray, List[Any]],
    survival_months: Union[pd.Series, np.ndarray, List[Any]],
) -> np.ndarray:
    """
    Construct a 1D structured numpy array for survival compatibility.
    Structured array with dtype [('event', bool), ('survival_months', float)].
    """
    event_bool = np.array([_to_bool(e) for e in events], dtype=bool)
    survival_float = np.array(survival_months, dtype=float)

    target = np.array(
        list(zip(event_bool, survival_float)),
        dtype=[("event", bool), ("survival_months", float)],
    )
    return target


class SurvivalFeatureTransformer:
    """
    Pure Python/NumPy feature preprocessor for cancer survival models.
    Performs z-score normalization on numeric features and drop-first one-hot
    encoding for categorical features without external dependencies.
    """
    def __init__(self, numeric_features: List[str], categorical_features: List[str]):
        self.numeric_features = list(numeric_features)
        self.categorical_features = list(categorical_features)
        self.means_: Dict[str, float] = {}
        self.stds_: Dict[str, float] = {}
        self.categories_: Dict[str, List[str]] = {}
        self.feature_names_: List[str] = []

    def fit(self, X_df: pd.DataFrame) -> "SurvivalFeatureTransformer":
        self.means_ = {}
        self.stds_ = {}
        for col in self.numeric_features:
            vals = pd.to_numeric(X_df[col], errors="coerce").values
            m = float(np.nanmean(vals))
            s = float(np.nanstd(vals))
            self.means_[col] = m
            self.stds_[col] = s if s > 1e-8 else 1.0

        self.categories_ = {}
        for col in self.categorical_features:
            unique_cats = sorted(list(X_df[col].astype(str).unique()))
            self.categories_[col] = unique_cats

        # Feature names with drop='first' on categorical
        names = list(self.numeric_features)
        for col in self.categorical_features:
            for cat in self.categories_[col][1:]:
                names.append(f"{col}_{cat}")
        self.feature_names_ = names
        return self

    def transform(self, X_df: pd.DataFrame) -> np.ndarray:
        n_samples = len(X_df)
        matrix = np.zeros((n_samples, len(self.feature_names_)), dtype=np.float64)

        # Standardize numeric features
        for i, col in enumerate(self.numeric_features):
            vals = pd.to_numeric(X_df[col], errors="coerce").fillna(self.means_[col]).values
            matrix[:, i] = (vals - self.means_[col]) / self.stds_[col]

        # One-hot encode categorical features (drop first category)
        curr_idx = len(self.numeric_features)
        for col in self.categorical_features:
            cats = self.categories_.get(col, [])
            for cat in cats[1:]:
                matrix[:, curr_idx] = (X_df[col].astype(str).values == str(cat)).astype(np.float64)
                curr_idx += 1

        return matrix

    def fit_transform(self, X_df: pd.DataFrame) -> np.ndarray:
        return self.fit(X_df).transform(X_df)

    def get_feature_names_out(self) -> List[str]:
        return list(self.feature_names_)


def stratified_survival_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform stratified train/test split on event indicator preserving censoring ratio.
    """
    rng = np.random.RandomState(random_seed)
    event_mask = (df["event"].values == 1)
    event_indices = np.where(event_mask)[0]
    censored_indices = np.where(~event_mask)[0]

    rng.shuffle(event_indices)
    rng.shuffle(censored_indices)

    n_test_events = int(round(len(event_indices) * test_size))
    n_test_censored = int(round(len(censored_indices) * test_size))

    test_idx = np.concatenate([
        event_indices[:n_test_events],
        censored_indices[:n_test_censored],
    ])
    train_idx = np.concatenate([
        event_indices[n_test_events:],
        censored_indices[n_test_censored:],
    ])

    rng.shuffle(test_idx)
    rng.shuffle(train_idx)
    return train_idx, test_idx


def preprocess_data(
    df: pd.DataFrame,
    random_seed: int = 42,
    test_size: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Preprocess clinical cancer dataset into stratified train and test splits,
    one-hot encode tumor stage, standardize numeric features, and prepare
    survival structured targets.
    """
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X_raw = df[feature_cols].copy()
    y = create_survival_target(df["event"], df["survival_months"])

    train_indices, test_indices = stratified_survival_split(
        df=df,
        test_size=test_size,
        random_seed=random_seed,
    )

    X_train_raw = X_raw.iloc[train_indices].reset_index(drop=True)
    X_test_raw = X_raw.iloc[test_indices].reset_index(drop=True)
    y_train = y[train_indices]
    y_test = y[test_indices]

    transformer = SurvivalFeatureTransformer(
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    )

    X_train = transformer.fit_transform(X_train_raw)
    X_test = transformer.transform(X_test_raw)
    feature_names = transformer.get_feature_names_out()

    preprocessor_info: Dict[str, Any] = {
        "column_transformer": transformer,
        "transformer": transformer,
        "feature_names": feature_names,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "random_seed": random_seed,
        "test_size": test_size,
    }

    return X_train, X_test, y_train, y_test, preprocessor_info


def preprocess_single_patient(
    patient_data: Union[Dict[str, Any], pd.Series, pd.DataFrame],
    preprocessor_info: Dict[str, Any],
) -> np.ndarray:
    """
    Transform a single patient record into a model-ready 2D feature array.
    """
    transformer = preprocessor_info.get("transformer") or preprocessor_info.get("column_transformer")
    if transformer is None:
        raise ValueError("preprocessor_info must contain 'transformer' or 'column_transformer'.")

    required_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    if isinstance(patient_data, dict):
        patient_df = pd.DataFrame([patient_data])
    elif isinstance(patient_data, pd.Series):
        patient_df = pd.DataFrame([patient_data.to_dict()])
    elif isinstance(patient_data, pd.DataFrame):
        patient_df = patient_data.copy()
    else:
        raise TypeError(
            f"Unsupported patient_data type: {type(patient_data)}. Expected dict, Series, or DataFrame."
        )

    missing = [feat for feat in required_features if feat not in patient_df.columns]
    if missing:
        raise ValueError(
            f"Patient data is missing required feature(s): {missing}. Expected: {required_features}"
        )

    aligned_df = patient_df[required_features].copy()
    for col in NUMERIC_FEATURES:
        aligned_df[col] = pd.to_numeric(aligned_df[col], errors="raise")

    X_patient = transformer.transform(aligned_df)
    return np.asarray(X_patient, dtype=np.float64)


def save_preprocessor(preprocessor_info: Dict[str, Any], filepath: str) -> None:
    """Save preprocessor information dictionary to disk using joblib."""
    dir_name = os.path.dirname(filepath)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    joblib.dump(preprocessor_info, filepath)


def load_preprocessor(filepath: str) -> Dict[str, Any]:
    """Load preprocessor information dictionary from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Preprocessor file not found at: {filepath}")
    loaded = joblib.load(filepath)
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected dict from preprocessor file, but got {type(loaded)}")
    return loaded
