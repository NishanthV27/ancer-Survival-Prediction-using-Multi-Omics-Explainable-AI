"""
Tests for ML preprocessing and risk prediction workflows.
"""

import unittest
import numpy as np
import pandas as pd
from config import Config
from ml.preprocessing import (
    load_dataset,
    preprocess_data,
    preprocess_single_patient,
    SurvivalFeatureTransformer,
)
from ml.train_models import load_models
from ml.prediction import predict_risk


class TestPredictionWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_dataset(Config.DATASET_PATH)
        cls.models = load_models(Config.MODELS_DIR)
        cls.sample_patient = {
            "age": 62,
            "tumor_stage": "III",
            "gene_expr_A": 6.5,
            "gene_expr_B": 4.2,
            "gene_expr_C": 7.1,
            "methylation_A": 0.65,
            "methylation_B": 0.42,
        }

    def test_dataset_integrity(self):
        self.assertGreater(len(self.df), 0)
        required_cols = [
            "patient_id", "age", "tumor_stage", "gene_expr_A",
            "gene_expr_B", "gene_expr_C", "methylation_A",
            "methylation_B", "survival_months", "event"
        ]
        for col in required_cols:
            self.assertIn(col, self.df.columns)
        self.assertFalse(self.df["patient_id"].duplicated().any())
        self.assertTrue((self.df["survival_months"] > 0).all())

    def test_preprocessing(self):
        X_tr, X_te, y_tr, y_te, info = preprocess_data(self.df, test_size=0.2, random_seed=42)
        self.assertGreater(len(X_tr), 0)
        self.assertGreater(len(X_te), 0)
        self.assertIn("feature_names", info)

    def test_single_patient_preprocessing(self):
        info = self.models["preprocessor_info"]
        x_patient = preprocess_single_patient(self.sample_patient, info)
        self.assertEqual(x_patient.shape[0], 1)
        self.assertFalse(np.isnan(x_patient).any())

    def test_predict_risk_rsf(self):
        result = predict_risk(self.sample_patient, self.models, model_name="rsf")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["model_name"], "rsf")
        self.assertIn("risk_score", result)
        self.assertIn("risk_category", result)
        self.assertIn(result["risk_category"], ["High Risk", "Low Risk"])
        self.assertIn("survival_probabilities", result)

    def test_predict_risk_cox(self):
        result = predict_risk(self.sample_patient, self.models, model_name="cox_ph")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["model_name"], "cox_ph")
        self.assertIn("risk_score", result)
        self.assertIn("risk_category", result)


if __name__ == "__main__":
    unittest.main()
