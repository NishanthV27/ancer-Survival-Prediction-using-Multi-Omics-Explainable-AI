"""
Tests for ML preprocessing and risk prediction workflows.
"""

import unittest
import numpy as np
import pandas as pd
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
        cls.df = load_dataset("data/synthetic_tcga_coad_demo.csv")
        cls.models = load_models("models")
        cls.sample_patient = {
            "age": 62,
            "tumor_stage": "III",
            "gene_expr_A": 9.5,
            "gene_expr_B": 4.2,
            "gene_expr_C": 7.1,
            "methylation_A": 0.65,
            "methylation_B": 0.42,
        }

    def test_dataset_integrity(self):
        self.assertEqual(len(self.df), 300)
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
        self.assertEqual(len(X_tr), 240)
        self.assertEqual(len(X_te), 60)
        self.assertEqual(X_tr.shape[1], 9)
        self.assertIn("feature_names", info)
        self.assertIn("tumor_stage_II", info["feature_names"])
        self.assertIn("tumor_stage_III", info["feature_names"])
        self.assertIn("tumor_stage_IV", info["feature_names"])

    def test_single_patient_preprocessing(self):
        info = self.models["preprocessor_info"]
        x_patient = preprocess_single_patient(self.sample_patient, info)
        self.assertEqual(x_patient.shape, (1, 9))
        self.assertFalse(np.isnan(x_patient).any())

    def test_predict_risk_rsf(self):
        res = predict_risk(self.sample_patient, self.models, model_name="rsf")
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["model_name"], "rsf")
        self.assertIn(res["risk_category"], ["High Risk", "Low Risk"])
        self.assertIsInstance(res["risk_score"], float)
        self.assertIn("survival_probabilities", res)
        # Check milestones
        surv = res["survival_probabilities"]
        for m in ["12", "24", "36", "60"]:
            self.assertIn(m, surv)
            self.assertTrue(0.0 <= surv[m] <= 1.0)
        # Probabilities should decrease or stay constant over time
        self.assertGreaterEqual(surv["12"], surv["60"])

    def test_predict_risk_cox(self):
        res = predict_risk(self.sample_patient, self.models, model_name="cox_ph")
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["model_name"], "cox_ph")
        self.assertIn(res["risk_category"], ["High Risk", "Low Risk"])
        self.assertIsInstance(res["risk_score"], float)
        self.assertIn("survival_probabilities", res)
        surv = res["survival_probabilities"]
        for m in ["12", "24", "36", "60"]:
            self.assertIn(m, surv)
            self.assertTrue(0.0 <= surv[m] <= 1.0)


if __name__ == "__main__":
    unittest.main()

