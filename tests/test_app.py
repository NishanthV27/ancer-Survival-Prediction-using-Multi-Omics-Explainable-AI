"""
Integration tests for Flask application routes and API endpoints.
"""

import json
import unittest
from app import app


class TestFlaskApplication(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()

    def test_health_endpoint(self):
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["dataset"])
        self.assertTrue(data["models"])

    def test_pages_render_successfully(self):
        pages = ["/", "/dashboard", "/predict", "/analytics", "/explainability", "/dataset", "/about"]
        for page in pages:
            with self.subTest(page=page):
                resp = self.client.get(page)
                self.assertEqual(resp.status_code, 200)
                # Check for synthetic disclaimer in html
                self.assertIn(b"synthetic", resp.data.lower())

    def test_api_dashboard(self):
        resp = self.client.get("/api/dashboard")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("basic_stats", data)
        self.assertEqual(data["basic_stats"]["total_patients"], 300)
        self.assertIn("age_distribution", data)
        self.assertIn("tumor_stage_distribution", data)
        self.assertIn("gene_expression_stats", data)

    def test_api_predict_valid(self):
        payload = {
            "age": 62,
            "tumor_stage": "III",
            "gene_expr_A": 9.5,
            "gene_expr_B": 4.2,
            "gene_expr_C": 7.1,
            "methylation_A": 0.65,
            "methylation_B": 0.42,
            "model": "rsf",
        }
        resp = self.client.post("/api/predict", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn(data["risk_category"], ["High Risk", "Low Risk"])
        self.assertIn("survival_probabilities", data)

    def test_api_predict_validation(self):
        # Missing fields
        resp = self.client.post("/api/predict", json={"age": 60})
        self.assertEqual(resp.status_code, 400)
        # Invalid stage
        resp = self.client.post("/api/predict", json={
            "age": 60, "tumor_stage": "STAGE_X",
            "gene_expr_A": 5.0, "gene_expr_B": 3.0, "gene_expr_C": 5.0,
            "methylation_A": 0.5, "methylation_B": 0.5
        })
        self.assertEqual(resp.status_code, 400)

    def test_api_analytics(self):
        resp = self.client.get("/api/analytics")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["model_available"])
        self.assertIn("km_data", data)
        self.assertIn("groups", data["km_data"])
        self.assertIn("High Risk", data["km_data"]["groups"])
        self.assertIn("Low Risk", data["km_data"]["groups"])
        self.assertIn("km_by_stage", data)

    def test_api_explainability_get(self):
        resp = self.client.get("/api/explainability")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("feature_descriptions", data)
        self.assertIn("global_importance", data)
        self.assertIn("features", data["global_importance"])

    def test_api_explainability_post(self):
        payload = {
            "patient": {
                "age": 65,
                "tumor_stage": "IV",
                "gene_expr_A": 11.2,
                "gene_expr_B": 2.1,
                "gene_expr_C": 8.5,
                "methylation_A": 0.75,
                "methylation_B": 0.25,
            },
            "model": "rsf"
        }
        resp = self.client.post("/api/explainability", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("local_explanation", data)
        self.assertIn("features", data["local_explanation"])
        self.assertIn("contributions", data["local_explanation"])

    def test_api_dataset_pagination_and_filter(self):
        # Default pagination
        resp = self.client.get("/api/dataset?page=1&per_page=10")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(len(data["data"]), 10)
        self.assertEqual(data["pagination"]["total"], 300)

        # Stage filter
        resp_stage = self.client.get("/api/dataset?stage=I")
        self.assertEqual(resp_stage.status_code, 200)
        data_stage = resp_stage.get_json()
        self.assertTrue(all(r["tumor_stage"] == "I" for r in data_stage["data"]))

        # Search filter
        resp_search = self.client.get("/api/dataset?search=0001")
        self.assertEqual(resp_search.status_code, 200)
        data_search = resp_search.get_json()
        self.assertGreaterEqual(len(data_search["data"]), 1)

    def test_dataset_download(self):
        resp = self.client.get("/api/dataset/download")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"TCGA-COAD", resp.data)


if __name__ == "__main__":
    unittest.main()

