"""
Tests for TCGA-COAD raw data processing and multi-omics matching.
"""

import os
import unittest
import pandas as pd
from data.process_clinical import normalize_stage, OUTPUT_CLINICAL_PATH
from data.process_multiomics import BIOMARKER_MAP
from data.download_gdc_data import verify_md5


class TestDataProcessing(unittest.TestCase):
    def test_normalize_stage(self):
        self.assertEqual(normalize_stage("Stage I"), "Stage I")
        self.assertEqual(normalize_stage("Stage IA"), "Stage I")
        self.assertEqual(normalize_stage("Stage IIA"), "Stage II")
        self.assertEqual(normalize_stage("Stage IIIB"), "Stage III")
        self.assertEqual(normalize_stage("Stage IVB"), "Stage IV")
        self.assertIsNone(normalize_stage("not reported"))
        self.assertIsNone(normalize_stage(None))

    def test_clinical_processed_data(self):
        if os.path.exists(OUTPUT_CLINICAL_PATH):
            df = pd.read_csv(OUTPUT_CLINICAL_PATH)
            self.assertGreater(len(df), 300)
            self.assertIn("patient_id", df.columns)
            self.assertIn("age", df.columns)
            self.assertIn("tumor_stage", df.columns)
            self.assertIn("survival_months", df.columns)
            self.assertIn("event", df.columns)
            self.assertTrue((df["survival_months"] > 0).all())
            self.assertTrue(set(df["event"].unique()).issubset({0, 1}))

    def test_multiomics_biomarkers(self):
        self.assertIn("gene_expr_A", BIOMARKER_MAP)
        self.assertIn("gene_expr_B", BIOMARKER_MAP)
        self.assertIn("gene_expr_C", BIOMARKER_MAP)
        self.assertIn("methylation_A", BIOMARKER_MAP)
        self.assertIn("methylation_B", BIOMARKER_MAP)
        self.assertEqual(BIOMARKER_MAP["gene_expr_A"]["symbol"], "TP53")
        self.assertEqual(BIOMARKER_MAP["gene_expr_B"]["symbol"], "KRAS")
        self.assertEqual(BIOMARKER_MAP["gene_expr_C"]["symbol"], "MUC2")


if __name__ == "__main__":
    unittest.main()
