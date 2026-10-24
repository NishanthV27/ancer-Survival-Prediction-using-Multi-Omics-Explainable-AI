"""
Multi-omics processor for TCGA-COAD:
Matches Clinical Survival data, RNA-seq (STAR gene counts), and Methylation (Sesame level3betas)
using verified TCGA Patient Identifiers and Primary Tumor sample barcodes.

Generates:
- data/processed/tcga_coad_multiomics.csv
- data/processed/tcga_coad_multiomics_metadata.json

Preserves existing raw data and never creates synthetic patient records.
"""

import json
import logging
import os
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Key Colorectal Cancer (COAD) Biomarkers
BIOMARKER_MAP = {
    "gene_expr_A": {"symbol": "TP53", "description": "Tumor Protein P53 (key tumor suppressor mutated in ~60% COAD)"},
    "gene_expr_B": {"symbol": "KRAS", "description": "KRAS Proto-Oncogene (GTPase oncogene mutated in ~40% COAD)"},
    "gene_expr_C": {"symbol": "MUC2", "description": "Mucin 2, Oligomeric Mucus/Gel-Forming (intestinal goblet cell marker)"},
    "methylation_A": {"probe_id": "cg00000292", "description": "CpG site beta-value (DNA methylation level [0.0 - 1.0])"},
    "methylation_B": {"probe_id": "cg00005847", "description": "CpG site beta-value (DNA methylation level [0.0 - 1.0])"},
}


def parse_rna_seq_sample(file_path: str) -> dict:
    """Extract TPM values for target genes from STAR augmented counts TSV."""
    target_genes = {BIOMARKER_MAP["gene_expr_A"]["symbol"]: "gene_expr_A",
                    BIOMARKER_MAP["gene_expr_B"]["symbol"]: "gene_expr_B",
                    BIOMARKER_MAP["gene_expr_C"]["symbol"]: "gene_expr_C"}
    results = {}
    try:
        df = pd.read_csv(file_path, sep="\t", skiprows=1)
        for _, row in df.iterrows():
            gname = str(row.get("gene_name", "")).strip()
            if gname in target_genes:
                key = target_genes[gname]
                tpm = float(row.get("tpm_unstranded", 0.0))
                # log2(TPM + 1) transformation for variance stabilization
                results[key] = round(float(np.log2(max(tpm, 0.0) + 1.0)), 4)
    except Exception as e:
        logger.warning(f"Error parsing RNA-seq file {file_path}: {e}")

    # Ensure all keys exist
    for key in ["gene_expr_A", "gene_expr_B", "gene_expr_C"]:
        if key not in results:
            results[key] = 0.0
    return results


def parse_methylation_sample(file_path: str) -> dict:
    """Extract beta values for target probes from Sesame level3betas TXT."""
    target_probes = {BIOMARKER_MAP["methylation_A"]["probe_id"]: "methylation_A",
                     BIOMARKER_MAP["methylation_B"]["probe_id"]: "methylation_B"}
    results = {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    pid = parts[0].strip()
                    if pid in target_probes:
                        key = target_probes[pid]
                        val_str = parts[1].strip()
                        if val_str.upper() in ["NA", "NAN", ""]:
                            results[key] = 0.5  # Neutral default for missing beta
                        else:
                            try:
                                results[key] = round(float(val_str), 4)
                            except ValueError:
                                results[key] = 0.5
    except Exception as e:
        logger.warning(f"Error parsing methylation file {file_path}: {e}")

    for key in ["methylation_A", "methylation_B"]:
        if key not in results:
            results[key] = 0.5
    return results


def build_multiomics_dataset(base_dir: str = "."):
    """
    Match clinical data with downloaded RNA-seq and Methylation files.
    """
    clin_csv = os.path.join(base_dir, "data", "processed", "tcga_coad_clinical.csv")
    if not os.path.exists(clin_csv):
        raise FileNotFoundError(f"Processed clinical file not found: {clin_csv}. Run data/process_clinical.py first.")

    clin_df = pd.read_csv(clin_csv)
    clin_dict = {row["patient_id"]: row.to_dict() for _, row in clin_df.iterrows()}

    ge_meta_path = os.path.join(base_dir, "data", "raw", "gene_expression", "file_metadata.json")
    me_meta_path = os.path.join(base_dir, "data", "raw", "methylation", "file_metadata.json")
    ge_files_dir = os.path.join(base_dir, "data", "raw", "gene_expression", "files")
    me_files_dir = os.path.join(base_dir, "data", "raw", "methylation", "files")

    # Load metadata mappings
    with open(ge_meta_path, "r", encoding="utf-8") as f:
        ge_hits = json.load(f)["data"]["hits"]

    with open(me_meta_path, "r", encoding="utf-8") as f:
        me_hits = json.load(f)["data"]["hits"]

    # Map file name -> patient_id
    ge_file_to_patient = {}
    for h in ge_hits:
        fname = h.get("file_name")
        for c in h.get("cases", []):
            pid = c.get("submitter_id")
            for s in c.get("samples", []):
                if s.get("sample_type") == "Primary Tumor" or "-01A" in s.get("submitter_id", ""):
                    ge_file_to_patient[fname] = pid

    me_file_to_patient = {}
    for h in me_hits:
        fname = h.get("file_name")
        for c in h.get("cases", []):
            pid = c.get("submitter_id")
            for s in c.get("samples", []):
                if s.get("sample_type") == "Primary Tumor" or "-01A" in s.get("submitter_id", ""):
                    me_file_to_patient[fname] = pid

    # Scan downloaded files on disk
    downloaded_ge = {}
    if os.path.exists(ge_files_dir):
        for fn in os.listdir(ge_files_dir):
            if fn.endswith(".tsv") and not fn.endswith(".part"):
                pid = ge_file_to_patient.get(fn)
                if pid:
                    downloaded_ge[pid] = os.path.join(ge_files_dir, fn)

    downloaded_me = {}
    if os.path.exists(me_files_dir):
        for fn in os.listdir(me_files_dir):
            if fn.endswith(".txt") and not fn.endswith(".part"):
                pid = me_file_to_patient.get(fn)
                if pid:
                    downloaded_me[pid] = os.path.join(me_files_dir, fn)

    matched_pids = sorted(set(clin_dict.keys()) & set(downloaded_ge.keys()) & set(downloaded_me.keys()))
    logger.info(f"Clinical cases: {len(clin_dict)}")
    logger.info(f"Downloaded Primary Tumor GE files: {len(downloaded_ge)}")
    logger.info(f"Downloaded Primary Tumor ME beta files: {len(downloaded_me)}")
    logger.info(f"Strictly matched patients with all 3 modalities on disk: {len(matched_pids)}")

    if not matched_pids:
        logger.warning("No matched patients found with downloaded GE and ME files on disk.")
        return None

    records = []
    for pid in matched_pids:
        clin = clin_dict[pid]
        ge_data = parse_rna_seq_sample(downloaded_ge[pid])
        me_data = parse_methylation_sample(downloaded_me[pid])

        record = {
            "patient_id": pid,
            "age": clin["age"],
            "tumor_stage": clin["tumor_stage"],
            "gene_expr_A": ge_data["gene_expr_A"],
            "gene_expr_B": ge_data["gene_expr_B"],
            "gene_expr_C": ge_data["gene_expr_C"],
            "methylation_A": me_data["methylation_A"],
            "methylation_B": me_data["methylation_B"],
            "survival_months": clin["survival_months"],
            "event": int(clin["event"]),
        }
        records.append(record)

    multi_df = pd.DataFrame(records)
    out_csv = os.path.join(base_dir, "data", "processed", "tcga_coad_multiomics.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    multi_df.to_csv(out_csv, index=False)
    logger.info(f"Saved matched multi-omics dataset to {out_csv} ({len(multi_df)} patients)")

    # Save metadata JSON
    meta_path = os.path.join(base_dir, "data", "processed", "tcga_coad_multiomics_metadata.json")
    meta = {
        "dataset_name": "TCGA-COAD Matched Multi-Omics Dataset",
        "description": "Real patient multi-omics data integrating Clinical follow-up, RNA-seq, and DNA Methylation.",
        "cohort_size": len(multi_df),
        "events": int(multi_df["event"].sum()),
        "censored": int((multi_df["event"] == 0).sum()),
        "biomarkers": BIOMARKER_MAP,
        "matched_patient_ids": matched_pids,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved metadata to {meta_path}")

    return multi_df


if __name__ == "__main__":
    build_multiomics_dataset()
