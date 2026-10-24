"""
Process TCGA-COAD clinical data from raw JSON into structured clinical survival dataset.
Extracts patient_id, age, tumor_stage, gender, vital_status, survival_months, and event.
"""

import json
import logging
import os
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RAW_CLINICAL_PATH = os.path.join("data", "raw", "clinical", "tcga_coad_cases.json")
OUTPUT_CLINICAL_PATH = os.path.join("data", "processed", "tcga_coad_clinical.csv")

def normalize_stage(stage_str):
    """Normalize AJCC pathologic stages to Stage I, II, III, IV."""
    if not stage_str or not isinstance(stage_str, str):
        return None
    s = stage_str.strip().upper()
    if s in ["NOT REPORTED", "UNKNOWN", "STAGE X", "DISCREPANCY"]:
        return None
    if "IV" in s:
        return "Stage IV"
    if "III" in s:
        return "Stage III"
    if "II" in s:
        return "Stage II"
    if "I" in s:
        return "Stage I"
    return None

def process_clinical_cases(raw_path=RAW_CLINICAL_PATH, out_path=OUTPUT_CLINICAL_PATH):
    """Parse raw GDC clinical cases JSON and produce verified clinical survival dataset."""
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw clinical file not found: {raw_path}")

    logger.info(f"Loading raw clinical cases from {raw_path}...")
    with open(raw_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    hits = payload.get("data", {}).get("hits", [])
    if not hits and isinstance(payload, list):
        hits = payload

    logger.info(f"Total raw clinical cases in JSON: {len(hits)}")

    records = []
    skipped_reasons = {}

    for case in hits:
        pid = case.get("submitter_id")
        if not pid:
            skipped_reasons["missing_patient_id"] = skipped_reasons.get("missing_patient_id", 0) + 1
            continue

        diagnoses = case.get("diagnoses", [])
        primary_diag = diagnoses[0] if diagnoses else {}
        demographic = case.get("demographic", {})

        # Age in years at diagnosis
        age_days = primary_diag.get("age_at_diagnosis")
        if age_days is None:
            age_days = demographic.get("age_at_index")
            if age_days is not None:
                age_days = age_days * 365.25

        if age_days is None or age_days <= 0:
            skipped_reasons["invalid_or_missing_age"] = skipped_reasons.get("invalid_or_missing_age", 0) + 1
            continue
        age_years = round(age_days / 365.25, 1)

        # Pathologic Tumor Stage
        raw_stage = primary_diag.get("ajcc_pathologic_stage")
        norm_stage = normalize_stage(raw_stage)
        if not norm_stage:
            skipped_reasons["missing_or_unmapped_stage"] = skipped_reasons.get("missing_or_unmapped_stage", 0) + 1
            continue

        # Vital status & survival time
        vital_status = demographic.get("vital_status") or primary_diag.get("vital_status")
        if not vital_status or vital_status.lower() not in ["alive", "dead"]:
            skipped_reasons["missing_vital_status"] = skipped_reasons.get("missing_vital_status", 0) + 1
            continue

        days_death = demographic.get("days_to_death") or primary_diag.get("days_to_death")
        days_followup = primary_diag.get("days_to_last_follow_up")

        # Check follow-ups array if missing
        if days_followup is None:
            for fu in case.get("follow_ups", []):
                fu_days = fu.get("days_to_follow_up")
                if fu_days is not None:
                    days_followup = max(days_followup or 0, fu_days)

        event = 1 if vital_status.lower() == "dead" else 0
        if event == 1:
            time_days = days_death if days_death is not None else days_followup
        else:
            time_days = days_followup

        if time_days is None or time_days < 0:
            skipped_reasons["missing_survival_time"] = skipped_reasons.get("missing_survival_time", 0) + 1
            continue

        # GDC sometimes records 0 days for immediate post-op death / initial follow-up;
        # set minimum floor of 0.1 months (~3 days) to avoid division by zero in log transforms
        survival_months = round(max(time_days / 30.4375, 0.1), 2)

        gender = demographic.get("gender", "unknown")
        primary_site = case.get("primary_site", "Colon")

        records.append({
            "patient_id": pid,
            "age": age_years,
            "tumor_stage": norm_stage,
            "raw_stage": raw_stage,
            "gender": gender,
            "survival_months": survival_months,
            "event": event,
            "vital_status": vital_status
        })

    df = pd.DataFrame(records)
    logger.info(f"Successfully processed {len(df)} verified clinical records.")
    logger.info(f"Skipped breakdown: {skipped_reasons}")

    # Deduplicate by patient_id if any duplicates exist
    df = df.drop_duplicates(subset=["patient_id"], keep="first")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"Saved processed clinical survival dataset to {out_path} ({len(df)} rows)")

    # Print summary
    event_counts = df["event"].value_counts().to_dict()
    stage_counts = df["tumor_stage"].value_counts().to_dict()
    logger.info(f"Summary: Events={event_counts.get(1, 0)}, Censored={event_counts.get(0, 0)}")
    logger.info(f"Stage distribution: {stage_counts}")
    logger.info(f"Age range: {df['age'].min()} - {df['age'].max()} (mean {df['age'].mean():.1f})")
    logger.info(f"Survival months range: {df['survival_months'].min()} - {df['survival_months'].max()} (median {df['survival_months'].median():.1f})")

    return df

if __name__ == "__main__":
    process_clinical_cases()
