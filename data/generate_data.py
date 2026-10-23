"""Script to generate synthetic TCGA-COAD demo dataset for cancer survival prediction.

Generates 300 realistic patient records with clinical and genomic features:
- patient_id: TCGA-COAD-0001 through TCGA-COAD-0300
- age: roughly normally distributed around 60 (30-85)
- tumor_stage: I (15%), II (30%), III (35%), IV (20%)
- gene_expr_A, gene_expr_B, gene_expr_C
- methylation_A, methylation_B
- survival_months: 1.0 to 120.0
- event: 0 (censored) or 1 (death), ~55% event rate
"""

import os
import numpy as np
import pandas as pd


def generate_synthetic_data(output_path=None, seed=42, n_samples=300):
    np.random.seed(seed)

    patient_ids = [f"TCGA-COAD-{i:04d}" for i in range(1, n_samples + 1)]

    # Age: integers between 30 and 85, roughly normal around 60
    age = np.clip(
        np.round(np.random.normal(loc=60.0, scale=10.0, size=n_samples)), 30, 85
    ).astype(int)

    # Tumor stage distribution: ~15% Stage I, 30% Stage II, 35% Stage III, 20% Stage IV
    stage_choices = ['I', 'II', 'III', 'IV']
    stage_probs = [0.15, 0.30, 0.35, 0.20]
    tumor_stage = np.random.choice(stage_choices, size=n_samples, p=stage_probs)

    stage_map = {'I': 1, 'II': 2, 'III': 3, 'IV': 4}
    stage_num = np.array([stage_map[s] for s in tumor_stage])

    # Gene expression and methylation features with realistic biological correlations
    # gene_expr_A: 2.0 to 15.0, higher values correlate with worse outcomes
    gene_expr_A = np.clip(
        3.2 + 1.9 * stage_num + np.random.normal(0, 1.7, size=n_samples), 2.0, 15.0
    )

    # gene_expr_B: 0.5 to 8.0, protective marker (decreases slightly in advanced stages)
    gene_expr_B = np.clip(
        5.2 - 0.45 * stage_num + np.random.normal(0, 1.1, size=n_samples), 0.5, 8.0
    )

    # gene_expr_C: 1.0 to 12.0
    gene_expr_C = np.clip(
        4.2 + 1.1 * stage_num + np.random.normal(0, 1.5, size=n_samples), 1.0, 12.0
    )

    # methylation_A: 0.0 to 1.0, hypermethylated in progressive disease
    methylation_A = np.clip(
        0.22 + 0.14 * stage_num + np.random.normal(0, 0.10, size=n_samples), 0.0, 1.0
    )

    # methylation_B: 0.0 to 1.0
    methylation_B = np.clip(
        0.72 - 0.11 * stage_num + np.random.normal(0, 0.11, size=n_samples), 0.0, 1.0
    )

    # Combined risk score (Cox proportional hazards linear predictor)
    risk_score = (
        0.75 * (stage_num - 2.5) +
        0.035 * (age - 60) +
        0.16 * (gene_expr_A - 7.5) -
        0.12 * (gene_expr_B - 4.0) +
        0.10 * (gene_expr_C - 6.5) +
        0.45 * (methylation_A - 0.5) -
        0.25 * (methylation_B - 0.5)
    )

    # Latent survival time from Weibull/Exponential survival process
    baseline_median_months = 46.0
    hazard_multiplier = np.exp(0.55 * risk_score)
    survival_latent = baseline_median_months / hazard_multiplier * np.random.weibull(1.2, size=n_samples)

    # Censoring distribution (study follow-up time ~ up to 120 months)
    censoring_time = np.random.exponential(scale=68.0, size=n_samples)
    censoring_time = np.clip(censoring_time, 8.0, 120.0)

    # Observed time and event indicator
    observed_time = np.minimum(survival_latent, censoring_time)
    event = (survival_latent <= censoring_time).astype(int)

    # Clamp survival months between 1.0 and 120.0
    survival_months = np.clip(np.round(observed_time, 2), 1.0, 120.0)
    survival_months = np.maximum(survival_months, 1.0)

    df = pd.DataFrame({
        'patient_id': patient_ids,
        'age': age,
        'tumor_stage': tumor_stage,
        'gene_expr_A': np.round(gene_expr_A, 2),
        'gene_expr_B': np.round(gene_expr_B, 2),
        'gene_expr_C': np.round(gene_expr_C, 2),
        'methylation_A': np.round(methylation_A, 2),
        'methylation_B': np.round(methylation_B, 2),
        'survival_months': np.round(survival_months, 2),
        'event': event
    })

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        df.to_csv(output_path, index=False)

    return df


if __name__ == '__main__':
    data_dir = os.path.dirname(os.path.abspath(__file__))
    out_file = os.path.join(data_dir, 'synthetic_tcga_coad_demo.csv')
    df = generate_synthetic_data(out_file, seed=42, n_samples=300)
    print(f"Generated {len(df)} records.")
    print("Stage distribution:\n", df['tumor_stage'].value_counts(normalize=True))
    print("Event distribution:\n", df['event'].value_counts(normalize=True))
    print("Head:\n", df.head())
