# Cancer Survival Prediction using Multi-Omics and Explainable AI

> **Final-Year Academic Project**: An end-to-end medical research dashboard for colon cancer survival risk prediction, Kaplan–Meier survival curve analysis, and Explainable AI (XAI) feature attributions using synthetic TCGA-COAD multi-omics data.
 
---

## 1. Project Disclaimer

> **IMPORTANT ACADEMIC DEMONSTRATION NOTICE**:
> This web application utilizes a synthetic demonstration dataset modelled after the Cancer Genome Atlas Colon Adenocarcinoma (TCGA-COAD) cohort. All patient identifiers, genomic expressions, epigenetic methylation beta-values, and clinical follow-up intervals are synthetically generated for educational and research demonstration purposes. 
> 
> **Do not use this system for real medical diagnosis, clinical prognosis, patient management, or clinical decision support.** No findings or feature attributions constitute clinically validated biomarkers.

---

## 2. Key Features

- **Medical AI Dashboard**: Interactive cohort statistics (mean age, vital status distribution, pathological tumor stage distribution, gene expression box summaries, and methylation profiles) powered by Chart.js.
- **Multimodal Patient Risk Prediction**: Real-time inference calculating continuous relative risk scores, risk stratification (High Risk vs. Low Risk) using median training thresholds, and landmark survival probabilities at 12, 24, 36, and 60 months.
- **Dual Survival Analysis Models**:
  - **Cox Proportional Hazards Regression** (Lifelines / Scikit-Survival regularized partial likelihood).
  - **Random Survival Forest** (Ishwaran et al. log-rank test splitting criterion with Nelson–Aalen leaf cumulative hazard estimation).
- **Explainable AI (XAI)**:
  - **Global Feature Importance**: Relative ranking of multi-omics features across the entire cohort.
  - **Patient-Specific Local Attribution**: Directional contributions (Red = increases mortality risk, Green = protective / decreases risk) for any patient query.
  - **Biological Feature Descriptions**: Annotations for oncogenes, tumor suppressors, DNA repair factors, and CpG island methylation sites.
- **Kaplan–Meier Survival Analytics**:
  - Model-stratified High vs. Low risk survival trajectories with 95% Greenwood confidence intervals.
  - Pathological tumor stage stratification (Stages I, II, III, and IV).
  - Exact Log-Rank test $p$-values and numbers-at-risk breakdown.
- **Dataset Explorer**: Searchable, filterable (by tumor stage and vital event status), paginated table with single-click CSV dataset export.
- **Comprehensive Project Overview**: In-depth explanations of multi-omics fusion, censoring mechanics, C-index discrimination, and future TCGA translation.

---

## 3. Technology Stack

- **Backend**: Python 3.12+, Flask 3.x, Jinja2, NumPy, Pandas, Lifelines, Scikit-Survival, Joblib, SciPy.
- **Frontend**: HTML5, CSS3, JavaScript (ES6+), Bootstrap 5.3, Bootstrap Icons, Chart.js 4.4.
- **Testing**: Python `unittest` suite (15 unit and integration tests).

---

## 4. Project Directory Structure

```text
cancer-survival-project/
├── app.py                      # Flask application factory, routes, and JSON APIs
├── config.py                   # Environment, security, and path configuration
├── requirements.txt            # Python package dependencies
├── README.md                   # Complete documentation and setup guide
├── data/
│   ├── generate_data.py        # Reproducible synthetic dataset generator (seed 42)
│   └── synthetic_tcga_coad_demo.csv # 300 synthetic TCGA-COAD patient records
├── models/
│   ├── cox_ph_model.joblib     # Serialized Cox Proportional Hazards model
│   ├── rsf_model.joblib        # Serialized Random Survival Forest model
│   ├── preprocessor.joblib     # Fitted feature transformer (scaling & encoding)
│   ├── metadata.joblib         # Training metadata, C-indices, and thresholds
│   └── metadata.json           # Human-readable JSON export of model metadata
├── ml/
│   ├── __init__.py             # ML package initializer
│   ├── preprocessing.py        # Data validation, stratified split, and feature scaling
│   ├── train_models.py         # Survival model training (Cox PH & Random Survival Forest)
│   ├── prediction.py           # Single-patient risk inference & survival curves
│   ├── survival_analysis.py    # Kaplan-Meier fitting and log-rank statistical testing
│   ├── explainability.py       # SHAP / marginal feature attributions & descriptions
│   └── evaluation.py           # Harrell's Concordance Index (C-index) calculation
├── templates/
│   ├── base.html               # Medical dashboard shell (sidebar, CDNs, footer)
│   ├── index.html              # Landing page, workflow flowcharts, and cohort highlights
│   ├── dashboard.html          # Clinical & molecular distribution charts
│   ├── predict.html            # Patient risk scoring form with dynamic Chart.js charts
│   ├── analytics.html          # Kaplan-Meier curves, log-rank tests, numbers-at-risk
│   ├── explainability.html     # Global feature importance & patient attribution
│   ├── dataset.html            # Searchable, paginated patient table with CSV download
│   └── about.html              # Methodology, clinical rationale, and limitations
├── static/
│   ├── css/
│   │   └── style.css           # Custom medical dashboard theme & responsive layout
│   └── js/
│       ├── main.js             # Navigation toggles, fetch helpers, and Chart defaults
│       ├── dashboard.js        # Dashboard Chart.js controllers
│       ├── prediction.js       # Risk prediction form handler & visualization
│       ├── analytics.js        # Kaplan-Meier line charts & log-rank indicators
│       └── explainability.js   # Feature importance bar charts & XAI handlers
└── tests/
    ├── __init__.py
    ├── test_prediction.py      # ML pipeline, preprocessing, and prediction unit tests
    └── test_app.py             # Flask API endpoints and web route integration tests
```

---

## 5. Model Evaluation Results

Models were trained and evaluated on a held-out test split (20% of the cohort, 60 patients) stratified on event occurrences (preserving the 55% event / 45% censored ratio):

| Model | Discrimination Metric (Harrell's C-Index) | Concordant Pairs | Discordant Pairs | Quality Rating |
|---|---|---|---|---|
| **Cox Proportional Hazards** | **0.9943** | 1,388 | 8 | Excellent Discrimination |
| **Random Survival Forest** | **0.7991** | 1,115 | 280 | Good Discrimination |

- **Risk Score Direction**: Higher risk score indicates higher relative mortality hazard.
- **Threshold Rule**: High-risk vs. Low-risk categorization is defined by the median risk score computed over the training split (`threshold = -0.0601` for Cox PH, `0.7490` for RSF).

---

## 6. Installation & Execution Guide (Windows PowerShell)

Follow these exact steps in **Windows PowerShell**:

### Step 1: Open PowerShell and Navigate to the Project
```powershell
cd "c:\Users\PSB 21\Desktop\project"
```

### Step 2: Set Up Python Virtual Environment
If you wish to use a virtual environment:
```powershell
# Create virtual environment using Python 3.12 or 3.14
py -3.12 -m venv venv

# Activate the virtual environment
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Required Dependencies
```powershell
python -m pip install -r requirements.txt
```

> **Note on Windows Application Control**: The application is built with native, zero-C-extension fallbacks for preprocessing and survival modeling to ensure 100% compatibility across environments where Windows Smart App Control or AppLocker restricts binary wheels.

### Step 4: Verify or Train Survival Models
The pre-trained models are already saved in `models/`. To retrain models from scratch on the dataset:
```powershell
python -c "from ml.train_models import train_and_save_all; res = train_and_save_all('data/synthetic_tcga_coad_demo.csv', 'models'); print(res['status'], res['best_model'], res['best_c_index'])"
```

### Step 5: Run the Test Suite
Verify that all unit and integration tests pass:
```powershell
python -m unittest discover tests -v
```
*Expected output: `Ran 15 tests ... OK`*

### Step 6: Start the Flask Web Application
```powershell
python app.py
```

### Step 7: Access the Application
Open your web browser and navigate to:
```text
http://127.0.0.1:5000
```

---

## 7. API Endpoints Reference

| HTTP Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Home page with project overview, multi-omics description, and workflow |
| `GET` | `/dashboard` | Interactive clinical, genomic, and epigenetic distributions |
| `GET` | `/api/dashboard` | JSON feed for summary counts, distributions, and model C-indices |
| `GET` | `/predict` | Patient survival risk prediction interface |
| `POST` | `/api/predict` | Risk inference returning risk score, category, milestones, and attributions |
| `GET` | `/analytics` | Kaplan–Meier survival curve analysis and log-rank test page |
| `GET` | `/api/analytics` | JSON feed of high vs. low risk KM curves, stage curves, and log-rank $p$-values |
| `GET` | `/explainability` | Explainable AI (SHAP / Feature Importance) page |
| `GET` | `/api/explainability` | Global feature ranking and biological feature descriptions |
| `POST` | `/api/explainability` | Patient-specific local feature contribution analysis |
| `GET` | `/dataset` | Searchable and filterable dataset explorer page |
| `GET` | `/api/dataset` | Paginated records with stage and event filters (`?page=1&per_page=20`) |
| `GET` | `/api/dataset/download` | Direct attachment download of `synthetic_tcga_coad_demo.csv` |
| `GET` | `/about` | Project background, mathematical formulas, and limitations |
| `GET` | `/api/health` | Health-check endpoint validating dataset and model readiness |

---

## 8. Troubleshooting

1. **`Program 'pip.exe' failed to run: An Application Control policy has blocked this file`**:
   - Run pip via python executable: `python -m pip install ...` instead of calling `pip.exe` directly.
2. **`SmartAppControlState: On` or `DLL load failed while importing _base`**:
   - The project includes native NumPy/Pandas preprocessors (`SurvivalFeatureTransformer`) and native log-rank survival trees (`NativeRandomSurvivalForest`) that execute without compiling external C extensions.
3. **Port 5000 already in use**:
   - Modify the port in `app.py`: `app.run(debug=True, host='127.0.0.1', port=5001)` or pass `flask run --port=5001`.
4. **Missing Dataset File**:
   - Ensure `synthetic_tcga_coad_demo.csv` is located in `data/`. If missing, regenerate it using `python data/generate_data.py`.

---

## 9. Academic Project Metadata

- **Focus Area**: Multi-Omics Bioinformatics, Machine Learning, Survival Analysis, Explainable AI.
- **Reference Cohort Structure**: TCGA-COAD (Colon Adenocarcinoma).
- **Evaluation Criteria**: Harrell's Concordance Index ($C$-index), Log-Rank Test Significance ($p < 0.05$).
- **Explainability**: Additive Feature Attributions and Permutation Importance for Right-Censored Lifetime Outcomes.

