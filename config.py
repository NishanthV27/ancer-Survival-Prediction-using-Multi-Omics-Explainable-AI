import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'cancer-survival-real-multiomics-key')
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')

    # Data paths in priority order:
    # 1. Real matched TCGA-COAD multi-omics dataset
    # 2. Real TCGA-COAD clinical dataset
    # 3. Fallback demo CSV
    MULTIOMICS_PATH = os.path.join(PROCESSED_DIR, 'tcga_coad_multiomics.csv')
    CLINICAL_PATH = os.path.join(PROCESSED_DIR, 'tcga_coad_clinical.csv')
    DEMO_PATH = os.path.join(DATA_DIR, 'synthetic_tcga_coad_demo.csv')

    # Default to matched real multi-omics dataset if present and valid
    if os.path.exists(MULTIOMICS_PATH) and os.path.getsize(MULTIOMICS_PATH) > 100:
        DATASET_PATH = MULTIOMICS_PATH
        DATASET_MODE = "real_multiomics"
    elif os.path.exists(CLINICAL_PATH) and os.path.getsize(CLINICAL_PATH) > 100:
        DATASET_PATH = CLINICAL_PATH
        DATASET_MODE = "real_clinical"
    else:
        DATASET_PATH = DEMO_PATH
        DATASET_MODE = "synthetic_demo"

    RANDOM_SEED = 42
    TEST_SIZE = 0.2
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() in ('true', '1')
