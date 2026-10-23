import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'cancer-survival-demo-key-change-in-production')
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    DATASET_PATH = os.path.join(DATA_DIR, 'synthetic_tcga_coad_demo.csv')
    RANDOM_SEED = 42
    TEST_SIZE = 0.2
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() in ('true', '1')
