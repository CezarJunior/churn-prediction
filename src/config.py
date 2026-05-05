"""
Central configuration for the Churn Prediction project.
All constants, paths, and model hyperparameters live here.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root (this file lives at src/config.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_DATA_PATH = RAW_DATA_DIR / "telco_churn.csv"

# ---------------------------------------------------------------------------
# Models & artefacts
# ---------------------------------------------------------------------------
MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"

# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------
MLFLOW_DIR = PROJECT_ROOT / "mlruns"
MLFLOW_EXPERIMENT_NAME = "churn-prediction"

# ---------------------------------------------------------------------------
# LLM / Embeddings
# ---------------------------------------------------------------------------
GROQ_MODEL = "llama-3.3-70b-versatile"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# ML training
# ---------------------------------------------------------------------------
TARGET = "Churn"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Class imbalance — SMOTE
SMOTE_RANDOM_STATE = RANDOM_STATE

# Optuna
N_OPTUNA_TRIALS = 20

# ---------------------------------------------------------------------------
# Business cost metric
# (False Negative is more costly: we miss a churner and lose revenue)
# ---------------------------------------------------------------------------
COST_FP = 10   # USD: unnecessary retention offer to a loyal customer
COST_FN = 100  # USD: lost revenue from an undetected churner

# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

BINARY_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
]

CATEGORICAL_FEATURES = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
]

# Derived / engineered features added in features.py
ENGINEERED_FEATURES = [
    "tenure_group",
    "monthly_charges_bin",
    "total_services",
]

# ---------------------------------------------------------------------------
# Ensure critical directories exist on import
# ---------------------------------------------------------------------------
for _dir in [RAW_DATA_DIR, MODELS_DIR, MLFLOW_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)
