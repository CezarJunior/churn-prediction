"""
Feature engineering module for the Churn Prediction pipeline.

Handles data loading, cleaning, and feature construction for the
IBM Telco Customer Churn dataset.
"""

import io
import logging
from typing import Tuple, List

import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

TELCO_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

# Columns that use Yes/No encoding and contribute to service count
_SERVICE_COLS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "AvgMonthlySpend",
    "HasMultipleServices",
]

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "tenure_group",
    "IsLongTermContract",
]


def load_data(url: str = TELCO_URL) -> pd.DataFrame:
    """
    Download the IBM Telco Customer Churn CSV from a remote URL.

    Falls back to reading from `data/Telco-Customer-Churn.csv` if the
    network request fails, so the pipeline also works offline when the
    file has been cached locally.

    Parameters
    ----------
    url : str
        Raw CSV URL. Defaults to the official IBM repository on GitHub.

    Returns
    -------
    pd.DataFrame
        Raw dataset as loaded from the CSV — no cleaning applied yet.
    """
    try:
        logger.info("Downloading Telco churn dataset from %s", url)
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        logger.info("Dataset downloaded successfully — shape: %s", df.shape)
        return df
    except requests.RequestException as exc:
        logger.warning(
            "Network download failed (%s). Trying local cache at data/Telco-Customer-Churn.csv",
            exc,
        )
        try:
            df = pd.read_csv("data/Telco-Customer-Churn.csv")
            logger.info("Loaded from local cache — shape: %s", df.shape)
            return df
        except FileNotFoundError:
            raise RuntimeError(
                "Could not download the dataset and no local cache found at "
                "data/Telco-Customer-Churn.csv. "
                "Run `python -c \"from src.pipeline.features import load_data; "
                "load_data()\"` with network access first."
            ) from exc


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply minimal cleaning to the raw Telco dataset.

    Steps
    -----
    1. Convert `TotalCharges` from object to float (the raw CSV stores it
       as a string, and rows with tenure=0 contain a whitespace string).
    2. Fill the resulting NaN values with 0 (new customers have no charges).
    3. Drop the `customerID` column — it is an opaque identifier with no
       predictive signal.
    4. Encode the binary target `Churn` as integer 0/1.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe as returned by :func:`load_data`.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe ready for feature engineering.
    """
    df = df.copy()

    # TotalCharges is stored as object in the raw CSV
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    null_charges = df["TotalCharges"].isna().sum()
    if null_charges > 0:
        logger.debug("Filling %d null TotalCharges with 0", null_charges)
        df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # customerID carries no signal
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Encode target
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)

    # SeniorCitizen is already 0/1 but we cast to string for consistency
    # with the OneHotEncoder downstream
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

    logger.info("Cleaning complete — shape: %s, churn rate: %.1f%%",
                df.shape,
                df["Churn"].mean() * 100 if "Churn" in df.columns else float("nan"))
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features that improve model signal.

    New columns
    -----------
    tenure_group : str
        Binned tenure in months: ``'0-12'``, ``'13-24'``, ``'25-48'``,
        ``'49-60'``, ``'61+'``.  Captures the non-linear relationship
        between tenure and churn risk.

    AvgMonthlySpend : float
        ``TotalCharges / tenure`` capped at ``MonthlyCharges`` to avoid
        division-by-zero for new customers (tenure=0).

    HasMultipleServices : int
        Count of the eight add-on service columns that equal ``'Yes'``.
        Customers with more services are stickier and churn less.

    IsLongTermContract : int (0/1)
        Flag indicating a ``'Two year'`` contract — the strongest single
        predictor of retention in this dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe from :func:`clean_data`.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional feature columns.
    """
    df = df.copy()

    # Tenure binning — captures non-linearity without polynomial expansion
    bins = [0, 12, 24, 48, 60, float("inf")]
    labels = ["0-12", "13-24", "25-48", "49-60", "61+"]
    df["tenure_group"] = pd.cut(
        df["tenure"], bins=bins, labels=labels, right=True
    ).astype(str)

    # Average monthly spend (proxy for CLV trajectory)
    df["AvgMonthlySpend"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"],  # for brand-new customers use current charge
    ).astype(float)

    # Service bundle depth
    service_matrix = df[_SERVICE_COLS].apply(lambda col: (col == "Yes").astype(int))
    df["HasMultipleServices"] = service_matrix.sum(axis=1)

    # Contract type flag
    df["IsLongTermContract"] = (df["Contract"] == "Two year").astype(int).astype(str)

    logger.info("Feature engineering complete — shape: %s", df.shape)
    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Build a scikit-learn :class:`~sklearn.compose.ColumnTransformer` that
    scales numeric features and one-hot encodes categoricals.

    Returns
    -------
    ColumnTransformer
        Unfitted preprocessor ready to be inserted into a Pipeline.
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def get_feature_target_split(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate the feature matrix ``X`` and target vector ``y``.

    Parameters
    ----------
    df : pd.DataFrame
        Fully engineered dataframe (output of :func:`engineer_features`).

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    """
    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    # Keep only columns that exist in the dataframe
    available = [c for c in all_features if c in df.columns]
    X = df[available].copy()
    y = df["Churn"].copy()
    return X, y


def get_feature_lists() -> Tuple[List[str], List[str]]:
    """Return (NUMERIC_FEATURES, CATEGORICAL_FEATURES) for external use."""
    return NUMERIC_FEATURES, CATEGORICAL_FEATURES
