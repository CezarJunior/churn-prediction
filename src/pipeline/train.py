"""
Model training, hyperparameter tuning, and evaluation.

Uses XGBoost inside a scikit-learn Pipeline, with MLflow for experiment
tracking and Optuna for Bayesian hyperparameter optimisation.
"""

import logging
import warnings
from typing import Any, Dict, Optional, Tuple

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .features import (
    build_preprocessor,
    clean_data,
    engineer_features,
    get_feature_target_split,
    load_data,
)

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_TRIALS = 20
CV_FOLDS = 5
MLFLOW_EXPERIMENT = "churn-prediction"


def _compute_scale_pos_weight(y: pd.Series) -> float:
    """Return XGBoost scale_pos_weight to handle class imbalance."""
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    return float(n_neg / n_pos)


def _build_pipeline(params: Dict[str, Any], scale_pos_weight: float) -> Pipeline:
    """Assemble preprocessor + XGBClassifier into a single Pipeline."""
    preprocessor = build_preprocessor()
    clf = XGBClassifier(
        n_estimators=params.get("n_estimators", 300),
        max_depth=params.get("max_depth", 6),
        learning_rate=params.get("learning_rate", 0.05),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        min_child_weight=params.get("min_child_weight", 3),
        reg_alpha=params.get("reg_alpha", 0.1),
        reg_lambda=params.get("reg_lambda", 1.0),
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])


def _optuna_objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scale_pos_weight: float,
) -> float:
    """Optuna objective: maximise CV ROC-AUC on training fold."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
    }
    pipeline = _build_pipeline(params, scale_pos_weight)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_val_score(
            pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
        )
    return float(scores.mean())


def train_model(
    df: Optional[pd.DataFrame] = None,
    n_trials: int = N_TRIALS,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Full training routine: optional data ingestion → feature engineering →
    Optuna tuning (``n_trials`` trials) → final fit → MLflow logging.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Pre-loaded dataframe. If *None*, the function calls
        :func:`~src.pipeline.features.load_data` and
        :func:`~src.pipeline.features.clean_data` automatically.
    n_trials : int
        Number of Optuna trials. Default is 20. Reduce for faster iteration
        during development.

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        Fitted pipeline (preprocessor + XGBClassifier).
    metrics : dict
        Dictionary containing ``auc``, ``f1``, ``precision``, ``recall``,
        ``accuracy``, and the best Optuna params.
    """
    if df is None:
        logger.info("No dataframe provided — downloading and cleaning data")
        raw = load_data()
        df = clean_data(raw)

    df_feat = engineer_features(df)
    X, y = get_feature_target_split(df_feat)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    scale_pos_weight = _compute_scale_pos_weight(y_train)
    logger.info(
        "Train: %d samples | Test: %d samples | scale_pos_weight: %.2f",
        len(X_train), len(X_test), scale_pos_weight,
    )

    # --- Optuna hyperparameter search ---
    logger.info("Starting Optuna search with %d trials ...", n_trials)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(
        lambda trial: _optuna_objective(trial, X_train, y_train, scale_pos_weight),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    best_params = study.best_params
    best_cv_auc = study.best_value
    logger.info("Best CV AUC: %.4f | Params: %s", best_cv_auc, best_params)

    # --- Train final model on full training set ---
    final_pipeline = _build_pipeline(best_params, scale_pos_weight)
    final_pipeline.fit(X_train, y_train)

    # --- Evaluate on held-out test set ---
    metrics = evaluate_model(final_pipeline, X_test, y_test)
    metrics["best_params"] = best_params
    metrics["cv_auc"] = round(best_cv_auc, 4)

    # --- MLflow tracking ---
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="xgb_optuna"):
        mlflow.log_params(best_params)
        mlflow.log_param("scale_pos_weight", round(scale_pos_weight, 4))
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))

        mlflow.log_metric("test_auc", metrics["auc"])
        mlflow.log_metric("test_f1", metrics["f1"])
        mlflow.log_metric("test_precision", metrics["precision"])
        mlflow.log_metric("test_recall", metrics["recall"])
        mlflow.log_metric("test_accuracy", metrics["accuracy"])
        mlflow.log_metric("cv_auc", best_cv_auc)

        mlflow.sklearn.log_model(
            final_pipeline,
            artifact_path="model",
            registered_model_name="churn-xgb",
        )
        logger.info("MLflow run logged — AUC=%.4f, F1=%.4f", metrics["auc"], metrics["f1"])

    return final_pipeline, metrics


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """
    Compute classification metrics on a test set.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Fitted pipeline.
    X_test : pd.DataFrame
    y_test : pd.Series

    Returns
    -------
    dict
        Keys: ``auc``, ``f1``, ``precision``, ``recall``, ``accuracy``,
        ``confusion_matrix``, ``classification_report``.
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["No Churn", "Churn"])

    metrics = {
        "auc": round(roc_auc_score(y_test, y_proba), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    logger.info(
        "Evaluation — AUC: %.4f | F1: %.4f | Precision: %.4f | Recall: %.4f",
        metrics["auc"],
        metrics["f1"],
        metrics["precision"],
        metrics["recall"],
    )
    return metrics
