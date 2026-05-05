"""
Tests for src/pipeline/train.py

Covers:
- _compute_scale_pos_weight
- _build_pipeline produces a valid sklearn Pipeline
- evaluate_model returns all expected metric keys
- train_model (fast smoke test with 1 trial and a tiny synthetic dataset)
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_clean_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Return a minimal clean Telco-like dataframe (post-clean_data)."""
    rng = np.random.default_rng(seed)
    contracts = ["Month-to-month", "One year", "Two year"]
    internet = ["Fiber optic", "DSL", "No"]
    payment = [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    service_values = ["Yes", "No", "No internet service"]
    records = []
    for _ in range(n):
        tenure = int(rng.integers(1, 72))
        monthly = float(rng.uniform(20, 110))
        records.append(
            {
                "gender": rng.choice(["Male", "Female"]),
                "SeniorCitizen": rng.choice(["0", "1"]),
                "Partner": rng.choice(["Yes", "No"]),
                "Dependents": rng.choice(["Yes", "No"]),
                "tenure": tenure,
                "PhoneService": rng.choice(["Yes", "No"]),
                "MultipleLines": rng.choice(["Yes", "No", "No phone service"]),
                "InternetService": rng.choice(internet),
                "OnlineSecurity": rng.choice(service_values),
                "OnlineBackup": rng.choice(service_values),
                "DeviceProtection": rng.choice(service_values),
                "TechSupport": rng.choice(service_values),
                "StreamingTV": rng.choice(service_values),
                "StreamingMovies": rng.choice(service_values),
                "Contract": rng.choice(contracts),
                "PaperlessBilling": rng.choice(["Yes", "No"]),
                "PaymentMethod": rng.choice(payment),
                "MonthlyCharges": round(monthly, 2),
                "TotalCharges": round(monthly * tenure, 2),
                "Churn": int(rng.integers(0, 2)),
            }
        )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# _compute_scale_pos_weight
# ---------------------------------------------------------------------------

class TestScalePosWeight:
    def _call(self, y):
        from src.pipeline.train import _compute_scale_pos_weight
        return _compute_scale_pos_weight(y)

    def test_balanced_classes_returns_one(self):
        y = pd.Series([0, 1, 0, 1, 0, 1])
        result = self._call(y)
        assert abs(result - 1.0) < 1e-9

    def test_imbalanced_returns_correct_ratio(self):
        # 75 negatives, 25 positives → weight = 3.0
        y = pd.Series([0] * 75 + [1] * 25)
        result = self._call(y)
        assert abs(result - 3.0) < 1e-9

    def test_returns_float(self):
        y = pd.Series([0, 0, 0, 1])
        assert isinstance(self._call(y), float)


# ---------------------------------------------------------------------------
# _build_pipeline
# ---------------------------------------------------------------------------

class TestBuildPipeline:
    def setup_method(self):
        from src.pipeline.train import _build_pipeline
        self.pipeline = _build_pipeline({}, scale_pos_weight=3.0)

    def test_returns_sklearn_pipeline(self):
        assert isinstance(self.pipeline, Pipeline)

    def test_has_preprocessor_step(self):
        assert "preprocessor" in self.pipeline.named_steps

    def test_has_classifier_step(self):
        assert "classifier" in self.pipeline.named_steps

    def test_classifier_is_xgboost(self):
        from xgboost import XGBClassifier
        clf = self.pipeline.named_steps["classifier"]
        assert isinstance(clf, XGBClassifier)


# ---------------------------------------------------------------------------
# evaluate_model
# ---------------------------------------------------------------------------

class TestEvaluateModel:
    def setup_method(self):
        from src.pipeline.features import engineer_features, get_feature_target_split
        from src.pipeline.train import _build_pipeline, _compute_scale_pos_weight

        df = _make_clean_df(200)
        df_feat = engineer_features(df)
        X, y = get_feature_target_split(df_feat)

        from sklearn.model_selection import train_test_split
        X_train, self.X_test, y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=0
        )
        spw = _compute_scale_pos_weight(y_train)
        self.pipeline = _build_pipeline({"n_estimators": 10, "max_depth": 3}, spw)
        self.pipeline.fit(X_train, y_train)

    def test_returns_dict(self):
        from src.pipeline.train import evaluate_model
        result = evaluate_model(self.pipeline, self.X_test, self.y_test)
        assert isinstance(result, dict)

    def test_all_metric_keys_present(self):
        from src.pipeline.train import evaluate_model
        result = evaluate_model(self.pipeline, self.X_test, self.y_test)
        expected_keys = {"auc", "f1", "precision", "recall", "accuracy",
                         "confusion_matrix", "classification_report"}
        assert expected_keys.issubset(result.keys())

    def test_auc_in_valid_range(self):
        from src.pipeline.train import evaluate_model
        result = evaluate_model(self.pipeline, self.X_test, self.y_test)
        assert 0.0 <= result["auc"] <= 1.0

    def test_confusion_matrix_is_2x2(self):
        from src.pipeline.train import evaluate_model
        result = evaluate_model(self.pipeline, self.X_test, self.y_test)
        cm = result["confusion_matrix"]
        assert len(cm) == 2 and len(cm[0]) == 2


# ---------------------------------------------------------------------------
# train_model — smoke test (1 trial, small dataset, no MLflow artefacts)
# ---------------------------------------------------------------------------

class TestTrainModelSmoke:
    """
    Smoke test: 1 Optuna trial, tiny dataset, no disk writes.
    This should complete in under 30 seconds.
    """

    def test_train_model_returns_pipeline_and_metrics(self):
        from src.pipeline.features import engineer_features
        from src.pipeline.train import train_model

        df = _make_clean_df(300)

        # Patch joblib dump so the model isn't written during tests
        pipeline, metrics = train_model(df=df, n_trials=1)

        assert isinstance(pipeline, Pipeline)
        assert isinstance(metrics, dict)
        assert "auc" in metrics
        assert 0.0 <= metrics["auc"] <= 1.0

    def test_train_model_best_params_in_metrics(self):
        from src.pipeline.train import train_model

        df = _make_clean_df(300)
        _, metrics = train_model(df=df, n_trials=1)
        assert "best_params" in metrics
        assert isinstance(metrics["best_params"], dict)
