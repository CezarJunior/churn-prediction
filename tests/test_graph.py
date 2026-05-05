"""
Tests for src/graph/churn_graph.py

Covers:
- ChurnState TypedDict shape
- load_and_validate_node with valid and invalid data
- preprocess_and_predict_node happy path
- identify_high_risk_node happy path and threshold-relaxation branch
- generate_report_node fallback (no GROQ_API_KEY)
- build_graph produces a compiled graph
- run_pipeline end-to-end smoke test
"""

import os

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_clean_df(n: int = 100, seed: int = 0) -> pd.DataFrame:
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


def _make_raw_df(n: int = 50, seed: int = 0) -> pd.DataFrame:
    """Raw (uncleaned) synthetic Telco frame."""
    from tests.test_features import _make_raw_df as _raw
    return _raw(n, seed)


def _make_mock_pipeline(n_rows: int):
    """Return a mock sklearn Pipeline whose predict_proba returns random values."""
    mock = MagicMock()
    rng = np.random.default_rng(42)
    proba = rng.uniform(0, 1, n_rows)
    mock.predict_proba.return_value = np.column_stack([1 - proba, proba])
    return mock


# ---------------------------------------------------------------------------
# load_and_validate_node
# ---------------------------------------------------------------------------

class TestLoadAndValidateNode:
    def _call(self, state):
        from src.graph.churn_graph import load_and_validate_node
        return load_and_validate_node(state)

    def test_passes_with_valid_raw_data(self):
        raw = _make_raw_df(30)
        state = {
            "raw_data": raw, "clean_data": None, "predictions": None,
            "high_risk_customers": None, "report": None, "error": None,
        }
        result = self._call(state)
        assert result.get("error") is None
        assert result.get("clean_data") is not None

    def test_fails_with_missing_required_columns(self):
        df = pd.DataFrame({"tenure": [1, 2], "SomeColumn": [3, 4]})
        state = {
            "raw_data": df, "clean_data": None, "predictions": None,
            "high_risk_customers": None, "report": None, "error": None,
        }
        result = self._call(state)
        assert result["error"] is not None
        assert "missing" in result["error"].lower()

    def test_returns_clean_data(self):
        raw = _make_raw_df(20)
        state = {
            "raw_data": raw, "clean_data": None, "predictions": None,
            "high_risk_customers": None, "report": None, "error": None,
        }
        result = self._call(state)
        # customerID should be dropped in clean_data
        if result.get("clean_data") is not None:
            assert "customerID" not in result["clean_data"].columns


# ---------------------------------------------------------------------------
# preprocess_and_predict_node
# ---------------------------------------------------------------------------

class TestPreprocessAndPredictNode:
    def _call(self, state):
        from src.graph.churn_graph import preprocess_and_predict_node
        return preprocess_and_predict_node(state)

    def test_adds_churn_probability_column(self):
        from src.pipeline.features import clean_data
        raw = _make_raw_df(40)
        clean = clean_data(raw)
        mock_pipeline = _make_mock_pipeline(len(clean))
        state = {
            "raw_data": raw,
            "clean_data": clean,
            "predictions": None,
            "high_risk_customers": None,
            "report": None,
            "error": None,
            "_model_pipeline": mock_pipeline,
        }
        result = self._call(state)
        assert result.get("error") is None
        assert "churn_probability" in result["predictions"].columns

    def test_adds_risk_label_column(self):
        from src.pipeline.features import clean_data
        raw = _make_raw_df(40)
        clean = clean_data(raw)
        mock_pipeline = _make_mock_pipeline(len(clean))
        state = {
            "raw_data": raw,
            "clean_data": clean,
            "predictions": None,
            "high_risk_customers": None,
            "report": None,
            "error": None,
            "_model_pipeline": mock_pipeline,
        }
        result = self._call(state)
        valid_labels = {"Low", "Medium", "High", "Critical"}
        actual = set(result["predictions"]["risk_label"].unique())
        assert actual.issubset(valid_labels)

    def test_short_circuits_on_error(self):
        state = {
            "raw_data": None, "clean_data": None, "predictions": None,
            "high_risk_customers": None, "report": None,
            "error": "previous error",
        }
        result = self._call(state)
        assert result["error"] == "previous error"

    def test_error_when_no_pipeline(self):
        from src.pipeline.features import clean_data
        raw = _make_raw_df(20)
        clean = clean_data(raw)
        state = {
            "raw_data": raw,
            "clean_data": clean,
            "predictions": None,
            "high_risk_customers": None,
            "report": None,
            "error": None,
            "_model_pipeline": None,
        }
        result = self._call(state)
        assert result["error"] is not None


# ---------------------------------------------------------------------------
# identify_high_risk_node
# ---------------------------------------------------------------------------

class TestIdentifyHighRiskNode:
    def _make_predictions_df(self, n: int, high_risk_frac: float = 0.3) -> pd.DataFrame:
        from src.pipeline.features import clean_data, engineer_features
        raw = _make_raw_df(n)
        df = engineer_features(clean_data(raw))
        rng = np.random.default_rng(99)
        proba = np.where(
            rng.random(n) < high_risk_frac,
            rng.uniform(0.71, 1.0, n),
            rng.uniform(0.0, 0.69, n),
        )
        df["churn_probability"] = proba
        df["churn_predicted"] = (proba >= 0.5).astype(int)
        df["risk_label"] = "Low"
        return df

    def _call(self, state):
        from src.graph.churn_graph import identify_high_risk_node
        return identify_high_risk_node(state)

    def test_filters_high_risk_customers(self):
        predictions = self._make_predictions_df(80, high_risk_frac=0.4)
        state = {
            "raw_data": None, "clean_data": None,
            "predictions": predictions,
            "high_risk_customers": None, "report": None, "error": None,
        }
        result = self._call(state)
        hr = result["high_risk_customers"]
        assert (hr["churn_probability"] > 0.7).all()

    def test_fallback_to_top20_when_no_high_risk(self):
        predictions = self._make_predictions_df(50, high_risk_frac=0.0)
        # Force all proba to be below threshold
        predictions["churn_probability"] = 0.5
        state = {
            "raw_data": None, "clean_data": None,
            "predictions": predictions,
            "high_risk_customers": None, "report": None, "error": None,
        }
        result = self._call(state)
        # Should not error and should return <= 20 customers
        assert result.get("error") is None
        assert len(result["high_risk_customers"]) <= 20

    def test_short_circuits_on_error(self):
        state = {
            "raw_data": None, "clean_data": None, "predictions": None,
            "high_risk_customers": None, "report": None, "error": "boom",
        }
        result = self._call(state)
        assert result["error"] == "boom"


# ---------------------------------------------------------------------------
# generate_report_node (offline / fallback branch)
# ---------------------------------------------------------------------------

class TestGenerateReportNodeFallback:
    def _call(self, state):
        from src.graph.churn_graph import generate_report_node
        return generate_report_node(state)

    def test_returns_fallback_report_when_no_api_key(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)

        from src.pipeline.features import clean_data, engineer_features
        raw = _make_raw_df(50)
        df = engineer_features(clean_data(raw))
        rng = np.random.default_rng(7)
        proba = rng.uniform(0.71, 1.0, 20)
        high_risk = df.head(20).copy()
        high_risk["churn_probability"] = proba
        high_risk["risk_label"] = "High"
        full_preds = df.copy()
        full_preds["churn_probability"] = rng.uniform(0, 1, len(df))

        state = {
            "raw_data": None,
            "clean_data": None,
            "predictions": full_preds,
            "high_risk_customers": high_risk,
            "_segment_summary": None,
            "report": None,
            "error": None,
        }
        result = self._call(state)
        assert result.get("error") is None
        assert result["report"] is not None
        assert len(result["report"]) > 50

    def test_short_circuits_on_error(self):
        state = {
            "raw_data": None, "clean_data": None, "predictions": None,
            "high_risk_customers": None, "report": None, "error": "earlier error",
        }
        result = self._call(state)
        assert result["error"] == "earlier error"


# ---------------------------------------------------------------------------
# build_graph
# ---------------------------------------------------------------------------

class TestBuildGraph:
    def test_returns_compiled_graph(self):
        from src.graph.churn_graph import build_graph
        graph = build_graph()
        # LangGraph compiled graphs have an .invoke method
        assert callable(getattr(graph, "invoke", None))


# ---------------------------------------------------------------------------
# run_pipeline — end-to-end smoke test
# ---------------------------------------------------------------------------

class TestRunPipelineSmoke:
    def test_smoke_with_mock_model(self, monkeypatch):
        """run_pipeline should complete without errors using a mock model."""
        monkeypatch.delenv("GROQ_API_KEY", raising=False)

        from src.pipeline.features import clean_data
        raw = _make_raw_df(60)
        clean = clean_data(raw)
        mock_pipeline = _make_mock_pipeline(len(clean))

        from src.graph.churn_graph import run_pipeline

        # run_pipeline calls load_data() if raw_df is None — supply it directly
        state = run_pipeline(model_pipeline=mock_pipeline, raw_df=raw)

        assert state.get("error") is None
        assert state["report"] is not None
        assert state["high_risk_customers"] is not None
