"""
Tests for src/agent/analyst_agent.py

Covers:
- _analyze_segment with valid and invalid inputs
- _suggest_retention_actions for all major retention branches
- _calculate_revenue_at_risk correctness
- run_analyst falls back gracefully when GROQ_API_KEY is absent
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_predictions_df(n: int = 50, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    contracts = ["Month-to-month", "One year", "Two year"]
    proba = rng.uniform(0, 1, n)
    records = []
    for i in range(n):
        monthly = float(rng.uniform(20, 110))
        records.append(
            {
                "customerID": f"T-{i:04d}",
                "Contract": rng.choice(contracts),
                "tenure": int(rng.integers(1, 72)),
                "MonthlyCharges": round(monthly, 2),
                "InternetService": rng.choice(["Fiber optic", "DSL", "No"]),
                "churn_probability": float(proba[i]),
                "risk_label": "High" if proba[i] > 0.7 else "Low",
            }
        )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# _analyze_segment
# ---------------------------------------------------------------------------

class TestAnalyzeSegment:
    def _call(self, df, segment_json):
        from src.agent.analyst_agent import _analyze_segment
        return _analyze_segment(df, segment_json)

    def test_valid_segment_returns_string(self):
        df = _make_predictions_df()
        result = self._call(df, '{"column": "Contract", "value": "Month-to-month"}')
        assert isinstance(result, str)
        assert "Month-to-month" in result

    def test_segment_contains_customer_count(self):
        df = _make_predictions_df()
        result = self._call(df, '{"column": "Contract", "value": "Month-to-month"}')
        assert "Customers:" in result

    def test_invalid_column_returns_error_string(self):
        df = _make_predictions_df()
        result = self._call(df, '{"column": "NonExistentColumn", "value": "X"}')
        assert "not found" in result.lower() or "NonExistentColumn" in result

    def test_empty_segment_returns_no_customers_message(self):
        df = _make_predictions_df()
        result = self._call(df, '{"column": "Contract", "value": "Five year"}')
        assert "No customers" in result

    def test_no_value_key_analyses_full_df(self):
        df = _make_predictions_df(50)
        result = self._call(df, '{"column": "Contract"}')
        assert isinstance(result, str)

    def test_handles_invalid_json_gracefully(self):
        df = _make_predictions_df()
        result = self._call(df, "not json at all")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _suggest_retention_actions
# ---------------------------------------------------------------------------

class TestSuggestRetentionActions:
    def _call(self, profile_json):
        from src.agent.analyst_agent import _suggest_retention_actions
        return _suggest_retention_actions(profile_json)

    def test_monthly_contract_triggers_upgrade_offer(self):
        result = self._call(
            '{"contract": "Month-to-month", "tenure_months": 6, '
            '"monthly_charges": 50, "internet_service": "DSL", "risk_level": "high"}'
        )
        assert "contract" in result.lower() or "upgrade" in result.lower()

    def test_early_tenure_triggers_loyalty_programme(self):
        result = self._call(
            '{"contract": "One year", "tenure_months": 8, '
            '"monthly_charges": 40, "internet_service": "DSL"}'
        )
        assert "loyalty" in result.lower() or "retention" in result.lower() or "welcome" in result.lower()

    def test_high_charge_triggers_account_manager(self):
        result = self._call(
            '{"contract": "One year", "tenure_months": 30, '
            '"monthly_charges": 95, "internet_service": "DSL"}'
        )
        assert "high-value" in result.lower() or "account" in result.lower()

    def test_fiber_optic_triggers_fiber_action(self):
        result = self._call(
            '{"contract": "One year", "tenure_months": 36, '
            '"monthly_charges": 90, "internet_service": "Fiber optic"}'
        )
        assert "fiber" in result.lower()

    def test_returns_at_least_one_bullet(self):
        result = self._call(
            '{"contract": "Two year", "tenure_months": 48, "monthly_charges": 45}'
        )
        assert "•" in result

    def test_handles_invalid_json_gracefully(self):
        result = self._call("not valid json")
        assert isinstance(result, str) and len(result) > 0


# ---------------------------------------------------------------------------
# _calculate_revenue_at_risk
# ---------------------------------------------------------------------------

class TestCalculateRevenueAtRisk:
    def _call(self, df, threshold=0.7):
        from src.agent.analyst_agent import _calculate_revenue_at_risk
        return _calculate_revenue_at_risk(df, threshold)

    def test_returns_string(self):
        df = _make_predictions_df()
        result = self._call(df)
        assert isinstance(result, str)

    def test_contains_monthly_revenue(self):
        df = _make_predictions_df()
        result = self._call(df)
        assert "monthly revenue at risk" in result.lower()

    def test_correct_customer_count(self):
        df = _make_predictions_df(100)
        n_high = (df["churn_probability"] >= 0.7).sum()
        result = self._call(df, threshold=0.7)
        assert str(n_high) in result

    def test_no_customers_at_high_threshold(self):
        df = _make_predictions_df(20)
        df["churn_probability"] = 0.1  # all below 0.9
        result = self._call(df, threshold=0.9)
        assert "No customers" in result

    def test_missing_churn_probability_column(self):
        df = pd.DataFrame({"MonthlyCharges": [50, 60]})
        result = self._call(df)
        assert "not found" in result.lower()


# ---------------------------------------------------------------------------
# run_analyst — fallback when no API key
# ---------------------------------------------------------------------------

class TestRunAnalystFallback:
    def test_returns_string_without_api_key(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        from src.agent.analyst_agent import run_analyst

        df = _make_predictions_df(30)
        result = run_analyst(df, "What is the revenue at risk?")
        assert isinstance(result, str)
        assert len(result) > 20

    def test_result_contains_disabled_message_without_api_key(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        from src.agent.analyst_agent import run_analyst

        df = _make_predictions_df(30)
        result = run_analyst(df, "Summarise high-risk customers")
        assert "disabled" in result.lower() or "groq" in result.lower() or "GROQ" in result
