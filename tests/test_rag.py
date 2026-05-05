"""
Tests for src/rag/insight_rag.py

We test only what doesn't require downloading model weights or a Groq API key:
- _row_to_text produces non-empty, well-formed strings
- ChurnRAG initialises correctly
- Module-level wrapper functions exist and are callable

The heavy integration tests (build_churn_index, query_churn_insights) require
sentence-transformers and faiss at runtime; they are marked with
`pytest.mark.integration` and skipped by default.
"""

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# _row_to_text
# ---------------------------------------------------------------------------

class TestRowToText:
    def _call(self, row: pd.Series) -> str:
        from src.rag.insight_rag import _row_to_text
        return _row_to_text(row)

    def _make_row(self, **kwargs) -> pd.Series:
        defaults = {
            "tenure": 12,
            "MonthlyCharges": 75.50,
            "TotalCharges": 906.0,
            "Contract": "Month-to-month",
            "InternetService": "Fiber optic",
            "PaymentMethod": "Electronic check",
            "churn_probability": 0.82,
            "risk_label": "Critical",
        }
        defaults.update(kwargs)
        return pd.Series(defaults)

    def test_returns_string(self):
        row = self._make_row()
        result = self._call(row)
        assert isinstance(result, str)

    def test_starts_with_customer_profile(self):
        row = self._make_row()
        result = self._call(row)
        assert result.startswith("Customer profile")

    def test_contains_tenure(self):
        row = self._make_row(tenure=24)
        result = self._call(row)
        assert "24" in result

    def test_contains_contract_type(self):
        row = self._make_row(**{"Contract": "Two year"})
        result = self._call(row)
        assert "Two year" in result

    def test_churn_probability_formatted_as_percent(self):
        row = self._make_row(churn_probability=0.75)
        result = self._call(row)
        assert "75.0%" in result

    def test_handles_missing_columns_gracefully(self):
        # A minimal row — should not raise
        row = pd.Series({"tenure": 5})
        result = self._call(row)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_pipe_separated_fields(self):
        row = self._make_row()
        result = self._call(row)
        assert "|" in result


# ---------------------------------------------------------------------------
# ChurnRAG initialisation
# ---------------------------------------------------------------------------

class TestChurnRAGInit:
    def test_default_embedding_model(self):
        from src.rag.insight_rag import ChurnRAG, _EMBEDDING_MODEL
        rag = ChurnRAG()
        assert rag.embedding_model == _EMBEDDING_MODEL

    def test_custom_embedding_model(self):
        from src.rag.insight_rag import ChurnRAG
        rag = ChurnRAG(embedding_model="sentence-transformers/all-MiniLM-L6-v2")
        assert rag.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"

    def test_embed_model_is_none_before_first_build(self):
        from src.rag.insight_rag import ChurnRAG
        rag = ChurnRAG()
        assert rag._embed_model is None


# ---------------------------------------------------------------------------
# Module-level wrappers exist
# ---------------------------------------------------------------------------

class TestModuleLevelWrappers:
    def test_build_churn_index_is_callable(self):
        from src.rag.insight_rag import build_churn_index
        assert callable(build_churn_index)

    def test_query_churn_insights_is_callable(self):
        from src.rag.insight_rag import query_churn_insights
        assert callable(query_churn_insights)


# ---------------------------------------------------------------------------
# Integration — only runs when FAISS + sentence-transformers are installed
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestChurnRAGIntegration:
    """Requires: faiss-cpu, sentence-transformers, llama-index installed."""

    def _make_high_risk_df(self, n: int = 5) -> pd.DataFrame:
        import numpy as np
        rng = np.random.default_rng(0)
        contracts = ["Month-to-month", "One year", "Two year"]
        records = []
        for i in range(n):
            tenure = int(rng.integers(1, 24))
            monthly = float(rng.uniform(70, 110))
            records.append(
                {
                    "tenure": tenure,
                    "MonthlyCharges": monthly,
                    "TotalCharges": monthly * tenure,
                    "Contract": rng.choice(contracts),
                    "InternetService": "Fiber optic",
                    "PaymentMethod": "Electronic check",
                    "churn_probability": float(rng.uniform(0.71, 0.99)),
                    "risk_label": "High",
                    "tenure_group": "0-12",
                    "HasMultipleServices": int(rng.integers(1, 5)),
                    "IsLongTermContract": "0",
                }
            )
        return pd.DataFrame(records)

    def test_build_index_returns_index_object(self):
        from src.rag.insight_rag import ChurnRAG
        rag = ChurnRAG()
        df = self._make_high_risk_df(5)
        index = rag.build_churn_index(df)
        assert index is not None

    def test_query_returns_string(self):
        from src.rag.insight_rag import ChurnRAG
        rag = ChurnRAG()
        df = self._make_high_risk_df(5)
        index = rag.build_churn_index(df)
        result = rag.query_churn_insights(index, "What contract type is most common?")
        assert isinstance(result, str)
        assert len(result) > 0
