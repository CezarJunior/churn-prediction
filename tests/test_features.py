"""
Tests for src/pipeline/features.py

Covers:
- clean_data correctness
- engineer_features produces expected columns
- build_preprocessor produces expected output shape
- get_feature_target_split returns aligned X and y
"""

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers — build a minimal synthetic Telco-like dataframe
# ---------------------------------------------------------------------------

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


def _make_raw_df(n: int = 10, seed: int = 0) -> pd.DataFrame:
    """Return a small synthetic raw Telco dataframe."""
    rng = np.random.default_rng(seed)
    contracts = ["Month-to-month", "One year", "Two year"]
    internet = ["Fiber optic", "DSL", "No"]
    payment = [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    records = []
    for i in range(n):
        tenure = int(rng.integers(1, 72))
        monthly = float(rng.uniform(20, 110))
        records.append(
            {
                "customerID": f"TEST-{i:04d}",
                "gender": rng.choice(["Male", "Female"]),
                "SeniorCitizen": int(rng.integers(0, 2)),
                "Partner": rng.choice(["Yes", "No"]),
                "Dependents": rng.choice(["Yes", "No"]),
                "tenure": tenure,
                "PhoneService": rng.choice(["Yes", "No"]),
                "MultipleLines": rng.choice(["Yes", "No", "No phone service"]),
                "InternetService": rng.choice(internet),
                "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"]),
                "OnlineBackup": rng.choice(["Yes", "No", "No internet service"]),
                "DeviceProtection": rng.choice(["Yes", "No", "No internet service"]),
                "TechSupport": rng.choice(["Yes", "No", "No internet service"]),
                "StreamingTV": rng.choice(["Yes", "No", "No internet service"]),
                "StreamingMovies": rng.choice(["Yes", "No", "No internet service"]),
                "Contract": rng.choice(contracts),
                "PaperlessBilling": rng.choice(["Yes", "No"]),
                "PaymentMethod": rng.choice(payment),
                "MonthlyCharges": round(monthly, 2),
                "TotalCharges": str(round(monthly * tenure, 2)),
                "Churn": rng.choice(["Yes", "No"]),
            }
        )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# clean_data
# ---------------------------------------------------------------------------

class TestCleanData:
    def setup_method(self):
        from src.pipeline.features import clean_data
        self.clean_data = clean_data
        self.raw = _make_raw_df(20)

    def test_removes_customer_id(self):
        df = self.clean_data(self.raw)
        assert "customerID" not in df.columns

    def test_total_charges_numeric(self):
        df = self.clean_data(self.raw)
        assert pd.api.types.is_float_dtype(df["TotalCharges"]), (
            "TotalCharges should be float after cleaning"
        )

    def test_no_nulls_in_total_charges(self):
        df = self.clean_data(self.raw)
        assert df["TotalCharges"].isna().sum() == 0

    def test_churn_encoded_as_int(self):
        df = self.clean_data(self.raw)
        assert set(df["Churn"].unique()).issubset({0, 1}), (
            "Churn should be 0/1 integers after cleaning"
        )

    def test_senior_citizen_is_string(self):
        df = self.clean_data(self.raw)
        # SeniorCitizen should be cast to str for consistent OHE handling
        assert df["SeniorCitizen"].dtype == object

    def test_total_charges_nan_filled_to_zero(self):
        raw = self.raw.copy()
        raw.loc[0, "TotalCharges"] = "  "  # Whitespace — occurs at tenure=0
        df = self.clean_data(raw)
        assert df["TotalCharges"].isna().sum() == 0

    def test_dataframe_length_preserved(self):
        df = self.clean_data(self.raw)
        assert len(df) == len(self.raw)


# ---------------------------------------------------------------------------
# engineer_features
# ---------------------------------------------------------------------------

class TestEngineerFeatures:
    def setup_method(self):
        from src.pipeline.features import clean_data, engineer_features
        raw = _make_raw_df(30)
        self.df = engineer_features(clean_data(raw))

    def test_tenure_group_exists(self):
        assert "tenure_group" in self.df.columns

    def test_tenure_group_valid_labels(self):
        valid = {"0-12", "13-24", "25-48", "49-60", "61+"}
        actual = set(self.df["tenure_group"].unique())
        assert actual.issubset(valid), f"Unexpected tenure_group values: {actual - valid}"

    def test_avg_monthly_spend_exists(self):
        assert "AvgMonthlySpend" in self.df.columns

    def test_avg_monthly_spend_non_negative(self):
        assert (self.df["AvgMonthlySpend"] >= 0).all()

    def test_has_multiple_services_exists(self):
        assert "HasMultipleServices" in self.df.columns

    def test_has_multiple_services_range(self):
        col = self.df["HasMultipleServices"]
        assert col.min() >= 0
        assert col.max() <= len(_SERVICE_COLS)

    def test_is_long_term_contract_exists(self):
        assert "IsLongTermContract" in self.df.columns

    def test_is_long_term_contract_values(self):
        assert set(self.df["IsLongTermContract"].unique()).issubset({"0", "1"})


# ---------------------------------------------------------------------------
# build_preprocessor
# ---------------------------------------------------------------------------

class TestBuildPreprocessor:
    def setup_method(self):
        from src.pipeline.features import (
            build_preprocessor,
            clean_data,
            engineer_features,
            get_feature_target_split,
        )
        raw = _make_raw_df(50)
        df_feat = engineer_features(clean_data(raw))
        X, _ = get_feature_target_split(df_feat)
        self.preprocessor = build_preprocessor()
        self.X_transformed = self.preprocessor.fit_transform(X)

    def test_output_is_numpy_array(self):
        assert isinstance(self.X_transformed, np.ndarray)

    def test_output_has_no_nans(self):
        assert not np.isnan(self.X_transformed).any()

    def test_output_row_count_matches_input(self):
        assert self.X_transformed.shape[0] == 50

    def test_output_has_more_columns_than_input(self):
        # OHE expands categoricals, so output cols > raw feature count
        from src.pipeline.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
        raw_cols = len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)
        assert self.X_transformed.shape[1] > raw_cols


# ---------------------------------------------------------------------------
# get_feature_target_split
# ---------------------------------------------------------------------------

class TestGetFeatureTargetSplit:
    def setup_method(self):
        from src.pipeline.features import (
            clean_data,
            engineer_features,
            get_feature_target_split,
        )
        raw = _make_raw_df(25)
        df_feat = engineer_features(clean_data(raw))
        self.X, self.y = get_feature_target_split(df_feat)

    def test_x_and_y_same_length(self):
        assert len(self.X) == len(self.y)

    def test_y_is_binary(self):
        assert set(self.y.unique()).issubset({0, 1})

    def test_churn_not_in_x(self):
        assert "Churn" not in self.X.columns

    def test_x_has_expected_feature_columns(self):
        from src.pipeline.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
        expected = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
        # Only features that exist in the df are retained
        assert set(self.X.columns).issubset(expected)
