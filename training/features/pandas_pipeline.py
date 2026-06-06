"""Pandas-based feature engineering and scikit-learn preprocessor."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# PaySim has mixed-case column names — pin them here to avoid scatter bugs
NUMERIC_FEATURES = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "balance_diff_orig",
    "balance_diff_dest",
]
CATEGORICAL_FEATURES = ["type"]
FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
LABEL_COL = "isFraud"

TRANSACTION_TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add balance-diff features that are the strongest fraud signals in PaySim."""
    df = df.copy()
    df["balance_diff_orig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["balance_diff_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]
    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OrdinalEncoder(
                    categories=[TRANSACTION_TYPES],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def build_full_pipeline(classifier) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("clf", classifier),
        ]
    )
