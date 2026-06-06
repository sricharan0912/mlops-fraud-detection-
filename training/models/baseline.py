"""Logistic Regression baseline model."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from training.features.pandas_pipeline import build_full_pipeline


def build_baseline(class_weight: str = "balanced") -> Pipeline:
    clf = LogisticRegression(
        class_weight=class_weight,
        max_iter=1000,
        solver="saga",  # fastest for large n, supports L1/L2/elasticnet
        n_jobs=-1,
        random_state=42,
    )
    return build_full_pipeline(clf)
