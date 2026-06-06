"""XGBoost model — NOT wrapped in an sklearn Pipeline.

XGBoost's early_stopping_rounds requires passing eval_set at fit time, which breaks
sklearn Pipeline's fit API. We therefore return only the preprocessor + the raw XGBClassifier
and handle the fit loop in train.py.
"""


from xgboost import XGBClassifier

from training.features.pandas_pipeline import build_preprocessor


def build_xgb(scale_pos_weight: float | None = None) -> tuple:
    """Return (preprocessor, classifier) as separate objects."""
    preprocessor = build_preprocessor()
    clf = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        early_stopping_rounds=20,
        tree_method="hist",  # fast on CPU; swap for "gpu_hist" with GPU
        n_jobs=-1,
        random_state=42,
        verbosity=0,
    )
    return preprocessor, clf
