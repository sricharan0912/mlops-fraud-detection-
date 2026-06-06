"""LightGBM model — NOT wrapped in an sklearn Pipeline.

Same early-stopping constraint as XGBoost; preprocessor and classifier are separate.
"""

from lightgbm import LGBMClassifier

from training.features.pandas_pipeline import build_preprocessor


def build_lgbm(class_weight: str = "balanced") -> tuple:
    """Return (preprocessor, classifier) as separate objects."""
    preprocessor = build_preprocessor()
    clf = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        class_weight=class_weight,
        metric="average_precision",
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    return preprocessor, clf
