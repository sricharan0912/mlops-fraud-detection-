"""Training entry point.

Usage:
    python training/train.py --model baseline --data data/raw/PS_20174392719_1491204439457_log.csv
    python training/train.py --model xgb --data data/raw/PS_20174392719_1491204439457_log.csv
    python training/train.py --model lgbm --data data/raw/...csv
    python training/train.py --model xgb --data data/raw/...csv --smote
"""

from __future__ import annotations

import argparse
import os
import pathlib
import pickle
from datetime import UTC, datetime

import pandas as pd
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn
from training.evaluate import best_threshold_for_f2, evaluate_at_threshold, fairness_report
from training.features.pandas_pipeline import FEATURE_COLS, LABEL_COL, engineer_features

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODELS_DIR = pathlib.Path("models")
TEST_SIZE = 0.2
VAL_SIZE = 0.1  # fraction of train used for early-stopping validation
RANDOM_STATE = 42
PROMOTION_F2_BAR = 0.70


def load_data(path: str) -> tuple[pd.DataFrame, pd.Series]:
    print(f"Loading data from {path} ...")
    df = pd.read_csv(path)
    df = engineer_features(df)
    X = df[FEATURE_COLS]
    y = df[LABEL_COL]
    n_fraud = int(y.sum())
    print(f"  {len(df):,} rows  |  {n_fraud:,} fraud ({n_fraud/len(df):.3%})")
    return X, y


def _apply_smote(X_train, y_train):
    from imblearn.over_sampling import SMOTE

    print("Applying SMOTE on training set ...")
    sm = SMOTE(random_state=RANDOM_STATE, n_jobs=-1)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"  After SMOTE: {len(X_res):,} rows  |  fraud {y_res.sum():,}")
    return X_res, y_res


def _save_artifact(artifact: dict, model_name: str) -> pathlib.Path:
    MODELS_DIR.mkdir(exist_ok=True)
    path = MODELS_DIR / f"{model_name}.pkl"
    with open(path, "wb") as fh:
        pickle.dump(artifact, fh)
    print(f"Saved artifact → {path}")
    return path


def train_baseline(X_train, y_train, X_test, y_test, smote: bool, run_name: str):
    from training.models.baseline import build_baseline

    pipeline = build_baseline()

    Xtr, ytr = (_apply_smote(X_train, y_train) if smote else (X_train, y_train))
    # Baseline is a full sklearn Pipeline — fit directly
    pipeline.fit(Xtr, ytr)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    threshold, _ = best_threshold_for_f2(y_test.values, y_prob)
    metrics = evaluate_at_threshold(y_test.values, y_prob, threshold)

    artifact = {"pipeline": pipeline, "threshold": threshold, **metrics, "model": "baseline"}
    return artifact, pipeline, y_prob, threshold, metrics


def train_xgb(X_train, y_train, X_val, y_val, X_test, y_test, smote: bool, run_name: str):
    from training.models.xgboost_model import build_xgb

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / n_pos if not smote else 1.0

    preprocessor, clf = build_xgb(scale_pos_weight=scale_pos_weight)

    Xtr, ytr = (_apply_smote(X_train, y_train) if smote else (X_train, y_train))

    # Fit preprocessor on train, transform train/val/test
    X_tr_t = preprocessor.fit_transform(Xtr, ytr)
    X_val_t = preprocessor.transform(X_val)
    X_te_t = preprocessor.transform(X_test)

    clf.fit(
        X_tr_t, ytr,
        eval_set=[(X_val_t, y_val.values)],
        verbose=50,
    )

    y_prob = clf.predict_proba(X_te_t)[:, 1]
    threshold, _ = best_threshold_for_f2(y_test.values, y_prob)
    metrics = evaluate_at_threshold(y_test.values, y_prob, threshold)

    # Wrap preprocessor + clf into a lightweight dict (not sklearn Pipeline)
    artifact = {
        "preprocessor": preprocessor,
        "clf": clf,
        "threshold": threshold,
        **metrics,
        "model": "xgb",
    }
    return artifact, y_prob, threshold, metrics


def train_lgbm(X_train, y_train, X_val, y_val, X_test, y_test, smote: bool, run_name: str):
    from training.models.lgbm_model import build_lgbm

    preprocessor, clf = build_lgbm()

    Xtr, ytr = (_apply_smote(X_train, y_train) if smote else (X_train, y_train))

    X_tr_t = preprocessor.fit_transform(Xtr, ytr)
    X_val_t = preprocessor.transform(X_val)
    X_te_t = preprocessor.transform(X_test)

    clf.fit(
        X_tr_t, ytr,
        eval_set=[(X_val_t, y_val.values)],
        callbacks=[],
    )

    y_prob = clf.predict_proba(X_te_t)[:, 1]
    threshold, _ = best_threshold_for_f2(y_test.values, y_prob)
    metrics = evaluate_at_threshold(y_test.values, y_prob, threshold)

    artifact = {
        "preprocessor": preprocessor,
        "clf": clf,
        "threshold": threshold,
        **metrics,
        "model": "lgbm",
    }
    return artifact, y_prob, threshold, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["baseline", "xgb", "lgbm"], default="baseline")
    parser.add_argument("--data", required=True)
    parser.add_argument("--smote", action="store_true")
    parser.add_argument("--output", default=None, help="Override output pickle path")
    args = parser.parse_args()

    X, y = load_data(args.data)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    val_frac = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_frac, stratify=y_trainval, random_state=RANDOM_STATE
    )

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.model}-smote={args.smote}-{timestamp}"

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("fraud-detection")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model": args.model,
            "smote": args.smote,
            "data": pathlib.Path(args.data).name,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "fraud_rate_train": float(y_train.mean()),
        })

        if args.model == "baseline":
            artifact, pipeline, y_prob, threshold, metrics = train_baseline(
                X_train, y_train, X_test, y_test, args.smote, run_name
            )
            mlflow_model = pipeline
        elif args.model == "xgb":
            artifact, y_prob, threshold, metrics = train_xgb(
                X_train, y_train, X_val, y_val, X_test, y_test, args.smote, run_name
            )
            mlflow_model = artifact["clf"]
        else:
            artifact, y_prob, threshold, metrics = train_lgbm(
                X_train, y_train, X_val, y_val, X_test, y_test, args.smote, run_name
            )
            mlflow_model = artifact["clf"]

        mlflow.log_metrics({
            "f2_test": metrics["f2"],
            "avg_precision_test": metrics["avg_precision"],
            "recall_test": metrics["recall"],
            "precision_test": metrics["precision"],
            "fpr_test": metrics["fpr"],
            "threshold": threshold,
        })
        mlflow.log_dict({"threshold": threshold, "f2": metrics["f2"]}, "threshold.json")

        # Save reference distribution for drift monitoring
        ref_path = "data/processed/reference_distribution.parquet"
        pathlib.Path("data/processed").mkdir(exist_ok=True)
        X_train.assign(isFraud=y_train.values).to_parquet(ref_path, index=False)
        mlflow.log_artifact(ref_path, artifact_path="reference")

        # Log and register
        try:
            mlflow.sklearn.log_model(
                mlflow_model,
                artifact_path="model",
                registered_model_name="fraud-detector",
            )
        except Exception:
            # MLflow may not be running during CI; log warning, continue
            print("WARNING: MLflow model registration skipped (tracking server unavailable)")

        # Save local pickle
        out_path = args.output or str(MODELS_DIR / f"{args.model}.pkl")
        MODELS_DIR.mkdir(exist_ok=True)
        with open(out_path, "wb") as fh:
            pickle.dump(artifact, fh)

        # Fairness report
        fair_df = fairness_report(y_test, y_prob, threshold, X_test["type"], "type")
        print("\nFairness (FPR by transaction type):")
        print(fair_df.to_string(index=False))

        print(f"\n{'='*50}")
        print(f"Model:     {args.model}  (SMOTE={args.smote})")
        print(f"F2:        {metrics['f2']:.4f}  (bar: {PROMOTION_F2_BAR})")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"AvgPrec:   {metrics['avg_precision']:.4f}")
        print(f"Threshold: {threshold:.3f}")
        print(f"Artifact:  {out_path}")
        if metrics["f2"] >= PROMOTION_F2_BAR:
            print("PROMOTION BAR MET ✓")
        else:
            print(f"Promotion bar NOT met ({metrics['f2']:.4f} < {PROMOTION_F2_BAR})")


if __name__ == "__main__":
    main()
