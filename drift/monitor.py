"""Evidently drift monitor. Run nightly via Airflow or cron.

Usage:
    python drift/monitor.py
"""

from __future__ import annotations

import json
import os
import pathlib
from datetime import date, timedelta

import pandas as pd
import psycopg2

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://fraud_user:changeme@localhost:5433/fraud")
REFERENCE_PATH = os.getenv(
    "REFERENCE_PATH", "data/processed/reference_distribution.parquet"
)
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.3"))
F2_DROP_THRESHOLD = float(os.getenv("F2_DROP_THRESHOLD", "0.05"))
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "7"))


def load_reference() -> pd.DataFrame:
    path = pathlib.Path(REFERENCE_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"Reference distribution not found at {path}. "
            "Run training/train.py first to generate it."
        )
    return pd.read_parquet(path)


def load_recent_predictions(conn) -> pd.DataFrame:
    cutoff = date.today() - timedelta(days=LOOKBACK_DAYS)
    query = """
        SELECT input_features, is_fraud, fraud_probability
        FROM predictions
        WHERE predicted_at >= %s
    """
    df = pd.read_sql(query, conn, params=(cutoff,))
    if df.empty:
        return df
    features = pd.json_normalize(df["input_features"])
    return pd.concat([features, df[["is_fraud", "fraud_probability"]]], axis=1)


def run_drift_check(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> dict:
    try:
        from evidently import ColumnMapping
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        num_cols = [c for c in reference_df.columns if c not in ("isFraud", "type")]
        col_map = ColumnMapping(numerical_features=num_cols, categorical_features=["type"])

        report = Report(metrics=[DataDriftPreset(drift_share=DRIFT_THRESHOLD)])
        report.run(
            reference_data=reference_df.drop(columns=["isFraud"], errors="ignore"),
            current_data=current_df.drop(columns=["is_fraud", "fraud_probability"], errors="ignore"),
            column_mapping=col_map,
        )
        result = report.as_dict()
        drift_result = result["metrics"][0]["result"]
        return {
            "dataset_drift": drift_result.get("dataset_drift", False),
            "drift_score": drift_result.get("share_of_drifted_columns", 0.0),
            "n_drifted_features": drift_result.get("number_of_drifted_columns", 0),
        }
    except Exception as exc:
        print(f"WARNING: Evidently drift check failed: {exc}")
        return {"dataset_drift": False, "drift_score": 0.0, "n_drifted_features": 0}


def main():
    print("Loading reference distribution...")
    reference_df = load_reference()
    print(f"  Reference: {len(reference_df):,} rows")

    try:
        conn = psycopg2.connect(DATABASE_URL)
    except Exception as exc:
        print(f"ERROR: Could not connect to Postgres: {exc}")
        return

    print(f"Loading last {LOOKBACK_DAYS} days of predictions...")
    current_df = load_recent_predictions(conn)
    print(f"  Current window: {len(current_df):,} rows")

    if len(current_df) < 100:
        print("Not enough predictions for meaningful drift analysis (need >= 100). Skipping.")
        conn.close()
        return

    drift_metrics = run_drift_check(reference_df, current_df)
    print(f"\nDrift metrics: {json.dumps(drift_metrics, indent=2)}")

    from drift.alert import check_and_alert
    triggered = check_and_alert(drift_metrics)

    # Persist report
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO drift_reports (report_date, drift_score, raw_report)
                VALUES (%s, %s, %s)
                """,
                (date.today(), drift_metrics["drift_score"], json.dumps(drift_metrics)),
            )
        conn.commit()
        print("Drift report saved to database.")
    except Exception as exc:
        print(f"WARNING: Could not save drift report: {exc}")

    conn.close()
    return triggered


if __name__ == "__main__":
    main()
