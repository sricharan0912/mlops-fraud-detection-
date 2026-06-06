"""Airflow DAG: check retrain flag → train → validate → conditional promote."""

from __future__ import annotations

import os
import subprocess

from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.utils.dates import days_ago

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
PROMOTION_F2_BAR = 0.70
DATA_PATH = os.getenv(
    "TRAINING_DATA_PATH",
    "data/raw/PS_20174392719_1491204439457_log.csv",
)


def check_retrain_needed(**ctx):
    """Query Postgres for any pending retrain requests."""
    try:
        import psycopg2
        conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM retrain_requests WHERE status = 'pending'")
            count = cur.fetchone()[0]
        conn.close()
        if count > 0:
            print(f"Found {count} pending retrain request(s). Proceeding.")
            return "run_training"
        print("No pending retrain requests. Skipping.")
        return "skip_training"
    except Exception as exc:
        print(f"WARNING: Could not check retrain flag ({exc}). Proceeding anyway.")
        return "run_training"


def run_training(**ctx):
    result = subprocess.run(
        ["python", "training/train.py", "--model", "lgbm", "--data", DATA_PATH],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("Training failed")


def validate_model(**ctx):
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()

    runs = client.search_runs(
        experiment_ids=[client.get_experiment_by_name("fraud-detection").experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError("No MLflow runs found")

    run = runs[0]
    f2 = float(run.data.metrics.get("f2_test", 0))
    run_id = run.info.run_id
    print(f"Latest run: {run_id}  |  F2={f2:.4f}")

    ctx["ti"].xcom_push(key="f2", value=f2)
    ctx["ti"].xcom_push(key="run_id", value=run_id)

    return "promote_model" if f2 >= PROMOTION_F2_BAR else "reject_model"


def promote_model(**ctx):
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()
    run_id = ctx["ti"].xcom_pull(key="run_id")
    f2 = ctx["ti"].xcom_pull(key="f2")

    versions = client.search_model_versions(f"run_id='{run_id}'")
    if versions:
        version = versions[0].version
        client.set_registered_model_alias("fraud-detector", "champion", version)
        print(f"Promoted version {version} (F2={f2:.4f}) to @champion")
    else:
        print("No registered model version found for this run.")

    # Mark retrain requests as done
    try:
        import psycopg2
        conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        with conn.cursor() as cur:
            cur.execute("UPDATE retrain_requests SET status = 'done' WHERE status = 'pending'")
        conn.commit()
        conn.close()
    except Exception as exc:
        print(f"WARNING: Could not update retrain_requests: {exc}")


def reject_model(**ctx):
    f2 = ctx["ti"].xcom_pull(key="f2")
    print(f"Model REJECTED: F2={f2:.4f} < bar={PROMOTION_F2_BAR}. Keeping current champion.")


def skip_training(**ctx):
    print("No retrain needed. Champion model unchanged.")


with DAG(
    dag_id="fraud_retrain",
    schedule="0 2 * * *",  # 2 AM daily
    start_date=days_ago(1),
    catchup=False,
    tags=["fraud", "mlops"],
    doc_md="""
    ## Fraud Model Retrain DAG

    1. Check `retrain_requests` table for pending requests (written by drift monitor)
    2. If pending: train LightGBM on full dataset
    3. Validate: F2 >= 0.70 on test set
    4. Promote to `@champion` alias in MLflow registry, or reject
    """,
) as dag:

    check = BranchPythonOperator(
        task_id="check_retrain_needed",
        python_callable=check_retrain_needed,
    )
    train = PythonOperator(task_id="run_training", python_callable=run_training)
    validate = BranchPythonOperator(task_id="validate_model", python_callable=validate_model)
    promote = PythonOperator(task_id="promote_model", python_callable=promote_model)
    reject = PythonOperator(task_id="reject_model", python_callable=reject_model)
    skip = PythonOperator(task_id="skip_training", python_callable=skip_training)

    check >> [train, skip]
    train >> validate >> [promote, reject]
