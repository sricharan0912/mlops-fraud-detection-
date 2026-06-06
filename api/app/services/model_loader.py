"""MLflow model registry loader — used when MODEL_PATH is not set."""

from __future__ import annotations

import json

import mlflow
import mlflow.sklearn  # noqa: F401


def load_champion_model(tracking_uri: str, model_name: str = "fraud-detector", alias: str = "champion"):
    """Load the @champion model from the MLflow registry."""
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()

    model_uri = f"models:/{model_name}@{alias}"
    pipeline = mlflow.sklearn.load_model(model_uri)

    # Fetch threshold from the run artifact
    version = client.get_model_version_by_alias(model_name, alias)
    run_id: str = version.run_id or ""
    local_path = client.download_artifacts(run_id, "threshold.json")
    threshold_data = json.loads(open(local_path).read())
    threshold = float(threshold_data["threshold"])

    return {"pipeline": pipeline, "threshold": threshold, "model": model_name}
