#!/bin/sh
set -e
exec mlflow server \
  --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" \
  --default-artifact-root "${MLFLOW_DEFAULT_ARTIFACT_ROOT:-/mlflow/artifacts}" \
  --host 0.0.0.0 \
  --port 5000
