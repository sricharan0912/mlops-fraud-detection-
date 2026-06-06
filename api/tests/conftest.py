"""Fixtures for API tests — uses a model trained on the 1k sample."""

import os
import pathlib
import subprocess

import pytest
from fastapi.testclient import TestClient

# Point at the sample-trained model so CI never needs the full dataset
SAMPLE_MODEL_PATH = "models/ci_model.pkl"


def _ensure_sample_model():
    if not pathlib.Path(SAMPLE_MODEL_PATH).exists():
        subprocess.run(
            [
                "python",
                "training/train.py",
                "--model", "baseline",
                "--data", "data/sample/paysim_sample_1k.csv",
                "--output", SAMPLE_MODEL_PATH,
            ],
            check=True,
        )


@pytest.fixture(scope="session", autouse=True)
def sample_model():
    _ensure_sample_model()
    os.environ["MODEL_PATH"] = SAMPLE_MODEL_PATH
    os.environ["USE_MLFLOW_REGISTRY"] = "false"


@pytest.fixture(scope="session")
def client(sample_model):
    # Import after env vars are set
    from api.app.main import app
    from api.app.services.predictor import get_predictor

    get_predictor.cache_clear()
    with TestClient(app) as c:
        yield c


VALID_TX = {
    "step": 1,
    "type": "TRANSFER",
    "amount": 181.0,
    "oldbalanceOrg": 181.0,
    "newbalanceOrig": 0.0,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 0.0,
}
