"""Model contract and schema tests — run in CI with the 1k sample model."""

import time

from fastapi.testclient import TestClient

from api.app.main import app

client = TestClient(app)

VALID_TX = {
    "step": 1,
    "type": "TRANSFER",
    "amount": 181.0,
    "oldbalanceOrg": 181.0,
    "newbalanceOrig": 0.0,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 0.0,
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_required_fields():
    r = client.post("/predict", json=VALID_TX)
    assert r.status_code == 200, r.text
    body = r.json()
    assert "fraud_probability" in body
    assert "is_fraud" in body
    assert "threshold" in body
    assert 0.0 <= body["fraud_probability"] <= 1.0


def test_predict_rejects_invalid_type():
    bad = {**VALID_TX, "type": "UNKNOWN"}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_rejects_negative_amount():
    bad = {**VALID_TX, "amount": -100.0}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_latency_under_200ms():
    # Pre-warm
    client.post("/predict", json=VALID_TX)
    start = time.perf_counter()
    client.post("/predict", json=VALID_TX)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < 200, f"Latency {elapsed_ms:.1f}ms exceeds 200ms SLA"


def test_transfer_likely_fraud():
    """A TRANSFER that drains the origin account should score above 0.1."""
    r = client.post("/predict", json=VALID_TX)
    body = r.json()
    assert body["fraud_probability"] > 0.1, "Expected elevated risk for full-drain TRANSFER"
