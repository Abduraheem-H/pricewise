"""API tests using FastAPI's TestClient (no running server needed).

The /predict tests skip gracefully if the model hasn't been trained, so this
suite stays green in CI where the (gitignored) model artifact is absent.
"""

import pytest
from fastapi.testclient import TestClient

from pricewise import config
from pricewise.serve.app import app

client = TestClient(app)


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body


def test_predict_returns_a_price():
    if not config.MODEL_PATH.exists():
        pytest.skip("model not trained; run `python -m pricewise.train`")
    r = client.post(
        "/predict",
        json={"OverallQual": 8, "GrLivArea": 2200, "GarageCars": 2, "Neighborhood": "NridgHt"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["predicted_price"] > 0
    assert body["currency"] == "USD"


def test_predict_rejects_out_of_range_quality():
    # OverallQual must be 1-10 (Field ge/le) -> 422 from validation.
    r = client.post("/predict", json={"OverallQual": 99})
    assert r.status_code == 422
