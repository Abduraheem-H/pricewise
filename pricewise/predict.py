"""Load the trained model and predict on new inputs.

This is the seam Phase 2 (the serving API) will build on. `predict_one` accepts
a partial dict of features — any missing columns are filled with NaN and handled
by the pipeline's imputers, so callers don't need to supply all ~79 fields.
"""

from __future__ import annotations

import json
from functools import lru_cache

import joblib
import pandas as pd

from pricewise import config


@lru_cache(maxsize=1)
def _load():
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError("Model not trained yet. Run `python -m pricewise.train`.")
    model = joblib.load(config.MODEL_PATH)
    features = json.loads(config.METADATA_PATH.read_text())["features"]
    return model, features


def predict_one(features: dict) -> float:
    """Predict the sale price ($) for a single house given a dict of features."""
    model, expected = _load()
    # Reindex to the training columns so order matches and missing fields → NaN.
    row = pd.DataFrame([features]).reindex(columns=expected)
    return float(model.predict(row)[0])


def predict_many(rows: list[dict]) -> list[float]:
    model, expected = _load()
    frame = pd.DataFrame(rows).reindex(columns=expected)
    return [float(p) for p in model.predict(frame)]
