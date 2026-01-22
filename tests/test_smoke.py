"""Network-free smoke tests for the preprocessing + prediction plumbing.

These run on tiny synthetic data so CI never needs to download the dataset.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from pricewise.features import FeatureEngineer, build_preprocessor

ENGINEERED = [
    "TotalSF", "TotalBath", "TotalPorchSF", "HouseAge", "SinceRemodel",
    "HasPool", "HasGarage", "HasFireplace", "HasBsmt", "HasSecondFloor",
]


def test_feature_engineer_computes_and_is_robust_to_missing_columns():
    # Full inputs: TotalSF should sum the three area columns.
    full = pd.DataFrame({"TotalBsmtSF": [500], "1stFlrSF": [800], "2ndFlrSF": [400]})
    out = FeatureEngineer().fit_transform(full)
    assert out["TotalSF"].iloc[0] == 1700
    for col in ENGINEERED:
        assert col in out.columns

    # Missing source columns must NOT raise and must still emit every column,
    # so the fitted ColumnTransformer keeps a stable schema at serve time.
    sparse = pd.DataFrame({"OverallQual": [7]})
    out2 = FeatureEngineer().fit_transform(sparse)
    for col in ENGINEERED:
        assert col in out2.columns


def _toy():
    X = pd.DataFrame(
        {
            "num1": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
            "num2": [10.0, 9.0, 8.0, 7.0, np.nan, 5.0],
            "cat1": ["a", "b", "a", "b", "a", None],
        }
    )
    y = pd.Series([100.0, 200.0, 150.0, 250.0, 120.0, 300.0])
    return X, y


def test_preprocessor_handles_missing_and_categoricals():
    X, y = _toy()
    pipe = Pipeline([("pre", build_preprocessor()), ("m", LinearRegression())])
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == len(X)
    assert np.isfinite(preds).all()


def test_preprocessor_ignores_unknown_categories_at_predict_time():
    X, y = _toy()
    pipe = Pipeline([("pre", build_preprocessor()), ("m", LinearRegression())])
    pipe.fit(X, y)
    # A previously unseen category must not crash (handle_unknown="ignore").
    unseen = pd.DataFrame({"num1": [3.0], "num2": [6.0], "cat1": ["zzz"]})
    preds = pipe.predict(unseen)
    assert preds.shape == (1,)
