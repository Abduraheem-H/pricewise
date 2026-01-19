"""Feature engineering + preprocessing.

Two transformers, both designed to live *inside* the model pipeline so the exact
same logic runs at train time and at prediction time (no train/serve skew):

  1. FeatureEngineer — adds domain features (total area, total baths, ages,
     presence flags). It always emits the same columns regardless of which raw
     inputs are present, so partial inputs at serve time never break the fitted
     ColumnTransformer.
  2. build_preprocessor — by dtype: median-impute + scale numerics;
     most-frequent-impute + one-hot encode categoricals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Add engineered features that are known to matter for house prices.

    Stateless (fit is a no-op). Robust to missing source columns: a missing
    input contributes 0, so every engineered column is always produced.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        def num(name: str) -> pd.Series:
            if name in df.columns:
                return pd.to_numeric(df[name], errors="coerce").fillna(0)
            return pd.Series(0.0, index=df.index)

        # Total living/usable area is the single strongest price driver.
        df["TotalSF"] = num("TotalBsmtSF") + num("1stFlrSF") + num("2ndFlrSF")
        # Bathrooms, with half-baths weighted at 0.5.
        df["TotalBath"] = (
            num("FullBath") + 0.5 * num("HalfBath")
            + num("BsmtFullBath") + 0.5 * num("BsmtHalfBath")
        )
        # Outdoor space rolled into one number.
        df["TotalPorchSF"] = (
            num("OpenPorchSF") + num("EnclosedPorch") + num("3SsnPorch")
            + num("ScreenPorch") + num("WoodDeckSF")
        )
        # Ages at time of sale (clip negatives that can arise from odd records).
        df["HouseAge"] = (num("YrSold") - num("YearBuilt")).clip(lower=0)
        df["SinceRemodel"] = (num("YrSold") - num("YearRemodAdd")).clip(lower=0)

        # Presence flags — cheap signal the trees can split on.
        df["HasPool"] = (num("PoolArea") > 0).astype(int)
        df["HasGarage"] = (num("GarageArea") > 0).astype(int)
        df["HasFireplace"] = (num("Fireplaces") > 0).astype(int)
        df["HasBsmt"] = (num("TotalBsmtSF") > 0).astype(int)
        df["HasSecondFloor"] = (num("2ndFlrSF") > 0).astype(int)
        return df


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, make_column_selector(dtype_include=np.number)),
            # "string" covers pandas 3's default string dtype; "object"/"category"
            # cover older frames and explicit categoricals.
            ("cat", categorical, make_column_selector(dtype_include=["object", "category", "string"])),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
