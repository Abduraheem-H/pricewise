"""Candidate models + their hyperparameter search spaces.

Each ModelSpec pairs an estimator with a search strategy. The training script
tunes the tunable ones (grid/random search), cross-validates all of them, and
keeps the overall winner. XGBoost is included when installed.

Param keys are prefixed `regressor__model__` because each estimator sits as the
"model" step of a Pipeline that is itself wrapped in a TransformedTargetRegressor
(see train.build_pipeline).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

from pricewise import config

try:
    from xgboost import XGBRegressor

    HAS_XGB = True
except ImportError:  # xgboost is optional — the rest still works without it
    HAS_XGB = False


@dataclass
class ModelSpec:
    name: str
    estimator: object
    param_dist: Optional[dict] = None
    search: str = "none"  # "none" | "grid" | "random"
    n_iter: int = 20


def _p(**kwargs) -> dict:
    """Prefix bare param names to target the pipeline's model step."""
    return {f"regressor__model__{k}": v for k, v in kwargs.items()}


def model_specs() -> list[ModelSpec]:
    rs = config.RANDOM_STATE
    specs: list[ModelSpec] = [
        # The bar every real model must clear.
        ModelSpec("baseline_mean", DummyRegressor(strategy="mean")),
        ModelSpec(
            "ridge",
            Ridge(),
            _p(alpha=[0.1, 1.0, 3.0, 10.0, 30.0, 100.0]),
            search="grid",
        ),
        ModelSpec(
            "random_forest",
            RandomForestRegressor(random_state=rs, n_jobs=-1),
            _p(
                n_estimators=[300, 500],
                max_depth=[None, 20, 30],
                max_features=["sqrt", 0.3, 0.5],
                min_samples_leaf=[1, 2, 4],
            ),
            search="random",
            n_iter=8,
        ),
        ModelSpec(
            "hist_gbr",
            HistGradientBoostingRegressor(random_state=rs),
            _p(
                learning_rate=[0.03, 0.05, 0.1],
                max_iter=[300, 500, 800],
                max_leaf_nodes=[15, 31, 63],
                min_samples_leaf=[10, 20, 30],
                l2_regularization=[0.0, 1.0],
            ),
            search="random",
            n_iter=20,
        ),
    ]
    if HAS_XGB:
        specs.append(
            ModelSpec(
                "xgboost",
                XGBRegressor(
                    tree_method="hist",
                    objective="reg:squarederror",
                    random_state=rs,
                    n_jobs=-1,
                ),
                _p(
                    n_estimators=[400, 800],
                    learning_rate=[0.03, 0.05, 0.1],
                    max_depth=[3, 4, 6],
                    subsample=[0.7, 0.9, 1.0],
                    colsample_bytree=[0.7, 0.9, 1.0],
                    reg_lambda=[1.0, 5.0],
                ),
                search="random",
                n_iter=20,
            )
        )
    return specs
