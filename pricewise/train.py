"""Train + tune + select the best model, then persist it.

Run:  python -m pricewise.train

Each candidate is a full pipeline:  FeatureEngineer -> preprocess -> model,
wrapped in a TransformedTargetRegressor that trains on log1p(SalePrice) and
inverts back to dollars on predict (prices are right-skewed).

Tunable models are hyperparameter-searched (grid/random) with cross-validation;
all candidates are compared by CV RMSE (dollars); the winner — already refit on
the full training set by the search — is saved along with its best params.
"""

from __future__ import annotations

import json
import time

import joblib
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline

from pricewise import config, data
from pricewise.features import FeatureEngineer, build_preprocessor
from pricewise.model import HAS_XGB, model_specs

SCORING = "neg_root_mean_squared_error"


def build_pipeline(model) -> TransformedTargetRegressor:
    inner = Pipeline(
        steps=[
            ("fe", FeatureEngineer()),
            ("pre", build_preprocessor()),
            ("model", model),
        ]
    )
    # Train on log1p(price); predictions are automatically expm1'd back to $.
    return TransformedTargetRegressor(regressor=inner, func=np.log1p, inverse_func=np.expm1)


def main() -> None:
    config.ensure_dirs()
    X_train, X_test, y_train, y_test = data.get_splits()
    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)} | Raw features: {X_train.shape[1]}")
    if not HAS_XGB:
        print("(xgboost not installed — skipping it; `pip install xgboost` to include)")
    print("\nTuning + cross-validating candidates (4-fold CV RMSE, lower is better):")

    cv = KFold(n_splits=4, shuffle=True, random_state=config.RANDOM_STATE)

    results: dict[str, dict] = {}
    best_name, best_rmse, best_estimator, best_params = None, float("inf"), None, {}

    for spec in model_specs():
        pipe = build_pipeline(spec.estimator)

        if spec.search == "none":
            scores = -cross_val_score(pipe, X_train, y_train, scoring=SCORING, cv=cv, n_jobs=-1)
            rmse, std = float(scores.mean()), float(scores.std())
            pipe.fit(X_train, y_train)
            estimator, params = pipe, {}
        else:
            if spec.search == "grid":
                search = GridSearchCV(pipe, spec.param_dist, scoring=SCORING, cv=cv, n_jobs=-1)
            else:
                search = RandomizedSearchCV(
                    pipe, spec.param_dist, n_iter=spec.n_iter, scoring=SCORING,
                    cv=cv, random_state=config.RANDOM_STATE, n_jobs=-1,
                )
            search.fit(X_train, y_train)
            rmse = float(-search.best_score_)
            std = float(abs(search.cv_results_["std_test_score"][search.best_index_]))
            estimator = search.best_estimator_  # refit on full training set
            params = {k.replace("regressor__model__", ""): v for k, v in search.best_params_.items()}

        results[spec.name] = {"cv_rmse_mean": rmse, "cv_rmse_std": std, "best_params": params}
        tuned = "  tuned" if params else ""
        print(f"  {spec.name:16s} ${rmse:>10,.0f}  (+/- {std:,.0f}){tuned}")

        if rmse < best_rmse:
            best_name, best_rmse, best_estimator, best_params = spec.name, rmse, estimator, params

    print(f"\nBest: {best_name}  (CV RMSE ${best_rmse:,.0f})")
    if best_params:
        print(f"Best params: {json.dumps(best_params)}")

    joblib.dump(best_estimator, config.MODEL_PATH)

    metadata = {
        "best_model": best_name,
        "best_params": best_params,
        "cv_results": results,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_raw_features": int(X_train.shape[1]),
        "features": list(map(str, X_train.columns)),
        "target": config.TARGET,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    config.METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    print(f"Saved model    -> {config.MODEL_PATH}")
    print(f"Saved metadata -> {config.METADATA_PATH}")
    print("\nNext: python -m pricewise.evaluate")


if __name__ == "__main__":
    main()
