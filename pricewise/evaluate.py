"""Evaluate the saved model on the held-out test set.

Run:  python -m pricewise.evaluate

Writes metrics to reports/metrics.json and saves three diagnostic plots:
predicted-vs-actual, a residual histogram, and the top permutation-importance
features (which inputs actually move the prediction).
"""

from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)

from pricewise import config, data


def main() -> None:
    if not config.MODEL_PATH.exists():
        raise SystemExit("No model found. Run `python -m pricewise.train` first.")

    config.ensure_dirs()
    model = joblib.load(config.MODEL_PATH)
    X_train, X_test, y_train, y_test = data.get_splits()

    preds = model.predict(X_test)
    rmse = float(root_mean_squared_error(y_test, preds))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))
    mape = float(np.mean(np.abs((y_test - preds) / y_test)) * 100)

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape_pct": mape,
        "n_test": int(len(y_test)),
    }
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print("Held-out test performance:")
    print(f"  RMSE   ${rmse:,.0f}")
    print(f"  MAE    ${mae:,.0f}")
    print(f"  MAPE   {mape:.1f}%")
    print(f"  R2     {r2:.3f}")
    print(f"\nSaved metrics -> {config.METRICS_PATH}")

    _plot_pred_vs_actual(y_test, preds)
    _plot_residuals(y_test, preds)
    _plot_importances(model, X_test, y_test)
    print(f"Saved plots   -> {config.REPORTS_DIR}")


def _plot_pred_vs_actual(y_true, preds) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, preds, alpha=0.4, edgecolor="none")
    lo, hi = float(min(y_true.min(), preds.min())), float(max(y_true.max(), preds.max()))
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="perfect")
    ax.set_xlabel("Actual sale price ($)")
    ax.set_ylabel("Predicted sale price ($)")
    ax.set_title("Predicted vs. actual")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / "pred_vs_actual.png", dpi=120)
    plt.close(fig)


def _plot_residuals(y_true, preds) -> None:
    residuals = np.asarray(y_true) - np.asarray(preds)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(residuals, bins=40)
    ax.axvline(0, color="r", linestyle="--", linewidth=1)
    ax.set_xlabel("Residual  (actual − predicted, $)")
    ax.set_ylabel("Count")
    ax.set_title("Residual distribution")
    fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / "residuals.png", dpi=120)
    plt.close(fig)


def _plot_importances(model, X_test, y_test, top_n: int = 15) -> None:
    # Permutation importance is model-agnostic: it measures how much shuffling
    # each *original* input column degrades predictions.
    result = permutation_importance(
        model, X_test, y_test, n_repeats=10,
        random_state=config.RANDOM_STATE, n_jobs=-1,
    )
    importances = (
        pd.Series(result.importances_mean, index=X_test.columns)
        .sort_values(ascending=False)
        .head(top_n)
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    importances.iloc[::-1].plot.barh(ax=ax)
    ax.set_xlabel("Mean importance (drop in R² when shuffled)")
    ax.set_title(f"Top {top_n} features")
    fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / "feature_importance.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
