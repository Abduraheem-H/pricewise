"""Dataset loading and splitting.

On first run the Ames Housing dataset is fetched from OpenML and cached to
`data/ames.csv`; subsequent runs read the local copy, so training is
reproducible and works without a network connection.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from pricewise import config


def load_raw() -> pd.DataFrame:
    """Return the full dataset as a DataFrame, downloading + caching if needed."""
    config.ensure_dirs()
    if config.LOCAL_CSV.exists():
        return pd.read_csv(config.LOCAL_CSV)

    from sklearn.datasets import fetch_openml

    print("Downloading Ames Housing dataset from OpenML (first run only)...")
    bunch = fetch_openml(
        data_id=config.OPENML_DATA_ID,
        as_frame=True,
        data_home=str(config.DATA_DIR),
        parser="auto",
    )
    df = bunch.frame.copy()
    df.to_csv(config.LOCAL_CSV, index=False)
    print(f"Cached dataset -> {config.LOCAL_CSV}  ({df.shape[0]} rows, {df.shape[1]} cols)")
    return df


def load_xy() -> tuple[pd.DataFrame, pd.Series]:
    """Split the raw frame into features X and numeric target y (SalePrice)."""
    df = load_raw()
    df = df.drop(columns=[c for c in config.DROP_COLS if c in df.columns], errors="ignore")

    y = pd.to_numeric(df[config.TARGET], errors="coerce")
    X = df.drop(columns=[config.TARGET])

    # Drop any rows with a missing/unparseable target.
    mask = y.notna()
    return X.loc[mask].reset_index(drop=True), y.loc[mask].reset_index(drop=True)


def get_splits():
    """Deterministic train/test split (same seed everywhere → reproducible eval)."""
    X, y = load_xy()
    return train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
