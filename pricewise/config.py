"""Central configuration: paths, dataset id, and training constants.

Keeping these in one place means the data, training, evaluation, and (later)
serving code all agree on where artifacts live and how the split is made.
"""

from __future__ import annotations

from pathlib import Path

# Project root is the parent of this package directory.
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# Dataset: the Ames Housing dataset (De Cock, 2011), the modern successor to the
# Boston Housing set. Hosted on OpenML as dataset id 42165 ("house_prices") — a
# free, no-login source. We cache a CSV copy locally after the first download so
# reruns (and the Phase-2 server) work offline.
OPENML_DATA_ID = 42165
LOCAL_CSV = DATA_DIR / "ames.csv"

TARGET = "SalePrice"
DROP_COLS = ["Id"]  # identifiers carry no predictive signal

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Artifacts produced by training / evaluation.
MODEL_PATH = MODELS_DIR / "model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"
METRICS_PATH = REPORTS_DIR / "metrics.json"


def ensure_dirs() -> None:
    for d in (DATA_DIR, MODELS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
