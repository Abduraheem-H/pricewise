# 🏠 PriceWise — House Price Prediction

A classic, end-to-end **machine-learning regression** project: it learns to
predict a home's sale price from its characteristics (size, quality, location,
year built, …). Unlike an LLM wrapper, **the model here is trained from data** —
this project owns the full loop: data → features → train → evaluate.

> **Held-out test performance:** **R² = 0.91**, **MAE ≈ $15,000**, **MAPE 8.7%**
> — i.e. predictions land within ~9% of the true sale price on average.

Built to be **modular**: **Phase 1** trains + evaluates the model; **Phase 2**
serves it via a **FastAPI** `/predict` endpoint and a web form — both on top of
the saved artifact via `pricewise.predict`, no retraining.

---

## 📊 Results

Every tunable model is hyperparameter-searched, then compared by 4-fold
cross-validated RMSE on the training set; the winner is refit and scored once on
a held-out test set.

| Model | CV RMSE | Notes |
| ----- | ------: | ----- |
| Mean baseline | $78,376 | predicts the average price every time |
| Ridge (linear, tuned) | $43,284 | regularized linear regression |
| Random Forest (tuned) | $29,610 | bagged trees |
| HistGradientBoosting (tuned) | $26,969 | gradient-boosted trees |
| **XGBoost (tuned)** ✅ | **$26,465** | gradient boosting — selected |

Winning hyperparameters (XGBoost): `n_estimators=800, max_depth=3,
learning_rate=0.03, subsample=0.7, colsample_bytree=0.7, reg_lambda=5.0`.

**Held-out test set (292 homes the model never saw):**

| Metric | Value | Meaning |
| ------ | ----: | ------- |
| R² | **0.910** | explains 91% of price variance |
| MAE | **$14,997** | average absolute error |
| MAPE | **8.7%** | average error as % of price |
| RMSE | **$26,282** | matches CV RMSE → no overfitting |

Diagnostic plots are written to `reports/`: predicted-vs-actual, residual
distribution, and the top features by permutation importance (typically overall
quality, living area, neighborhood, and garage/basement size).

---

## 🗂️ Dataset

The **Ames Housing** dataset (De Cock, 2011) — 1,460 home sales with 79 features.
It's the modern successor to the Boston Housing set and a standard regression
benchmark. Pulled from **OpenML** (id `42165`, free, no login) on first run and
cached to `data/ames.csv`, so reruns are reproducible and offline.

---

## 🧠 Approach

- **Feature engineering.** A custom `FeatureEngineer` transformer adds
  domain features — total square footage, total bathrooms, house/remodel age,
  total porch area, and presence flags (pool, garage, fireplace, basement, 2nd
  floor). It lives inside the pipeline and always emits the same columns, so
  partial inputs at serve time never break it.
- **One leak-proof pipeline.** `FeatureEngineer → ColumnTransformer → model`,
  all *inside* one estimator — the exact same transforms (impute, scale,
  one-hot) apply at train and predict time, so there's no train/serve skew.
- **Log-target modeling.** Prices are right-skewed, so the model trains on
  `log1p(SalePrice)` (via `TransformedTargetRegressor`) and inverts back to
  dollars automatically — better fit, and the saved model is dollar-native.
- **Hyperparameter tuning.** Each model is tuned with grid/randomized search
  under cross-validation before the final comparison — no hand-picked params.
- **Model selection by evidence.** Five candidates (incl. a naive baseline and
  XGBoost) are cross-validated; the best is chosen by CV RMSE, refit, and tested.
- **Honest evaluation.** Metrics come from a held-out test set, reported in
  interpretable dollars (MAE/RMSE) and percent (MAPE), not just R².

---

## 📁 Project structure

```
pricewise/
├── pricewise/
│   ├── config.py      # paths, dataset id, split + CV constants
│   ├── data.py        # download/cache Ames, train/test split
│   ├── features.py    # ColumnTransformer preprocessing (dtype-driven)
│   ├── model.py       # candidate models to compare
│   ├── train.py       # tune + CV compare → pick best → refit → save
│   ├── evaluate.py    # test metrics + diagnostic plots
│   ├── predict.py     # load model + predict (the serving seam)
│   └── serve/         # Phase 2: FastAPI app + web form
│       ├── app.py     # /predict, /health, /meta + serves the form
│       ├── schemas.py # typed request/response models
│       └── static/index.html
├── tests/             # network-free smoke + API tests
├── data/  models/  reports/   # artifacts (gitignored)
├── Dockerfile  docker-compose.yml   # containerized train + serve
├── .github/workflows/ci.yml         # lint + test + image build
├── ruff.toml  .pre-commit-config.yaml
├── requirements.txt   requirements-serve.txt
└── Makefile
```

---

## 🚀 Run it

Requires **Python ≥ 3.10**.

```bash
# 1. setup (creates .venv and installs deps)
python -m venv .venv
./.venv/Scripts/python -m pip install -r requirements.txt     # Windows
# source .venv/bin/activate && pip install -r requirements.txt  # macOS/Linux

# 2. train (downloads the dataset on first run, ~1 min)
python -m pricewise.train

# 3. evaluate (writes metrics.json + plots to reports/)
python -m pricewise.evaluate

# tests
python -m pytest -q
```

`make setup`, `make train`, `make evaluate`, `make all`, `make test` are shortcuts.

Predict on a new house (most of the 79 fields are optional — missing ones are
imputed):

```python
from pricewise.predict import predict_one
predict_one({"OverallQual": 8, "GrLivArea": 2200, "GarageCars": 2,
             "YearBuilt": 2005, "Neighborhood": "NridgHt"})
# The more fields you supply, the sharper the estimate — with most of the 79
# left blank, the (regularized) model sensibly trends toward typical prices.
```

---

## 🌐 Serving (Phase 2)

A **FastAPI** app loads `models/model.joblib` and exposes a JSON API plus a
single-page web form (no separate frontend needed).

```bash
python -m pricewise.serve        # -> http://127.0.0.1:8000   (or: make serve)
```

Open <http://127.0.0.1:8000> for the form, or call the API directly:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"OverallQual": 8, "GrLivArea": 2200, "GarageCars": 3,
       "YearBuilt": 2006, "Neighborhood": "NridgHt"}'
# {"predicted_price": 250792.09, "currency": "USD", "model_name": "xgboost"}
```

| Method | Path | Description |
| ------ | ---- | ----------- |
| `GET`  | `/` | Web form UI |
| `GET`  | `/health` | Liveness + whether a model is loaded |
| `GET`  | `/meta` | Best model, params, and test metrics |
| `POST` | `/predict` | Predict a sale price from house features |
| `GET`  | `/docs` | Interactive OpenAPI docs (auto-generated) |

Only the high-signal fields are required — the model imputes the rest, and the
more you supply, the sharper the estimate. Train the model first (`make all`) so
`models/model.joblib` exists.

---

## 🐳 Docker & CI

Run the whole thing in one container — the image **trains the model at build
time**, so it's fully self-contained and serves immediately:

```bash
docker compose up --build        # -> http://localhost:8000   (or: make docker-up)
```

| Tool | What it does |
| ---- | ------------ |
| **Docker** | `Dockerfile` + `docker-compose.yml` — installs deps, trains, and serves. |
| **CI (GitHub Actions)** | `.github/workflows/ci.yml` — ruff lint + pytest, then a Docker image build to prove it's deployable. |
| **Ruff** | `make lint` / `ruff check .` — fast linting (`ruff.toml`). |
| **pre-commit** | `pip install pre-commit && pre-commit install` — runs ruff (lint + format) + hygiene on every commit. |

---

## 🎓 What this demonstrates

Genuine ML fundamentals — not an API call: feature engineering, leak-proof
feature pipelines, handling missing data and categoricals, target
transformation, hyperparameter tuning, model selection via cross-validation,
leakage-free evaluation, interpretable metrics, feature importance, and a clean
train → evaluate → serve separation.

## 📝 License

MIT.
