# syntax=docker/dockerfile:1

# ──────────────────────────────────────────────────────────────────────────────
# PriceWise — trains the model at build time and serves it.
#
# The build runs `pricewise.train` (downloading the Ames dataset from OpenML
# once), so the resulting image ships with a ready model.joblib and serves
# predictions immediately — no volume or external artifact needed.
#
#   docker build -t pricewise .
#   docker run -p 8000:8000 pricewise      # -> http://localhost:8000
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app

# libgomp1 provides the OpenMP runtime that xgboost / scikit-learn need.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install deps first (cached layer) — serve subset, no matplotlib/test tooling.
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY pricewise ./pricewise

# Train at build so the image is self-contained and instantly serveable.
RUN python -m pricewise.train

# Run as a non-root user.
RUN useradd --create-home app && chown -R app /app
USER app

EXPOSE 8000
CMD ["python", "-m", "pricewise.serve"]
