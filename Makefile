# PriceWise — common tasks. On Windows, the explicit commands in the README work
# too if you don't have `make`.

PY := python

.PHONY: help setup train evaluate all test clean

help:
	@echo "setup     - create .venv and install dependencies"
	@echo "train     - train models, select the best, save the artifact"
	@echo "evaluate  - score the saved model on the test set + write plots"
	@echo "all       - train then evaluate"
	@echo "test      - run unit tests"
	@echo "clean     - remove generated artifacts (keeps cached dataset)"

setup:
	$(PY) -m venv .venv
	./.venv/Scripts/python -m pip install --upgrade pip
	./.venv/Scripts/python -m pip install -r requirements.txt

train:
	$(PY) -m pricewise.train

evaluate:
	$(PY) -m pricewise.evaluate

all: train evaluate

test:
	$(PY) -m pytest -q

clean:
	rm -rf models reports/*.png reports/metrics.json
