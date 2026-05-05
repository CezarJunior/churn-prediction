# ============================================================
#  Churn Prediction — convenience Makefile
#  Usage: make <target>
# ============================================================

PYTHON   := python
PIP      := pip
VENV     := .venv
TRIALS   := 20

.PHONY: help install data train app test lint clean docker-build docker-run

# ── default target ───────────────────────────────────────────
help:
	@echo ""
	@echo "  Churn Prediction — available make targets"
	@echo ""
	@echo "  install        Install all dependencies into the current environment"
	@echo "  data           Download the IBM Telco churn dataset"
	@echo "  train          Train the model (TRIALS=N to override trial count)"
	@echo "  app            Launch the Gradio demo on http://localhost:7860"
	@echo "  test           Run the test suite with pytest"
	@echo "  lint           Run ruff linter"
	@echo "  clean          Remove cached artefacts (pyc, mlruns, model files)"
	@echo "  docker-build   Build the Docker image"
	@echo "  docker-run     Run the app in Docker on port 7860"
	@echo ""

# ── environment ──────────────────────────────────────────────
install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# ── data ─────────────────────────────────────────────────────
data:
	$(PYTHON) data/download_data.py

# ── training ─────────────────────────────────────────────────
train:
	$(PYTHON) -m src.pipeline.train --trials $(TRIALS)

# ── demo ─────────────────────────────────────────────────────
app:
	$(PYTHON) app.py

# ── tests ────────────────────────────────────────────────────
test:
	$(PYTHON) -m pytest tests/ -v --tb=short

# ── code quality ─────────────────────────────────────────────
lint:
	$(PYTHON) -m ruff check src/ app.py data/

# ── cleanup ──────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf mlruns/ mlartifacts/ .optuna/
	rm -f models/best_model.joblib models/preprocessor.joblib models/_shap_plot.png
	@echo "Clean complete."

# ── Docker ───────────────────────────────────────────────────
docker-build:
	docker build -t churn-prediction:latest .

docker-run:
	docker run --rm -p 7860:7860 \
	    -e GROQ_API_KEY=$${GROQ_API_KEY} \
	    churn-prediction:latest
