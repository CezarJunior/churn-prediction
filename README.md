**English** · [Português](./README.pt.md)

---

# Telecom Customer Churn Prediction

A production-grade ML + LLM portfolio project that predicts customer churn for a telecom company using **XGBoost + Optuna**, then lets analysts ask natural-language questions about the results via a **RAG pipeline (LlamaIndex + FAISS + Groq)**.

Live demo: [Hugging Face Spaces](https://huggingface.co/spaces) *(deploy with the steps below)*

---

## Architecture

```
IBM Telco CSV
     │
     ▼
src/pipeline/features.py   ← cleaning + feature engineering
     │
     ▼
src/pipeline/train.py      ← XGBoost + Optuna (20 trials) + MLflow
     │
     ▼
models/best_model.joblib   ← persisted sklearn Pipeline
     │
     ├──► app.py (Gradio)
     │         ├── Predict tab  ← batch scoring + SHAP plot
     │         └── Analyze tab  ← natural-language Q&A
     │
     ├──► src/graph/churn_graph.py   ← LangGraph orchestration
     │         load → preprocess → identify high-risk → LLM report
     │
     ├──► src/rag/insight_rag.py     ← LlamaIndex + FAISS RAG
     │
     └──► src/agent/analyst_agent.py ← LangChain tool-calling agent
```

---

## Features

| Layer | Technology |
|---|---|
| Data | IBM Telco Customer Churn (7,043 rows, 20 features) |
| ML model | XGBoost inside scikit-learn Pipeline |
| Tuning | Optuna (TPE sampler, 20 trials, stratified 5-fold CV) |
| Imbalance | `scale_pos_weight` (XGBoost native class weighting) |
| Explainability | SHAP TreeExplainer — top-15 feature bar chart |
| Experiment tracking | MLflow local server |
| LLM orchestration | LangGraph (4-node DAG: load → predict → filter → report) |
| RAG | LlamaIndex + FAISS + `BAAI/bge-small-en-v1.5` embeddings |
| LLM | Groq API — `llama-3.3-70b-versatile` (free tier) |
| Agent | LangChain tool-calling agent with 3 domain tools |
| Demo UI | Gradio Blocks (2 tabs: Predict + Analyze) |
| Deployment | Hugging Face Spaces (free tier) / Docker |

---

## Quickstart

### 1. Clone and install

```bash
git clone <repo-url>
cd churn-prediction
pip install -r requirements.txt
```

### 2. (Optional) Set your Groq API key

The Predict tab and training work **without** an API key.
The Analyze tab and LLM report generation require Groq.

```bash
cp .env.example .env
# Edit .env and add your key:
# GROQ_API_KEY=gsk_...
```

Get a free key at <https://console.groq.com>.

### 3. Download data and train

```bash
# Download the IBM Telco dataset (~460 KB)
python data/download_data.py

# Train the model (≈ 3–5 minutes for 20 Optuna trials)
python -m src.pipeline.train --trials 20
```

Or with Make:

```bash
make data
make train
```

### 4. Launch the demo

```bash
python app.py
# Open http://localhost:7860
```

Or:

```bash
make app
```

---

## Project Structure

```
churn-prediction/
├── app.py                         # Gradio demo (2 tabs)
├── requirements.txt
├── Makefile
├── Dockerfile
├── .env.example
├── data/
│   ├── download_data.py           # Downloads the IBM Telco CSV
│   └── raw/
│       └── telco_churn.csv        # Downloaded dataset (gitignored)
├── models/
│   └── best_model.joblib          # Trained pipeline (gitignored)
├── notebooks/
│   ├── 01_eda.ipynb               # Exploratory Data Analysis
│   └── 02_model_training.ipynb    # Training walkthrough
├── src/
│   ├── config.py                  # All constants, paths, hyperparams
│   ├── pipeline/
│   │   ├── features.py            # Cleaning + feature engineering
│   │   ├── train.py               # Training, Optuna, MLflow, evaluation
│   │   └── __main__.py            # `python -m src.pipeline.train` entry point
│   ├── graph/
│   │   └── churn_graph.py         # LangGraph 4-node pipeline
│   ├── rag/
│   │   └── insight_rag.py         # LlamaIndex FAISS RAG
│   └── agent/
│       └── analyst_agent.py       # LangChain tool-calling agent
└── mlruns/                        # MLflow experiment logs (gitignored)
```

---

## Model Performance

Typical results on the held-out 20% test set after 20 Optuna trials:

| Metric | Score |
|---|---|
| ROC-AUC | ~0.847 |
| F1-score | ~0.623 |
| Precision | ~0.662 |
| Recall | ~0.588 |
| Accuracy | ~0.805 |

Results vary slightly across runs due to Optuna's stochastic search.

---

## Running the LangGraph Pipeline Standalone

```python
import joblib
from src.graph.churn_graph import run_pipeline

model = joblib.load("models/best_model.joblib")
state = run_pipeline(model_pipeline=model)

print(state["report"])          # Business retention report
print(len(state["high_risk_customers"]))  # Number of high-risk customers
```

---

## Docker

```bash
# Build (trains model inside the image — takes ~5 min on first build)
docker build -t churn-prediction .

# Run
docker run -p 7860:7860 -e GROQ_API_KEY=gsk_... churn-prediction
# Open http://localhost:7860
```

---

## Deploy to Hugging Face Spaces

1. Create a new Space (Gradio SDK, Python 3.10).
2. Push this repository to the Space.
3. Add `GROQ_API_KEY` as a **Secret** in the Space settings.
4. The Space will run `python app.py` automatically.

If you want the model pre-trained in the Docker image rather than trained on each cold start, copy `models/best_model.joblib` into the repo before pushing (remove the `*.joblib` line from `.gitignore`).

---

## Key Design Decisions

**Why XGBoost over LightGBM / CatBoost?**
XGBoost's `scale_pos_weight` parameter handles the 73:27 class imbalance natively without resampling. It also has first-class SHAP support via `shap.TreeExplainer`.

**Why Optuna over GridSearch?**
Optuna's TPE sampler converges to good hyperparameters in 20 trials versus the hundreds of combinations a grid search would require. Bayesian optimisation is strictly better for continuous hyperparameter spaces.

**Why LlamaIndex for RAG and LangChain for the agent?**
LlamaIndex provides a clean document-indexing abstraction that maps naturally to the "customer profile → document" use case. LangChain's `create_tool_calling_agent` is more mature for structured tool-use workflows.

**Why Groq?**
Groq's free API tier provides inference on `llama-3.3-70b-versatile` with very low latency, making it the practical choice for a portfolio demo that should remain free to run.

---

## License

MIT
