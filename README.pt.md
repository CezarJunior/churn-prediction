# Previsão de Churn de Clientes — Telecom

Projeto de portfólio nível produção que combina **ML + LLM** para prever cancelamentos de clientes em uma operadora de telecomunicações usando **XGBoost + Optuna**, e permite que analistas façam perguntas em linguagem natural sobre os resultados via **pipeline RAG (LlamaIndex + FAISS + Groq)**.

Demo ao vivo: [Hugging Face Spaces](https://huggingface.co/spaces) *(siga os passos abaixo para subir)*

---

## Arquitetura

```
IBM Telco CSV
     │
     ▼
src/pipeline/features.py   ← limpeza + engenharia de features
     │
     ▼
src/pipeline/train.py      ← XGBoost + Optuna (20 trials) + MLflow
     │
     ▼
models/best_model.joblib   ← Pipeline sklearn salvo em disco
     │
     ├──► app.py (Gradio)
     │         ├── Aba Prever    ← pontuação em lote + gráfico SHAP
     │         └── Aba Analisar  ← perguntas em linguagem natural
     │
     ├──► src/graph/churn_graph.py   ← orquestração com LangGraph
     │         carregar → pré-processar → identificar alto risco → relatório LLM
     │
     ├──► src/rag/insight_rag.py     ← RAG com LlamaIndex + FAISS
     │
     └──► src/agent/analyst_agent.py ← agente LangChain com chamada de ferramentas
```

---

## Funcionalidades

| Camada | Tecnologia |
|---|---|
| Dados | IBM Telco Customer Churn (7.043 linhas, 20 features) |
| Modelo ML | XGBoost dentro de um Pipeline scikit-learn |
| Tuning | Optuna (TPE sampler, 20 trials, CV estratificado 5-fold) |
| Desbalanceamento | `scale_pos_weight` (ponderação nativa de classes do XGBoost) |
| Explicabilidade | SHAP TreeExplainer — gráfico de barras com top-15 features |
| Rastreamento de experimentos | MLflow servidor local |
| Orquestração LLM | LangGraph (DAG com 4 nós: carregar → prever → filtrar → relatório) |
| RAG | LlamaIndex + FAISS + embeddings `BAAI/bge-small-en-v1.5` |
| LLM | API Groq — `llama-3.3-70b-versatile` (gratuito) |
| Agente | Agente LangChain com chamada de ferramentas e 3 ferramentas de domínio |
| Interface | Gradio Blocks (2 abas: Prever + Analisar) |
| Deploy | Hugging Face Spaces (gratuito) / Docker |

---

## Como Rodar

### 1. Clone e instale as dependências

```bash
git clone <url-do-repositorio>
cd churn-prediction
pip install -r requirements.txt
```

### 2. (Opcional) Configure sua chave da API Groq

A aba **Prever** e o treinamento funcionam **sem** chave de API.
A aba **Analisar** e a geração de relatórios com LLM precisam do Groq.

```bash
cp .env.example .env
# Edite o .env e adicione sua chave:
# GROQ_API_KEY=gsk_...
```

Obtenha uma chave gratuita em <https://console.groq.com>.

### 3. Baixe os dados e treine o modelo

```bash
# Baixa o dataset IBM Telco (~460 KB)
python data/download_data.py

# Treina o modelo (aprox. 3–5 minutos para 20 trials do Optuna)
python -m src.pipeline.train --trials 20
```

Ou usando o Make:

```bash
make data
make train
```

### 4. Inicie a demo

```bash
python app.py
# Acesse http://localhost:7860
```

Ou:

```bash
make app
```

---

## Estrutura do Projeto

```
churn-prediction/
├── app.py                         # Demo Gradio (2 abas)
├── requirements.txt
├── Makefile
├── Dockerfile
├── .env.example
├── data/
│   ├── download_data.py           # Faz download do CSV IBM Telco
│   └── raw/
│       └── telco_churn.csv        # Dataset baixado (ignorado pelo git)
├── models/
│   └── best_model.joblib          # Pipeline treinado (ignorado pelo git)
├── notebooks/
│   ├── 01_eda.ipynb               # Análise Exploratória de Dados
│   └── 02_model_training.ipynb    # Passo a passo do treinamento
├── src/
│   ├── config.py                  # Todas as constantes, caminhos e hiperparâmetros
│   ├── pipeline/
│   │   ├── features.py            # Limpeza + engenharia de features
│   │   ├── train.py               # Treinamento, Optuna, MLflow, avaliação
│   │   └── __main__.py            # Entry point `python -m src.pipeline.train`
│   ├── graph/
│   │   └── churn_graph.py         # Pipeline LangGraph com 4 nós
│   ├── rag/
│   │   └── insight_rag.py         # RAG com LlamaIndex e FAISS
│   └── agent/
│       └── analyst_agent.py       # Agente LangChain com ferramentas
└── mlruns/                        # Logs do MLflow (ignorados pelo git)
```

---

## Desempenho do Modelo

Resultados típicos no conjunto de teste (20% reservado) após 20 trials do Optuna:

| Métrica | Valor |
|---|---|
| ROC-AUC | ~0,847 |
| F1-score | ~0,623 |
| Precisão | ~0,662 |
| Recall | ~0,588 |
| Acurácia | ~0,805 |

Os resultados variam levemente entre execuções devido à busca estocástica do Optuna.

---

## Executando o Pipeline LangGraph Isolado

```python
import joblib
from src.graph.churn_graph import run_pipeline

model = joblib.load("models/best_model.joblib")
state = run_pipeline(model_pipeline=model)

print(state["report"])                         # Relatório de retenção de negócios
print(len(state["high_risk_customers"]))       # Número de clientes de alto risco
```

---

## Docker

```bash
# Build (treina o modelo dentro da imagem — leva ~5 min no primeiro build)
docker build -t churn-prediction .

# Executar
docker run -p 7860:7860 -e GROQ_API_KEY=gsk_... churn-prediction
# Acesse http://localhost:7860
```

---

## Deploy no Hugging Face Spaces

1. Crie um novo Space (SDK Gradio, Python 3.10).
2. Faça o push deste repositório para o Space.
3. Adicione `GROQ_API_KEY` como **Secret** nas configurações do Space.
4. O Space executará `python app.py` automaticamente.

Se quiser que o modelo já venha pré-treinado na imagem Docker (em vez de treinar a cada cold start), copie `models/best_model.joblib` para o repositório antes do push e remova a linha `*.joblib` do `.gitignore`.

---

## Decisões de Design

**Por que XGBoost e não LightGBM / CatBoost?**
O parâmetro `scale_pos_weight` do XGBoost lida nativamente com o desbalanceamento de classes (73:27) sem precisar de reamostragem. Além disso, possui suporte nativo ao SHAP via `shap.TreeExplainer`.

**Por que Optuna e não GridSearch?**
O TPE sampler do Optuna converge para bons hiperparâmetros em 20 trials, contra centenas de combinações que uma busca em grade exigiria. Otimização Bayesiana é estritamente superior para espaços de hiperparâmetros contínuos.

**Por que LlamaIndex para RAG e LangChain para o agente?**
O LlamaIndex oferece uma abstração limpa de indexação de documentos, que se encaixa naturalmente no caso de uso "perfil de cliente → documento". O `create_tool_calling_agent` do LangChain é mais maduro para fluxos estruturados com uso de ferramentas.

**Por que Groq?**
O tier gratuito da API Groq oferece inferência no `llama-3.3-70b-versatile` com latência muito baixa, sendo a escolha prática para uma demo de portfólio que deve permanecer gratuita.

---

## Licença

MIT
