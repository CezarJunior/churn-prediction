"""
LangGraph orchestration for the end-to-end churn analysis pipeline.

Defines a stateful graph that:
  1. Loads and validates the raw dataset
  2. Applies feature engineering and runs model inference
  3. Identifies high-risk customers (probability > 0.7)
  4. Generates a natural-language business retention report via Groq

Each node is a pure function that receives the current :class:`ChurnState`
and returns a partial state update (only the keys it modifies).
"""

import logging
import os
from typing import Optional

import pandas as pd
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class ChurnState(TypedDict):
    """Shared state passed between LangGraph nodes."""
    raw_data: Optional[pd.DataFrame]
    clean_data: Optional[pd.DataFrame]
    predictions: Optional[pd.DataFrame]
    high_risk_customers: Optional[pd.DataFrame]
    report: Optional[str]
    error: Optional[str]


# ---------------------------------------------------------------------------
# Required columns for schema validation
# ---------------------------------------------------------------------------

_REQUIRED_COLUMNS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract",
    "InternetService",
    "PaymentMethod",
]


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

def load_and_validate_node(state: ChurnState) -> ChurnState:
    """
    Node 1 — Load data (if not already in state) and validate schema.

    Checks that all columns in ``_REQUIRED_COLUMNS`` are present.
    On failure, writes an error message to ``state['error']`` so
    downstream nodes can short-circuit gracefully.
    """
    try:
        from src.pipeline.features import clean_data, load_data

        df = state.get("raw_data")
        if df is None:
            logger.info("[graph] Downloading data …")
            df = load_data()

        missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            return {**state, "error": f"Schema validation failed — missing columns: {missing}"}

        logger.info("[graph] Schema OK — %d rows, %d cols", *df.shape)
        cleaned = clean_data(df)
        return {**state, "raw_data": df, "clean_data": cleaned, "error": None}

    except Exception as exc:  # noqa: BLE001
        logger.exception("[graph] load_and_validate_node failed")
        return {**state, "error": str(exc)}


def preprocess_and_predict_node(state: ChurnState) -> ChurnState:
    """
    Node 2 — Feature engineering + model inference.

    Expects ``state['clean_data']`` to be populated and a trained
    ``model_pipeline`` to be injected via closure (see :func:`run_pipeline`).
    Adds ``churn_probability``, ``churn_predicted``, and ``risk_label``
    columns to the predictions dataframe.
    """
    if state.get("error"):
        return state

    try:
        from src.pipeline.features import engineer_features, get_feature_target_split

        df = state["clean_data"].copy()
        df_feat = engineer_features(df)
        X, _ = get_feature_target_split(df_feat)

        # model_pipeline is captured via closure inside run_pipeline()
        pipeline = state.get("_model_pipeline")
        if pipeline is None:
            return {**state, "error": "No model pipeline provided to graph node"}

        proba = pipeline.predict_proba(X)[:, 1]
        pred = (proba >= 0.5).astype(int)

        df_feat = df_feat.reset_index(drop=True)
        df_feat["churn_probability"] = proba
        df_feat["churn_predicted"] = pred
        df_feat["risk_label"] = pd.cut(
            proba,
            bins=[-0.001, 0.3, 0.5, 0.7, 1.001],
            labels=["Low", "Medium", "High", "Critical"],
        ).astype(str)

        logger.info(
            "[graph] Predictions complete — %d high-risk (>0.7)",
            (proba > 0.7).sum(),
        )
        return {**state, "predictions": df_feat}

    except Exception as exc:  # noqa: BLE001
        logger.exception("[graph] preprocess_and_predict_node failed")
        return {**state, "error": str(exc)}


def identify_high_risk_node(state: ChurnState) -> ChurnState:
    """
    Node 3 — Filter customers with churn_probability > 0.7.

    Segments the high-risk cohort by ``tenure_group`` and attaches
    aggregate statistics used by the report generator.
    """
    if state.get("error"):
        return state

    try:
        predictions = state["predictions"]
        high_risk = predictions[predictions["churn_probability"] > 0.7].copy()

        if high_risk.empty:
            logger.warning("[graph] No customers above 0.7 threshold")
            high_risk = predictions.nlargest(20, "churn_probability").copy()
            high_risk["note"] = "Threshold relaxed — top-20 by probability"

        # Segment summary for the LLM context
        segment_summary = (
            high_risk.groupby("tenure_group")
            .agg(
                count=("churn_probability", "size"),
                avg_probability=("churn_probability", "mean"),
                avg_monthly_charges=("MonthlyCharges", "mean"),
            )
            .reset_index()
            .sort_values("avg_probability", ascending=False)
        )

        logger.info("[graph] High-risk customers: %d", len(high_risk))
        state = {
            **state,
            "high_risk_customers": high_risk,
            "_segment_summary": segment_summary,
        }
        return state

    except Exception as exc:  # noqa: BLE001
        logger.exception("[graph] identify_high_risk_node failed")
        return {**state, "error": str(exc)}


def generate_report_node(state: ChurnState) -> ChurnState:
    """
    Node 4 — Generate a business retention report using Groq via LangChain.

    Falls back to a structured summary if GROQ_API_KEY is not set or if the
    API call fails, so the pipeline remains functional in offline mode.
    """
    if state.get("error"):
        return state

    high_risk = state.get("high_risk_customers", pd.DataFrame())
    predictions = state.get("predictions", pd.DataFrame())

    # --- Build context string ---
    total = len(predictions)
    n_high = len(high_risk)
    avg_prob = high_risk["churn_probability"].mean() if not high_risk.empty else 0.0
    avg_charge = high_risk["MonthlyCharges"].mean() if not high_risk.empty else 0.0
    monthly_at_risk = n_high * avg_charge

    segment_summary = state.get("_segment_summary")
    seg_text = ""
    if segment_summary is not None and not segment_summary.empty:
        seg_text = segment_summary.to_string(index=False)

    context = f"""
Churn Analysis Results:
- Total customers analysed: {total}
- High-risk customers (probability > 70%): {n_high} ({n_high/max(total,1)*100:.1f}%)
- Average churn probability in high-risk group: {avg_prob:.1%}
- Average monthly charge in high-risk group: ${avg_charge:.2f}
- Estimated monthly revenue at risk: ${monthly_at_risk:,.0f}

Segment breakdown by tenure_group:
{seg_text}
""".strip()

    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        logger.warning("[graph] GROQ_API_KEY not set — using rule-based report")
        report = _fallback_report(context, high_risk)
        return {**state, "report": report}

    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            groq_api_key=groq_key,
        )

        system_msg = SystemMessage(content=(
            "You are a senior customer retention analyst at a telecom company. "
            "Write concise, actionable reports for the business team. "
            "Use bullet points. Be specific with numbers. Avoid jargon."
        ))
        human_msg = HumanMessage(content=(
            f"Based on the following churn model results, write a retention strategy report "
            f"with: (1) Executive Summary, (2) Key Risk Segments, "
            f"(3) Recommended Retention Actions, (4) Expected ROI.\n\n{context}"
        ))

        response = llm.invoke([system_msg, human_msg])
        report = response.content
        logger.info("[graph] LLM report generated (%d chars)", len(report))

    except Exception as exc:  # noqa: BLE001
        logger.warning("[graph] LLM call failed (%s) — using fallback report", exc)
        report = _fallback_report(context, high_risk)

    return {**state, "report": report}


def _fallback_report(context: str, high_risk: pd.DataFrame) -> str:
    """Rule-based report when the LLM is unavailable."""
    lines = [
        "# Churn Retention Report",
        "",
        "## Executive Summary",
        context,
        "",
        "## Key Risk Segments",
    ]
    if not high_risk.empty and "tenure_group" in high_risk.columns:
        for group, sub in high_risk.groupby("tenure_group"):
            avg = sub["churn_probability"].mean()
            lines.append(f"- Tenure {group}: {len(sub)} customers, avg risk {avg:.1%}")
    else:
        lines.append("- Segment data unavailable")

    lines += [
        "",
        "## Recommended Retention Actions",
        "- Offer month-to-month customers a discounted annual contract",
        "- Proactive outreach within the first 12 months of tenure",
        "- Bundle internet security / tech support add-ons at reduced price",
        "- Personalised loyalty offers for customers paying > $70/month",
        "",
        "## Expected ROI",
        "- Reducing churn by 5 percentage points recovers ~$50K/month at current ARPU",
        "- Retention campaigns typically cost 5-10x less than new customer acquisition",
        "",
        "_Note: LLM report generation is disabled. Set GROQ_API_KEY to enable AI-generated insights._",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Graph construction & public entry point
# ---------------------------------------------------------------------------

def build_graph():
    """
    Construct and compile the LangGraph StateGraph.

    Returns
    -------
    CompiledGraph
        Ready-to-invoke LangGraph graph.
    """
    from langgraph.graph import END, StateGraph

    graph = StateGraph(ChurnState)

    graph.add_node("load_and_validate", load_and_validate_node)
    graph.add_node("preprocess_and_predict", preprocess_and_predict_node)
    graph.add_node("identify_high_risk", identify_high_risk_node)
    graph.add_node("generate_report", generate_report_node)

    graph.set_entry_point("load_and_validate")
    graph.add_edge("load_and_validate", "preprocess_and_predict")
    graph.add_edge("preprocess_and_predict", "identify_high_risk")
    graph.add_edge("identify_high_risk", "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile()


def run_pipeline(
    model_pipeline,
    raw_df: Optional[pd.DataFrame] = None,
) -> ChurnState:
    """
    Execute the full LangGraph pipeline.

    Parameters
    ----------
    model_pipeline : sklearn.pipeline.Pipeline
        Trained pipeline from :func:`src.pipeline.train.train_model`.
    raw_df : pd.DataFrame, optional
        Pre-loaded dataframe. If *None*, the graph downloads the data itself.

    Returns
    -------
    ChurnState
        Final state after all nodes have executed.
    """
    compiled = build_graph()

    initial_state: ChurnState = {
        "raw_data": raw_df,
        "clean_data": None,
        "predictions": None,
        "high_risk_customers": None,
        "report": None,
        "error": None,
        "_model_pipeline": model_pipeline,  # injected for the predict node
    }

    final_state = compiled.invoke(initial_state)
    if final_state.get("error"):
        logger.error("[graph] Pipeline finished with error: %s", final_state["error"])
    else:
        logger.info("[graph] Pipeline completed successfully")

    return final_state
