"""
LangChain analyst agent for churn segment analysis and retention strategy.

Exposes three tools:
  - analyze_churn_segment   : compute statistics for a named customer segment
  - suggest_retention_actions: produce action recommendations for a risk profile
  - calculate_revenue_at_risk: estimate monthly recurring revenue at risk

The agent is powered by Groq's llama-3.3-70b-versatile model.
If GROQ_API_KEY is not set, the agent is disabled and a descriptive
error string is returned instead of raising an exception.
"""

import json
import logging
import os
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _analyze_segment(predictions_df: pd.DataFrame, segment_json: str) -> str:
    """
    Internal implementation of the analyze_churn_segment tool.

    Accepts a JSON string such as:
      ``{"column": "Contract", "value": "Month-to-month"}``
    Returns a statistical summary of that segment.
    """
    try:
        params = json.loads(segment_json)
        column = params.get("column", "Contract")
        value = params.get("value")
    except (json.JSONDecodeError, KeyError):
        column = "Contract"
        value = segment_json  # treat raw string as the filter value

    if column not in predictions_df.columns:
        return f"Column '{column}' not found. Available: {list(predictions_df.columns)}"

    if value:
        subset = predictions_df[predictions_df[column].astype(str) == str(value)]
    else:
        subset = predictions_df

    if subset.empty:
        return f"No customers found for {column}={value}"

    total = len(subset)
    avg_prob = subset["churn_probability"].mean() if "churn_probability" in subset.columns else 0
    high_risk_count = (subset["churn_probability"] > 0.7).sum() if "churn_probability" in subset.columns else 0
    avg_charge = subset["MonthlyCharges"].mean() if "MonthlyCharges" in subset.columns else 0

    contract_dist = ""
    if "Contract" in subset.columns:
        contract_counts = subset["Contract"].value_counts()
        contract_dist = ", ".join(f"{k}: {v}" for k, v in contract_counts.items())

    tenure_avg = subset["tenure"].mean() if "tenure" in subset.columns else 0

    return (
        f"Segment analysis — {column}={value}\n"
        f"  Customers: {total}\n"
        f"  Avg churn probability: {avg_prob:.1%}\n"
        f"  High-risk (>70%): {high_risk_count} ({high_risk_count/max(total,1)*100:.1f}%)\n"
        f"  Avg monthly charge: ${avg_charge:.2f}\n"
        f"  Avg tenure: {tenure_avg:.1f} months\n"
        f"  Contract breakdown: {contract_dist or 'N/A'}"
    )


def _suggest_retention_actions(profile_json: str) -> str:
    """
    Return rule-based retention suggestions for a customer risk profile.

    Accepts JSON with keys like ``contract``, ``tenure_months``,
    ``monthly_charges``, ``internet_service``, ``risk_level``.
    """
    try:
        profile = json.loads(profile_json)
    except json.JSONDecodeError:
        profile = {"description": profile_json}

    contract = str(profile.get("contract", "")).lower()
    tenure = int(profile.get("tenure_months", 0))
    charges = float(profile.get("monthly_charges", 0))
    internet = str(profile.get("internet_service", "")).lower()
    risk = str(profile.get("risk_level", "high")).lower()

    actions = []

    if "month" in contract:
        actions.append(
            "Contract upgrade offer: Provide a 20% discount to switch from "
            "month-to-month to a 1-year contract. Cost: ~$15/month discount for "
            "12 months vs. $0 revenue if they churn."
        )

    if tenure < 12:
        actions.append(
            "Early loyalty programme: Trigger a welcome retention campaign at "
            "month 6 and month 10. Include a free service upgrade for 3 months."
        )
    elif tenure < 24:
        actions.append(
            "Mid-tenure retention: Customer is past the early churn window but "
            "still at risk. Offer a bundled add-on (TechSupport + OnlineSecurity) "
            "at 30% off for 6 months."
        )

    if charges > 70:
        actions.append(
            "High-value customer: Assign a dedicated account manager. "
            "Consider a loyalty reward or a bill credit of $20 to reduce "
            "perceived price sensitivity."
        )

    if "fiber" in internet:
        actions.append(
            "Fiber optic customer: Often price-sensitive due to competitive "
            "alternatives. Emphasise reliability metrics and offer a speed "
            "upgrade at no additional cost for 3 months."
        )

    if not actions:
        actions.append(
            "General retention: Send personalised re-engagement email highlighting "
            "new features and a 10% loyalty discount valid for 30 days."
        )

    return "\n".join(f"• {a}" for a in actions)


def _calculate_revenue_at_risk(predictions_df: pd.DataFrame, threshold: float = 0.7) -> str:
    """
    Estimate the monthly recurring revenue (MRR) at risk of churning.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Must contain ``churn_probability`` and ``MonthlyCharges``.
    threshold : float
        Probability threshold above which a customer is considered at risk.
    """
    if "churn_probability" not in predictions_df.columns:
        return "churn_probability column not found in predictions dataframe"

    at_risk = predictions_df[predictions_df["churn_probability"] >= threshold]

    if at_risk.empty:
        return f"No customers above {threshold:.0%} churn probability threshold"

    monthly_at_risk = at_risk["MonthlyCharges"].sum() if "MonthlyCharges" in at_risk.columns else 0
    annual_at_risk = monthly_at_risk * 12
    count = len(at_risk)
    avg_charge = at_risk["MonthlyCharges"].mean() if "MonthlyCharges" in at_risk.columns else 0

    # Simple CLV estimate: avg_charge * avg_remaining_tenure (12 mo assumed)
    avg_clv_estimate = avg_charge * 12

    return (
        f"Revenue at Risk Analysis (threshold: {threshold:.0%})\n"
        f"  Customers at risk: {count}\n"
        f"  Monthly revenue at risk: ${monthly_at_risk:,.0f}\n"
        f"  Annualised revenue at risk: ${annual_at_risk:,.0f}\n"
        f"  Avg monthly charge per at-risk customer: ${avg_charge:.2f}\n"
        f"  Estimated avg CLV at risk per customer: ${avg_clv_estimate:,.0f}\n"
        f"  Cost of 5% churn reduction: saves ~${monthly_at_risk * 0.05:,.0f}/month"
    )


# ---------------------------------------------------------------------------
# Agent builder
# ---------------------------------------------------------------------------

def _build_agent(predictions_df: pd.DataFrame):
    """
    Build and return a LangChain AgentExecutor with churn-analysis tools.

    Returns *None* if GROQ_API_KEY is not set.
    """
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        logger.warning("GROQ_API_KEY not set — analyst agent disabled")
        return None

    try:
        from langchain.agents import AgentExecutor, create_tool_calling_agent
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.tools import tool
        from langchain_groq import ChatGroq
    except ImportError as exc:
        logger.error("LangChain/Groq packages not installed: %s", exc)
        return None

    @tool
    def analyze_churn_segment(segment_json: str) -> str:
        """
        Analyse a specific customer segment.
        Input must be a JSON string with keys 'column' and 'value'.
        Example: {"column": "Contract", "value": "Month-to-month"}
        Returns segment statistics including churn probability and revenue.
        """
        return _analyze_segment(predictions_df, segment_json)

    @tool
    def suggest_retention_actions(profile_json: str) -> str:
        """
        Suggest retention actions for a customer risk profile.
        Input must be a JSON string with optional keys:
        contract, tenure_months, monthly_charges, internet_service, risk_level.
        Returns actionable retention recommendations.
        """
        return _suggest_retention_actions(profile_json)

    @tool
    def calculate_revenue_at_risk(threshold: str = "0.7") -> str:
        """
        Calculate the monthly recurring revenue at risk of churning.
        Input is a probability threshold string (e.g., '0.7' for 70%).
        Returns detailed revenue impact analysis.
        """
        try:
            t = float(threshold)
        except ValueError:
            t = 0.7
        return _calculate_revenue_at_risk(predictions_df, t)

    tools = [analyze_churn_segment, suggest_retention_actions, calculate_revenue_at_risk]

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        groq_api_key=groq_key,
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert customer retention analyst for a telecom company. "
            "You have access to churn prediction data and specialised tools. "
            "Always use the tools to gather data before drawing conclusions. "
            "Be specific, reference numbers, and provide actionable recommendations.",
        ),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=6,
        return_intermediate_steps=False,
    )
    return executor


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_analyst(
    predictions_df: pd.DataFrame,
    query: str,
) -> str:
    """
    Run the LangChain analyst agent on a churn predictions dataframe.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Output of the prediction pipeline containing ``churn_probability``,
        ``MonthlyCharges``, ``Contract``, ``tenure``, etc.
    query : str
        Natural-language question or instruction for the analyst agent.

    Returns
    -------
    str
        Agent's response, or a descriptive error message if the agent
        cannot be initialised (e.g., missing API key).
    """
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        return (
            "Analyst agent is disabled — GROQ_API_KEY is not set.\n\n"
            "To enable: set GROQ_API_KEY in your .env file and restart the app.\n\n"
            + _calculate_revenue_at_risk(predictions_df)
        )

    agent = _build_agent(predictions_df)
    if agent is None:
        return "Failed to initialise the analyst agent. Check logs for details."

    try:
        result = agent.invoke({"input": query})
        return result.get("output", str(result))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Analyst agent invocation failed")
        return (
            f"Agent encountered an error: {exc}\n\n"
            "Falling back to revenue summary:\n\n"
            + _calculate_revenue_at_risk(predictions_df)
        )
