import os
import json
import operator
import pandas as pd
from typing import TypedDict, Annotated, Sequence, Literal
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
)
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from app.tools.eda import run_eda
from app.tools.stats import run_stats, run_aggregation, run_topn, check_assumptions
from app.tools.viz import run_viz
from app.prompts.system import SYSTEM_PROMPT

load_dotenv()


# ── State ──────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


# ── LLM ───────────────────────────────────────────────────────────────────────

def load_llm(provider: str = "groq"):
    try:
        import streamlit as st
        groq_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
        google_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    except Exception:
        groq_key = os.getenv("GROQ_API_KEY")
        google_key = os.getenv("GOOGLE_API_KEY")

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=google_key,
            temperature=0
        )
    else:
        return ChatGroq(
            api_key=groq_key,
            model="llama-3.3-70b-versatile",
            temperature=0
        )


def build_agent(df: pd.DataFrame, provider: str = "groq"):
    tools = create_tools(df)
    llm = load_llm(provider).bind_tools(tools)

    def agent_node(state: AgentState):
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    def should_continue(state: AgentState) -> Literal["tools", "end"]:
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "end"

    tool_node = ToolNode(tools)

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END}
    )
    graph.add_edge("tools", "agent")

    return graph.compile()


# ── Tool Factory ───────────────────────────────────────────────────────────────

def create_tools(df: pd.DataFrame):

    @tool
    def run_eda_tool() -> dict:
        """Run exploratory data analysis on the dataset. Use this for overviews,
        data quality checks, missing value detection, and column summaries."""
        return run_eda(df)

    @tool
    def run_stats_tool(
        analysis_type: str,
        numeric_col: str = "",
        group_col: str = ""
    ) -> dict:
        """Run statistical analysis on the dataset.
        analysis_type: correlation | ttest | anova | normality | mannwhitney | kruskal
        numeric_col: the numeric column to analyze
        group_col: categorical column to group by (required for ttest, anova, mannwhitney, kruskal)
        IMPORTANT: Always run check_assumptions_tool first before choosing between ttest/mannwhitney or anova/kruskal.
        """
        params = {"analysis_type": analysis_type}
        if numeric_col:
            params["numeric_col"] = numeric_col
        if group_col:
            params["group_col"] = group_col
        return run_stats(df, **params)

    @tool
    def run_aggregation_tool(
        group_col: str,
        numeric_col: str,
        agg: str = "mean"
    ) -> dict:
        """Aggregate a numeric column grouped by a categorical column.
        group_col: categorical column to group by
        numeric_col: numeric column to aggregate
        agg: mean | sum | max | min
        Use this for questions like 'which region has highest sales' or
        'what is the average profit by category'.
        """
        return run_aggregation(df, group_col, numeric_col, agg)

    @tool
    def run_viz_tool(
        chart_type: str,
        numeric_col: str = "",
        group_col: str = "",
        x_col: str = "",
        y_col: str = "",
        color_col: str = ""
    ) -> dict:
        """Create a visualization.
        chart_type: histogram | bar | scatter | correlation_heatmap | boxplot | line | pie | pivot_heatmap
        numeric_col: for histogram and boxplot
        group_col: for bar chart and boxplot grouping
        x_col, y_col: for scatter/line plot
        color_col: optional color grouping for scatter
        """
        params = {"chart_type": chart_type}
        if numeric_col:
            params["numeric_col"] = numeric_col
        if group_col:
            params["group_col"] = group_col
        if x_col:
            params["x_col"] = x_col
        if y_col:
            params["y_col"] = y_col
        if color_col:
            params["color_col"] = color_col
        return run_viz(df, **params)

    @tool
    def run_topn_tool(
        group_col: str,
        numeric_col: str,
        n: int = 5,
        agg: str = "sum"
    ) -> dict:
        """Get the top N or bottom N groups by aggregated value.
        group_col: categorical column to group by
        numeric_col: numeric column to aggregate
        n: number of top results to return (default 5)
        agg: sum | mean | max | min
        Use this for questions like 'top 5 stores by sales' or
        'which 3 categories have lowest profit'.
        """
        return run_topn(df, group_col, numeric_col, n, agg)

    @tool
    def check_assumptions_tool(
        numeric_col: str,
        group_col: str = ""
    ) -> dict:
        """Check statistical assumptions before running a test.
        Always call this before running ttest, anova, mannwhitney, or kruskal
        when the user asks about differences between groups.
        numeric_col: the numeric column to test
        group_col: the categorical column defining groups
        Returns the recommended test to use based on normality and variance checks.
        """
        return check_assumptions(df, numeric_col, group_col if group_col else None)

    return [run_eda_tool, run_stats_tool, run_aggregation_tool,
            run_viz_tool, run_topn_tool, check_assumptions_tool]


# ── Dataset Context ────────────────────────────────────────────────────────────

def build_dataset_context(df: pd.DataFrame) -> str:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["str", "category"]).columns.tolist()
    return f"""
DATASET OVERVIEW:
- Shape: {df.shape[0]} rows x {df.shape[1]} columns
- Numeric columns: {numeric_cols}
- Categorical columns: {categorical_cols}
- Column names: {df.columns.tolist()}
- First 3 rows:
{df.head(3).to_string()}
"""


# ── Tool Name Normalizer ───────────────────────────────────────────────────────

TOOL_NAME_MAP = {
    "run_eda_tool": "run_eda",
    "run_stats_tool": "run_stats",
    "run_aggregation_tool": "run_aggregation",
    "run_viz_tool": "run_viz",
    "run_topn_tool": "run_topn",
    "check_assumptions_tool": "check_assumptions"
}


# ── Main Entry Point ───────────────────────────────────────────────────────────

def run_agent(
    user_query: str,
    df: pd.DataFrame,
    chat_history: list,
    provider: str = "groq"
) -> dict:
    dataset_context = build_dataset_context(df)
    system_content = SYSTEM_PROMPT + "\n\n" + dataset_context

    messages = [SystemMessage(content=system_content)]

    for msg in chat_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=user_query))

    agent = build_agent(df, provider)
    result = agent.invoke({"messages": messages})

    final_message = result["messages"][-1]
    final_text = final_message.content if hasattr(final_message, "content") else str(final_message)

    chart_json = None
    tools_used = []

    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                raw_name = tc["name"]
                tools_used.append(TOOL_NAME_MAP.get(raw_name, raw_name))

        if isinstance(msg, ToolMessage):
            try:
                tool_result = json.loads(msg.content)
                if isinstance(tool_result, dict):
                    if tool_result.get("success") and "chart_json" in tool_result:
                        chart_json = tool_result["chart_json"]
            except (json.JSONDecodeError, TypeError):
                pass

    tool_used = tools_used[-1] if tools_used else None

    return {
        "text": final_text,
        "chart_json": chart_json,
        "tool_used": tool_used,
        "tools_used": tools_used
    }