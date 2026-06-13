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
from app.tools.stats import run_stats, run_aggregation
from app.tools.viz import run_viz
from app.prompts.system import SYSTEM_PROMPT

load_dotenv()


# ── State ──────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


# ── LLM ───────────────────────────────────────────────────────────────────────

def load_llm():
    try:
        import streamlit as st
        api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    except Exception:
        api_key = os.getenv("GROQ_API_KEY")
    
    return ChatGroq(
        api_key=api_key,
        model="llama-3.3-70b-versatile",
        temperature=0
    )


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
        analysis_type: correlation | ttest | anova | normality
        numeric_col: the numeric column to analyze
        group_col: categorical column to group by (required for ttest and anova)
        Use ttest for 2 groups, anova for 3+ groups.
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
        chart_type: histogram | bar | scatter | correlation_heatmap | boxplot
        numeric_col: for histogram and boxplot
        group_col: for bar chart and boxplot grouping
        x_col, y_col: for scatter plot
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

    return [run_eda_tool, run_stats_tool, run_aggregation_tool, run_viz_tool]


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


# ── Graph Builder ──────────────────────────────────────────────────────────────

def build_agent(df: pd.DataFrame):
    tools = create_tools(df)
    llm = load_llm().bind_tools(tools)

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


# ── Tool Name Normalizer ───────────────────────────────────────────────────────

TOOL_NAME_MAP = {
    "run_eda_tool": "run_eda",
    "run_stats_tool": "run_stats",
    "run_aggregation_tool": "run_aggregation",
    "run_viz_tool": "run_viz"
}


# ── Main Entry Point ───────────────────────────────────────────────────────────

def run_agent(
    user_query: str,
    df: pd.DataFrame,
    chat_history: list
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

    agent = build_agent(df)
    result = agent.invoke({"messages": messages})

    # Extract final text response
    final_message = result["messages"][-1]
    final_text = final_message.content if hasattr(final_message, "content") else str(final_message)

    # Extract chart and tools used from message history
    chart_json = None
    tools_used = []

    for msg in result["messages"]:
        # Collect tool names from AI tool calls
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                raw_name = tc["name"]
                tools_used.append(TOOL_NAME_MAP.get(raw_name, raw_name))

        # Extract chart from tool results
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