import os
import json
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.tools.eda import run_eda
from app.tools.stats import run_stats
from app.tools.viz import run_viz
from app.prompts.system import SYSTEM_PROMPT

load_dotenv()


def load_llm():
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile",
        temperature=0
    )


def build_dataset_context(df: pd.DataFrame) -> str:
    """Gives the LLM a compact summary of the dataset to reason about."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    
    return f"""
DATASET OVERVIEW:
- Shape: {df.shape[0]} rows x {df.shape[1]} columns
- Numeric columns: {numeric_cols}
- Categorical columns: {categorical_cols}
- Column names: {df.columns.tolist()}
- First 3 rows:
{df.head(3).to_string()}
"""


def parse_tool_call(response_text: str) -> dict | None:
    """
    Extracts a JSON tool call from the LLM response if present.
    Expected format: <tool_call>{"tool": "...", "params": {...}}</tool_call>
    """
    import re
    pattern = r"<tool_call>(.*?)</tool_call>"
    match = re.search(pattern, response_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            return None
    return None


def execute_tool(tool_name: str, params: dict, df: pd.DataFrame) -> dict:
    """Routes tool calls to the correct function."""
    if tool_name == "run_eda":
        return run_eda(df)
    elif tool_name == "run_stats":
        return run_stats(df, **params)
    elif tool_name == "run_viz":
        return run_viz(df, **params)
    else:
        return {"error": f"Unknown tool: {tool_name}"}


def run_agent(
    user_query: str,
    df: pd.DataFrame,
    chat_history: list
) -> dict:
    """
    Main agent function.
    Returns a dict with keys: text, chart_json (optional), tool_used (optional)
    """
    llm = load_llm()
    dataset_context = build_dataset_context(df)

    # Build the planning prompt
    planning_prompt = f"""
{dataset_context}

You have access to these tools. Use them by outputting a <tool_call> block:

1. run_eda - Full exploratory data analysis. No params needed.
   Example: <tool_call>{{"tool": "run_eda", "params": {{}}}}</tool_call>

2. run_stats - Statistical analysis.
   Params: analysis_type (correlation/ttest/anova/normality), numeric_col, group_col
   Example: <tool_call>{{"tool": "run_stats", "params": {{"analysis_type": "correlation"}}}}</tool_call>

3. run_viz - Create a visualization.
   Params: chart_type (histogram/bar/scatter/correlation_heatmap/boxplot), plus relevant col names
   Example: <tool_call>{{"tool": "run_viz", "params": {{"chart_type": "histogram", "numeric_col": "sales"}}}}</tool_call>

If no tool is needed, just answer directly.

User question: {user_query}

Think about what the user wants, pick the right tool if needed, and output the tool call.
"""

    # Step 1: Planning call
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for msg in chat_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=planning_prompt))

    plan_response = llm.invoke(messages)
    plan_text = plan_response.content

    # Step 2: Check for tool call
    tool_call = parse_tool_call(plan_text)
    chart_json = None
    tool_result = None
    tool_used = None

    if tool_call:
        tool_used = tool_call.get("tool")
        params = tool_call.get("params", {})
        tool_result = execute_tool(tool_used, params, df)

        # If viz tool, extract chart separately
        if tool_used == "run_viz" and tool_result.get("success"):
            chart_json = tool_result.get("chart_json")
            tool_result_for_llm = {"chart_created": True, "chart_type": params.get("chart_type")}
        else:
            tool_result_for_llm = tool_result

        # Step 3: Synthesis call
        synthesis_prompt = f"""
The user asked: {user_query}

You called the tool '{tool_used}' and got this result:
{json.dumps(tool_result_for_llm, indent=2)}

Now explain the results clearly in plain English. Follow the response format in your instructions.
"""
        messages.append(AIMessage(content=plan_text))
        messages.append(HumanMessage(content=synthesis_prompt))
        final_response = llm.invoke(messages)
        final_text = final_response.content

    else:
        # No tool needed, use plan response directly
        final_text = plan_text

    return {
        "text": final_text,
        "chart_json": chart_json,
        "tool_used": tool_used
    }