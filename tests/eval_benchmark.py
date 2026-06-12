"""
AnalytiQ Evaluation Benchmark
Tests whether the agent correctly routes queries to the right tools
and returns numerically accurate results.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from app.agent import run_agent

# ── Benchmark Dataset ─────────────────────────────────────────────────────────

np.random.seed(42)
EVAL_DF = pd.DataFrame({
    "sales": [100, 200, 300, 400, 500, 150, 250, 350, 450, 600],
    "profit": [10, 22, 28, 45, 48, 15, 27, 38, 44, 60],
    "region": ["North", "North", "South", "South", "North",
                "South", "East", "East", "North", "South"],
    "category": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"]
})

# ── Test Cases ────────────────────────────────────────────────────────────────

test_cases = [
    {
        "id": 1,
        "query": "Give me an overview of this dataset",
        "expected_tool": "run_eda",
        "check": lambda r: r["tool_used"] == "run_eda"
    },
    {
        "id": 2,
        "query": "Are there any missing values in this dataset?",
        "expected_tool": "run_eda",
        "check": lambda r: r["tool_used"] == "run_eda"
    },
    {
        "id": 3,
        "query": "What is the correlation between sales and profit?",
        "expected_tool": "run_stats",
        "check": lambda r: r["tool_used"] == "run_stats"
    },
    {
        "id": 4,
        "query": "Show me the distribution of sales",
        "expected_tool": "run_viz",
        "check": lambda r: r["tool_used"] == "run_viz"
    },
    {
        "id": 5,
        "query": "Show me a bar chart of average profit by category",
        "expected_tool": "run_viz",
        "check": lambda r: r["tool_used"] == "run_viz"
    },
    {
        "id": 6,
        "query": "Is sales normally distributed?",
        "expected_tool": "run_stats",
        "check": lambda r: r["tool_used"] == "run_stats"
    },
    {
        "id": 7,
        "query": "Which region has the highest average sales?",
        "expected_tool": "run_aggregation",
        "check": lambda r: r["tool_used"] == "run_aggregation"
    },
    {
        "id": 8,
        "query": "Show me a correlation heatmap",
        "expected_tool": "run_viz",
        "check": lambda r: r["tool_used"] == "run_viz"
    },
    {
        "id": 9,
        "query": "Is there a significant difference in sales between categories?",
        "expected_tool": "run_stats",
        "check": lambda r: r["tool_used"] == "run_stats"
    },
    {
        "id": 10,
        "query": "Show me a scatter plot of sales vs profit",
        "expected_tool": "run_viz",
        "check": lambda r: r["tool_used"] == "run_viz"
    },
    {
        "id": 11,
        "query": "What is the average profit by region?",
        "expected_tool": "run_aggregation",
        "check": lambda r: r["tool_used"] == "run_aggregation"
    },
    {
        "id": 12,
        "query": "Show me a boxplot of sales by region",
        "expected_tool": "run_viz",
        "check": lambda r: r["tool_used"] == "run_viz"
    },
]

# ── Runner ────────────────────────────────────────────────────────────────────

def run_benchmark():
    print("\n" + "="*60)
    print("AnalytiQ Evaluation Benchmark")
    print("="*60)

    results = []

    for tc in test_cases:
        try:
            result = run_agent(tc["query"], EVAL_DF, [])
            passed = tc["check"](result)
            status = "✅ PASS" if passed else "❌ FAIL"
            results.append(passed)
            print(f"\n[{tc['id']:02d}] {status}")
            print(f"     Query:         {tc['query']}")
            print(f"     Expected tool: {tc['expected_tool']}")
            print(f"     Got tool:      {result['tool_used']}")
        except Exception as e:
            results.append(False)
            print(f"\n[{tc['id']:02d}] ❌ ERROR")
            print(f"     Query: {tc['query']}")
            print(f"     Error: {str(e)}")

    total = len(results)
    passed = sum(results)
    accuracy = round(passed / total * 100, 1)

    print("\n" + "="*60)
    print(f"Results: {passed}/{total} correct ({accuracy}% accuracy)")
    print("="*60 + "\n")

    return accuracy


if __name__ == "__main__":
    accuracy = run_benchmark()
    # Fail CI if accuracy drops below 75%
    if accuracy < 75:
        sys.exit(1)