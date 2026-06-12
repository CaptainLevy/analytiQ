import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
import json


def run_viz(df: pd.DataFrame, chart_type: str, **kwargs) -> dict:
    """
    Creates visualizations and returns them as JSON for Streamlit rendering.

    Supported chart_type values:
    - histogram: distribution of a numeric column
    - bar: count or mean of a numeric col grouped by categorical col
    - scatter: relationship between two numeric columns
    - correlation_heatmap: heatmap of correlation matrix
    - boxplot: distribution across groups
    """

    if chart_type == "histogram":
        return _histogram(df, **kwargs)
    elif chart_type == "bar":
        return _bar(df, **kwargs)
    elif chart_type == "scatter":
        return _scatter(df, **kwargs)
    elif chart_type == "correlation_heatmap":
        return _correlation_heatmap(df, **kwargs)
    elif chart_type == "boxplot":
        return _boxplot(df, **kwargs)
    else:
        return {"error": f"Unknown chart type: {chart_type}"}


def _to_json(fig) -> dict:
    return {
        "chart_json": json.dumps(fig, cls=PlotlyJSONEncoder),
        "success": True
    }


def _histogram(df: pd.DataFrame, numeric_col: str, **kwargs) -> dict:
    if numeric_col not in df.columns:
        return {"error": f"Column '{numeric_col}' not found"}

    fig = px.histogram(
        df, x=numeric_col,
        title=f"Distribution of {numeric_col}",
        color_discrete_sequence=["#636EFA"]
    )
    fig.update_layout(bargap=0.1)
    return _to_json(fig)


def _bar(df: pd.DataFrame, group_col: str, numeric_col: str, 
         agg: str = "mean", **kwargs) -> dict:
    if group_col not in df.columns or numeric_col not in df.columns:
        return {"error": "Column not found"}

    if agg == "mean":
        grouped = df.groupby(group_col)[numeric_col].mean().reset_index()
        title = f"Mean {numeric_col} by {group_col}"
    else:
        grouped = df.groupby(group_col)[numeric_col].sum().reset_index()
        title = f"Total {numeric_col} by {group_col}"

    fig = px.bar(
        grouped, x=group_col, y=numeric_col,
        title=title,
        color=numeric_col,
        color_continuous_scale="Blues"
    )
    return _to_json(fig)


def _scatter(df: pd.DataFrame, x_col: str, y_col: str, 
             color_col: str = None, **kwargs) -> dict:
    if x_col not in df.columns or y_col not in df.columns:
        return {"error": "Column not found"}

    fig = px.scatter(
        df, x=x_col, y=y_col,
        color=color_col if color_col and color_col in df.columns else None,
        title=f"{x_col} vs {y_col}",
        trendline="ols"
    )
    return _to_json(fig)


def _correlation_heatmap(df: pd.DataFrame, **kwargs) -> dict:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return {"error": "Need at least 2 numeric columns"}

    corr = numeric_df.corr().round(4)
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale="RdBu",
        zmid=0,
        text=corr.values.round(2),
        texttemplate="%{text}",
    ))
    fig.update_layout(title="Correlation Heatmap")
    return _to_json(fig)


def _boxplot(df: pd.DataFrame, numeric_col: str, 
             group_col: str = None, **kwargs) -> dict:
    if numeric_col not in df.columns:
        return {"error": f"Column '{numeric_col}' not found"}

    fig = px.box(
        df, y=numeric_col,
        x=group_col if group_col and group_col in df.columns else None,
        title=f"Boxplot of {numeric_col}" + (f" by {group_col}" if group_col else ""),
        color=group_col if group_col and group_col in df.columns else None
    )
    return _to_json(fig)