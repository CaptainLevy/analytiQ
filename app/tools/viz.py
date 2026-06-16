import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
import json

def _line_chart(df: pd.DataFrame, x_col: str, y_col: str,
                color_col: str = None, **kwargs) -> dict:
    if x_col not in df.columns or y_col not in df.columns:
        return {"error": "Column not found"}

    sorted_df = df.sort_values(x_col)

    fig = px.line(
        sorted_df,
        x=x_col,
        y=y_col,
        color=color_col if color_col and color_col in df.columns else None,
        title=f"{y_col} over {x_col}",
        markers=True
    )
    fig.update_layout(xaxis_title=x_col, yaxis_title=y_col)
    return _to_json(fig)


def _pie_chart(df: pd.DataFrame, category_col: str = None,
               group_col: str = None, value_col: str = None, 
               numeric_col: str = None, **kwargs) -> dict:
    
    # Accept either category_col or group_col
    col = category_col or group_col
    val = value_col or numeric_col
    
    if not col:
        return {"error": "Please provide a category column"}
    if col not in df.columns:
        return {"error": f"Column '{col}' not found"}

    if val and val in df.columns:
        grouped = df.groupby(col)[val].sum().reset_index()
        fig = px.pie(
            grouped,
            names=col,
            values=val,
            title=f"{val} by {col}"
        )
    else:
        counts = df[col].value_counts().reset_index()
        counts.columns = [col, "count"]
        fig = px.pie(
            counts,
            names=col,
            values="count",
            title=f"Distribution of {col}"
        )

    fig.update_traces(textposition="inside", textinfo="percent+label")
    return _to_json(fig)

def run_viz(df: pd.DataFrame, chart_type: str, **kwargs) -> dict:
    """
    Creates visualizations. Normalizes parameter names before routing
    so the LLM doesn't need to remember exact parameter names.
    """
    # Normalize common parameter aliases
    if "x" in kwargs and "x_col" not in kwargs:
        kwargs["x_col"] = kwargs.pop("x")
    if "y" in kwargs and "y_col" not in kwargs:
        kwargs["y_col"] = kwargs.pop("y")
    if "column" in kwargs and "numeric_col" not in kwargs:
        kwargs["numeric_col"] = kwargs.pop("column")
    if "value_col" in kwargs and "numeric_col" not in kwargs:
        kwargs["numeric_col"] = kwargs.pop("value_col")
    if "color" in kwargs and "color_col" not in kwargs:
        kwargs["color_col"] = kwargs.pop("color")

    # For line chart — if x_col not provided, try to find a date column
    if chart_type == "line" and "x_col" not in kwargs:
        date_cols = [col for col in df.columns 
                    if pd.api.types.is_datetime64_any_dtype(df[col])]
        if date_cols:
            kwargs["x_col"] = date_cols[0]

    # For line/scatter — if y_col not provided, use first numeric col
    if chart_type in ["line", "scatter"] and "y_col" not in kwargs:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            kwargs["y_col"] = numeric_cols[0]

    # For scatter — if x_col not provided, use second numeric col
    if chart_type == "scatter" and "x_col" not in kwargs:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if len(numeric_cols) >= 2:
            kwargs["x_col"] = numeric_cols[1]

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
    elif chart_type == "line":
        return _line_chart(df, **kwargs)
    elif chart_type == "pie":
        return _pie_chart(df, **kwargs)
    elif chart_type == "pivot_heatmap":
        return _pivot_heatmap(df, **kwargs)
    else:
        return {"error": f"Unknown chart type: {chart_type}. Supported: histogram, bar, scatter, correlation_heatmap, boxplot, line, pie, pivot_heatmap"}


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
        color_discrete_sequence=["#636EFA"],
        nbins=50
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