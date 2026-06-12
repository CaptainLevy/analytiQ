import pandas as pd
import numpy as np


def convert_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_types(i) for i in obj]
    return obj


def run_eda(df: pd.DataFrame) -> dict:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    shape = df.shape
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    duplicates = df.duplicated().sum()

    numeric_summary = {}
    for col in numeric_cols:
        numeric_summary[col] = {
            "mean": round(float(df[col].mean()), 4),
            "median": round(float(df[col].median()), 4),
            "std": round(float(df[col].std()), 4),
            "min": round(float(df[col].min()), 4),
            "max": round(float(df[col].max()), 4),
            "skewness": round(float(df[col].skew()), 4),
        }

    categorical_summary = {}
    for col in categorical_cols:
        categorical_summary[col] = {
            "unique_values": int(df[col].nunique()),
            "top_value": df[col].mode()[0] if not df[col].mode().empty else None,
            "top_frequency": int(df[col].value_counts().iloc[0]) if len(df[col].value_counts()) > 0 else 0,
        }

    quality_issues = []
    for col in df.columns:
        pct = missing_pct[col]
        if pct > 0:
            quality_issues.append(f"'{col}' has {pct}% missing values")
    if duplicates > 0:
        quality_issues.append(f"{int(duplicates)} duplicate rows found")

    result = {
        "shape": {"rows": int(shape[0]), "columns": int(shape[1])},
        "column_types": {
            "numeric": numeric_cols,
            "categorical": categorical_cols,
        },
        "missing_values": {k: float(v) for k, v in missing_pct[missing_pct > 0].to_dict().items()},
        "duplicates": int(duplicates),
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
        "quality_issues": quality_issues,
    }

    return convert_types(result)