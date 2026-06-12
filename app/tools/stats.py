import pandas as pd
import numpy as np
from scipy import stats


def run_stats(df: pd.DataFrame, analysis_type: str, **kwargs) -> dict:
    """
    Runs statistical analyses on the dataframe.
    
    Supported analysis_type values:
    - correlation: correlation matrix for numeric columns
    - ttest: independent samples t-test between two groups
    - anova: one-way ANOVA across multiple groups
    - normality: Shapiro-Wilk normality test
    """

    if analysis_type == "correlation":
        return _correlation(df, **kwargs)
    elif analysis_type == "ttest":
        return _ttest(df, **kwargs)
    elif analysis_type == "anova":
        return _anova(df, **kwargs)
    elif analysis_type == "normality":
        return _normality(df, **kwargs)
    else:
        return {"error": f"Unknown analysis type: {analysis_type}"}


def _correlation(df: pd.DataFrame, **kwargs) -> dict:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return {"error": "Need at least 2 numeric columns for correlation"}
    
    corr_matrix = numeric_df.corr().round(4)
    
    # Find strongest pairs
    pairs = []
    cols = corr_matrix.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append({
                "col1": cols[i],
                "col2": cols[j],
                "correlation": round(float(corr_matrix.iloc[i, j]), 4)
            })
    
    pairs_sorted = sorted(pairs, key=lambda x: abs(x["correlation"]), reverse=True)

    return {
        "analysis": "correlation",
        "matrix": {col: {c: float(v) for c, v in row.items()} 
                   for col, row in corr_matrix.to_dict().items()},
        "strongest_pairs": pairs_sorted[:5]
    }


def _ttest(df: pd.DataFrame, numeric_col: str, group_col: str, **kwargs) -> dict:
    groups = df[group_col].dropna().unique()
    if len(groups) != 2:
        return {"error": f"T-test requires exactly 2 groups, found {len(groups)}"}
    
    g1 = df[df[group_col] == groups[0]][numeric_col].dropna()
    g2 = df[df[group_col] == groups[1]][numeric_col].dropna()
    
    t_stat, p_value = stats.ttest_ind(g1, g2)

    return {
        "analysis": "ttest",
        "group_col": group_col,
        "numeric_col": numeric_col,
        "group1": {"name": str(groups[0]), "mean": round(float(g1.mean()), 4), "n": len(g1)},
        "group2": {"name": str(groups[1]), "mean": round(float(g2.mean()), 4), "n": len(g2)},
        "t_statistic": round(float(t_stat), 4),
        "p_value": round(float(p_value), 6),
        "significant": bool(p_value < 0.05),
        "interpretation": (
            f"There IS a statistically significant difference in {numeric_col} "
            f"between {groups[0]} and {groups[1]} (p={round(float(p_value), 4)})"
            if p_value < 0.05 else
            f"There is NO statistically significant difference in {numeric_col} "
            f"between {groups[0]} and {groups[1]} (p={round(float(p_value), 4)})"
        )
    }


def _anova(df: pd.DataFrame, numeric_col: str, group_col: str, **kwargs) -> dict:
    groups = df[group_col].dropna().unique()
    if len(groups) < 3:
        return {"error": "ANOVA requires at least 3 groups"}
    
    group_data = [df[df[group_col] == g][numeric_col].dropna() for g in groups]
    f_stat, p_value = stats.f_oneway(*group_data)

    group_means = {
        str(g): round(float(df[df[group_col] == g][numeric_col].mean()), 4)
        for g in groups
    }

    return {
        "analysis": "anova",
        "group_col": group_col,
        "numeric_col": numeric_col,
        "group_means": group_means,
        "f_statistic": round(float(f_stat), 4),
        "p_value": round(float(p_value), 6),
        "significant": bool(p_value < 0.05),
        "interpretation": (
            f"There IS a statistically significant difference in {numeric_col} "
            f"across {group_col} groups (p={round(float(p_value), 4)})"
            if p_value < 0.05 else
            f"There is NO statistically significant difference in {numeric_col} "
            f"across {group_col} groups (p={round(float(p_value), 4)})"
        )
    }


def _normality(df: pd.DataFrame, numeric_col: str, **kwargs) -> dict:
    col_data = df[numeric_col].dropna()
    if len(col_data) < 3:
        return {"error": "Need at least 3 data points for normality test"}
    
    stat, p_value = stats.shapiro(col_data)

    return {
        "analysis": "normality",
        "column": numeric_col,
        "shapiro_statistic": round(float(stat), 4),
        "p_value": round(float(p_value), 6),
        "is_normal": bool(p_value > 0.05),
        "interpretation": (
            f"{numeric_col} appears normally distributed (p={round(float(p_value), 4)})"
            if p_value > 0.05 else
            f"{numeric_col} does NOT appear normally distributed (p={round(float(p_value), 4)})"
        )
    }