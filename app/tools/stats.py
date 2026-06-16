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
    elif analysis_type == "mannwhitney":
        return _mannwhitney(df, **kwargs)
    elif analysis_type == "kruskal":
        return _kruskal(df, **kwargs)
    elif analysis_type == "check_assumptions":
        return check_assumptions(df, **kwargs)
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

def run_aggregation(df: pd.DataFrame, group_col: str, 
                    numeric_col: str, agg: str = "mean") -> dict:
    """Simple group-by aggregation."""
    if group_col not in df.columns or numeric_col not in df.columns:
        return {"error": "Column not found"}

    grouped = df.groupby(group_col)[numeric_col].agg(agg).round(2)
    sorted_result = grouped.sort_values(ascending=False)

    return {
        "analysis": "aggregation",
        "group_col": group_col,
        "numeric_col": numeric_col,
        "aggregation": agg,
        "results": {str(k): float(v) for k, v in sorted_result.items()},
        "top_group": str(sorted_result.index[0]),
        "top_value": float(sorted_result.iloc[0])
    }

def run_topn(df: pd.DataFrame, group_col: str,
             numeric_col: str, n: int = 5,
             agg: str = "sum") -> dict:
    """Returns top N groups by aggregated value."""
    if group_col not in df.columns or numeric_col not in df.columns:
        return {"error": "Column not found"}

    grouped = df.groupby(group_col)[numeric_col].agg(agg).round(2)
    top_n = grouped.nlargest(n)
    bottom_n = grouped.nsmallest(n)

    return {
        "analysis": "top_n",
        "group_col": group_col,
        "numeric_col": numeric_col,
        "n": n,
        "aggregation": agg,
        "top": {str(k): float(v) for k, v in top_n.items()},
        "bottom": {str(k): float(v) for k, v in bottom_n.items()},
        "top_group": str(top_n.index[0]),
        "top_value": float(top_n.iloc[0])
    }

def _mannwhitney(df: pd.DataFrame, numeric_col: str,
                 group_col: str, **kwargs) -> dict:
    groups = df[group_col].dropna().unique()
    if len(groups) != 2:
        return {"error": "Mann-Whitney requires exactly 2 groups"}

    g1 = df[df[group_col] == groups[0]][numeric_col].dropna()
    g2 = df[df[group_col] == groups[1]][numeric_col].dropna()

    stat, p_value = stats.mannwhitneyu(g1, g2, alternative="two-sided")

    return {
        "analysis": "mann_whitney",
        "group_col": group_col,
        "numeric_col": numeric_col,
        "group1": {"name": str(groups[0]), "median": round(float(g1.median()), 4)},
        "group2": {"name": str(groups[1]), "median": round(float(g2.median()), 4)},
        "statistic": round(float(stat), 4),
        "p_value": round(float(p_value), 6),
        "significant": bool(p_value < 0.05),
        "interpretation": (
            f"There IS a significant difference in {numeric_col} between "
            f"{groups[0]} and {groups[1]} (p={round(float(p_value), 4)})"
            if p_value < 0.05 else
            f"There is NO significant difference in {numeric_col} between "
            f"{groups[0]} and {groups[1]} (p={round(float(p_value), 4)})"
        ),
        "note": "Mann-Whitney U used because data is not normally distributed"
    }


def _kruskal(df: pd.DataFrame, numeric_col: str,
             group_col: str, **kwargs) -> dict:
    groups = df[group_col].dropna().unique()
    if len(groups) < 3:
        return {"error": "Kruskal-Wallis requires at least 3 groups"}

    group_data = [df[df[group_col] == g][numeric_col].dropna()
                  for g in groups]
    stat, p_value = stats.kruskal(*group_data)

    group_medians = {
        str(g): round(float(df[df[group_col] == g][numeric_col].median()), 4)
        for g in groups
    }

    return {
        "analysis": "kruskal_wallis",
        "group_col": group_col,
        "numeric_col": numeric_col,
        "group_medians": group_medians,
        "statistic": round(float(stat), 4),
        "p_value": round(float(p_value), 6),
        "significant": bool(p_value < 0.05),
        "interpretation": (
            f"There IS a significant difference in {numeric_col} across "
            f"{group_col} groups (p={round(float(p_value), 4)})"
            if p_value < 0.05 else
            f"There is NO significant difference in {numeric_col} across "
            f"{group_col} groups (p={round(float(p_value), 4)})"
        ),
        "note": "Kruskal-Wallis used because data is not normally distributed"
    }


def check_assumptions(df: pd.DataFrame, numeric_col: str,
                      group_col: str = None) -> dict:
    """
    Checks statistical assumptions before running tests.
    Returns recommended test based on data properties.
    """
    col_data = df[numeric_col].dropna()
    results = {}

    # Normality check
    if len(col_data) >= 3:
        stat, p_norm = stats.shapiro(col_data)
        results["normality"] = {
            "is_normal": bool(p_norm > 0.05),
            "p_value": round(float(p_norm), 4),
            "test": "Shapiro-Wilk"
        }

    # Group specific checks
    if group_col and group_col in df.columns:
        groups = df[group_col].dropna().unique()
        n_groups = len(groups)

        # Levene's test for equal variances
        group_data = [df[df[group_col] == g][numeric_col].dropna()
                      for g in groups]
        if len(group_data) >= 2:
            lev_stat, lev_p = stats.levene(*group_data)
            results["equal_variances"] = {
                "equal": bool(lev_p > 0.05),
                "p_value": round(float(lev_p), 4),
                "test": "Levene"
            }

        is_normal = results.get("normality", {}).get("is_normal", True)
        equal_var = results.get("equal_variances", {}).get("equal", True)

        if n_groups == 2:
            if is_normal and equal_var:
                results["recommended_test"] = "ttest"
                results["reason"] = "Data is normal with equal variances — use Independent T-test"
            elif is_normal and not equal_var:
                results["recommended_test"] = "ttest_welch"
                results["reason"] = "Data is normal but unequal variances — use Welch T-test"
            else:
                results["recommended_test"] = "mannwhitney"
                results["reason"] = "Data is not normal — use Mann-Whitney U (non-parametric)"
        else:
            if is_normal and equal_var:
                results["recommended_test"] = "anova"
                results["reason"] = "Data is normal with equal variances — use ANOVA"
            else:
                results["recommended_test"] = "kruskal"
                results["reason"] = "Data is not normal — use Kruskal-Wallis (non-parametric)"

    return results