import pytest
import pandas as pd
import numpy as np
from app.tools.eda import run_eda
from app.tools.stats import run_stats, run_aggregation
from app.tools.viz import run_viz


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "sales": [100, 200, 300, 400, 500, 150, 250, 350],
        "profit": [10, 22, 28, 45, 48, 15, 27, 38],
        "region": ["North", "North", "South", "South", "North", "South", "East", "East"],
        "category": ["A", "B", "A", "B", "A", "B", "A", "B"]
    })


@pytest.fixture
def df_with_missing():
    return pd.DataFrame({
        "sales": [100, 200, None, 400, 500],
        "region": ["North", "South", "North", "East", "South"]
    })


# ── EDA Tests ─────────────────────────────────────────────────────────────────

def test_eda_shape(sample_df):
    result = run_eda(sample_df)
    assert result["shape"]["rows"] == 8
    assert result["shape"]["columns"] == 4


def test_eda_column_types(sample_df):
    result = run_eda(sample_df)
    assert "sales" in result["column_types"]["numeric"]
    assert "region" in result["column_types"]["categorical"]


def test_eda_detects_missing(df_with_missing):
    result = run_eda(df_with_missing)
    assert "sales" in result["missing_values"]
    assert result["missing_values"]["sales"] == 20.0


def test_eda_no_missing(sample_df):
    result = run_eda(sample_df)
    assert result["missing_values"] == {}


def test_eda_json_serializable(sample_df):
    import json
    result = run_eda(sample_df)
    # Should not raise
    json.dumps(result)


# ── Stats Tests ───────────────────────────────────────────────────────────────

def test_correlation_returns_matrix(sample_df):
    result = run_stats(sample_df, "correlation")
    assert "matrix" in result
    assert "sales" in result["matrix"]
    assert "profit" in result["matrix"]["sales"]


def test_correlation_strongest_pairs(sample_df):
    result = run_stats(sample_df, "correlation")
    assert len(result["strongest_pairs"]) >= 1
    assert "correlation" in result["strongest_pairs"][0]


def test_ttest_significant_difference():
    df = pd.DataFrame({
        "sales": [100, 110, 105, 108, 500, 510, 505, 508],
        "group": ["A", "A", "A", "A", "B", "B", "B", "B"]
    })
    result = run_stats(df, "ttest", numeric_col="sales", group_col="group")
    assert result["significant"] == True
    assert result["p_value"] < 0.05


def test_ttest_no_difference():
    df = pd.DataFrame({
        "sales": [100, 105, 110, 95, 102, 108],
        "group": ["A", "A", "A", "B", "B", "B"]
    })
    result = run_stats(df, "ttest", numeric_col="sales", group_col="group")
    assert "p_value" in result
    assert result["significant"] == False


def test_anova_requires_three_groups(sample_df):
    result = run_stats(sample_df, "anova", numeric_col="sales", group_col="region")
    assert "f_statistic" in result
    assert "p_value" in result


def test_normality_returns_interpretation(sample_df):
    result = run_stats(sample_df, "normality", numeric_col="sales")
    assert "is_normal" in result
    assert "interpretation" in result


def test_unknown_analysis_type(sample_df):
    result = run_stats(sample_df, "unknown_type")
    assert "error" in result


# ── Aggregation Tests ─────────────────────────────────────────────────────────

def test_aggregation_mean(sample_df):
    result = run_aggregation(sample_df, "region", "sales", "mean")
    assert "top_group" in result
    assert "results" in result
    assert isinstance(result["top_value"], float)


def test_aggregation_sum(sample_df):
    result = run_aggregation(sample_df, "category", "profit", "sum")
    assert result["aggregation"] == "sum"


def test_aggregation_invalid_column(sample_df):
    result = run_aggregation(sample_df, "nonexistent", "sales", "mean")
    assert "error" in result


# ── Viz Tests ─────────────────────────────────────────────────────────────────

def test_histogram_success(sample_df):
    result = run_viz(sample_df, "histogram", numeric_col="sales")
    assert result["success"] == True
    assert "chart_json" in result


def test_bar_success(sample_df):
    result = run_viz(sample_df, "bar", group_col="region", numeric_col="sales")
    assert result["success"] == True


def test_scatter_success(sample_df):
    result = run_viz(sample_df, "scatter", x_col="sales", y_col="profit")
    assert result["success"] == True


def test_correlation_heatmap_success(sample_df):
    result = run_viz(sample_df, "correlation_heatmap")
    assert result["success"] == True


def test_boxplot_success(sample_df):
    result = run_viz(sample_df, "boxplot", numeric_col="sales", group_col="region")
    assert result["success"] == True


def test_viz_invalid_column(sample_df):
    result = run_viz(sample_df, "histogram", numeric_col="nonexistent")
    assert "error" in result


def test_viz_unknown_chart_type(sample_df):
    result = run_viz(sample_df, "unknown_chart")
    assert "error" in result