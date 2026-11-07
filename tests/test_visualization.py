import numpy as np
import pandas as pd
import pytest

from src.visualization.chart_builders import (
    CHART_TYPES,
    ChartConfigurationError,
    build_chart,
)


@pytest.fixture
def eda_frame():
    count = 36
    dates = pd.date_range("2023-01-01", periods=count, freq="MS")
    values = 100 + np.sin(np.arange(count) * 2 * np.pi / 12) * 20 + np.arange(count) * 0.5
    return pd.DataFrame({
        "date": dates,
        "sales": values,
        "visits": np.arange(count) * 3 + 20,
        "volume": np.arange(count) + 5,
        "category": np.where(np.arange(count) % 2, "Retail", "Online"),
        "region": np.where(np.arange(count) % 3, "West", "North"),
        "missing": [np.nan if i % 7 == 0 else float(i) for i in range(count)],
    })


@pytest.mark.parametrize("chart_type", CHART_TYPES)
def test_every_chart_type_builds_a_plotly_figure(eda_frame, chart_type):
    configs = {
        "Histogram": {"x": "sales"},
        "Box Plot": {"x": "sales", "color": "category"},
        "Violin Plot": {"x": "sales", "color": "category"},
        "Density Plot": {"x": "sales"},
        "ECDF Plot": {"x": "sales"},
        "Bar Chart": {"x": "category", "y": "sales"},
        "Count Plot": {"x": "category"},
        "Pie Chart": {"x": "category"},
        "Stacked Bar Chart": {"x": "category", "y": "sales", "color": "region"},
        "Scatter Plot": {"x": "visits", "y": "sales", "color": "category"},
        "Bubble Chart": {"x": "visits", "y": "sales", "size": "volume", "color": "category"},
        "Correlation Heatmap": {"correlation_columns": ["sales", "visits", "volume"]},
        "Pair Plot": {"pair_columns": ["sales", "visits", "volume"]},
        "Grouped Box Plot": {"x": "sales", "color": "category"},
        "Line Chart": {"x": "date", "y": "sales"},
        "Rolling Average Plot": {"x": "date", "y": "sales", "rolling_window": 3},
        "Seasonal Decomposition": {"x": "date", "y": "sales", "seasonal_period": 12},
        "Missing Values Bar Chart": {},
        "Missing Values Heatmap": {},
        "Outlier Analysis Plot": {"x": "volume"},
    }
    figure = build_chart(eda_frame, chart_type, **configs[chart_type])
    assert hasattr(figure, "to_plotly_json")
    assert len(figure.data) > 0


def test_line_chart_accepts_datetime_strings_and_sorts_them():
    frame = pd.DataFrame({"month": ["2024-02-01", "2024-01-01", "2024-03-01"], "metric": [2, 1, 3]})
    figure = build_chart(frame, "Line Chart", x="month", y="metric")
    assert list(figure.data[0].x) == sorted(figure.data[0].x)


def test_histogram_rejects_categorical_primary_column(eda_frame):
    with pytest.raises(ChartConfigurationError, match="must be numerical"):
        build_chart(eda_frame, "Histogram", x="category")


def test_pair_plot_limits_number_of_columns(eda_frame):
    with pytest.raises(ChartConfigurationError, match="no more than 5"):
        build_chart(eda_frame, "Pair Plot", pair_columns=["sales", "visits", "volume", "missing", "sales", "visits"])


def test_seasonal_decomposition_requires_two_periods(eda_frame):
    with pytest.raises(ChartConfigurationError, match="at least two full periods"):
        build_chart(eda_frame.iloc[:10], "Seasonal Decomposition", x="date", y="sales", seasonal_period=6)
