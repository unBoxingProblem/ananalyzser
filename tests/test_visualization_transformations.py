import pandas as pd

from src.visualization.chart_builders import build_chart
from src.visualization.transformation_service import prepare_visualization


def test_scatter_one_hot_and_multi_y_are_temporary():
    frame = pd.DataFrame({"x": [1, 2, 3], "region": ["North", "South", "North"], "sales": [10, 20, 30]})
    original = frame.copy(deep=True)
    prepared = prepare_visualization(frame, "Scatter Plot", "x", ["region", "sales"])
    assert prepared.error is None
    assert prepared.dataframe is not frame
    assert any("one-hot encoded" in item for item in prepared.notices)
    assert list(frame.columns) == list(original.columns)
    pd.testing.assert_frame_equal(frame, original)
    figure = build_chart(prepared.dataframe, "Scatter Plot", x=prepared.x,
                         y_columns=prepared.y_columns, color=prepared.color)
    assert len(figure.data) >= 2


def test_bar_numeric_x_is_binned_temporarily():
    frame = pd.DataFrame({"amount": [1, 2, 3, 4], "sales": [10, 20, 30, 40]})
    prepared = prepare_visualization(frame, "Bar Chart", "amount", ["sales"], bins=2)
    assert prepared.error is None
    assert prepared.x != "amount"
    assert "amount" in frame.columns
    assert len(prepared.dataframe[prepared.x].cat.categories) == 2


def test_categorical_scatter_x_explains_incompatibility_without_encoding():
    frame = pd.DataFrame({"region": ["East", "West"], "sales": [10, 20]})
    prepared = prepare_visualization(frame, "Scatter Plot", "region", ["sales"])
    assert prepared.error
    assert "categorical" in prepared.error
    pd.testing.assert_frame_equal(frame, pd.DataFrame({"region": ["East", "West"], "sales": [10, 20]}))


def test_correlation_categorical_encoding_is_explicit_and_temporary():
    frame = pd.DataFrame({"sales": [10, 20, 30], "region": ["N", "S", "N"]})
    prepared = prepare_visualization(frame, "Correlation Heatmap", None, [],
                                     correlation_columns=["sales", "region"],
                                     encode_columns=["region"], categorical_encoding="one-hot")
    assert prepared.error is None
    assert any("one-hot encoded" in item for item in prepared.notices)
    assert list(frame.columns) == ["sales", "region"]
    fig = build_chart(prepared.dataframe, "Correlation Heatmap", correlation_columns=prepared.correlation_columns)
    assert len(fig.data) == 1


def test_normalization_creates_temporary_series_only():
    frame = pd.DataFrame({"x": [1, 2, 3], "a": [10, 20, 30], "b": [100, 200, 300]})
    original = frame.copy(deep=True)
    prepared = prepare_visualization(frame, "Scatter Plot", "x", ["a", "b"], normalization="Z-score")
    assert prepared.error is None
    assert prepared.y_columns != ["a", "b"]
    assert any("normalized" in notice for notice in prepared.notices)
    pd.testing.assert_frame_equal(frame, original)


def test_categorical_y_on_bar_uses_counts_not_arbitrary_codes():
    frame = pd.DataFrame({"region": ["N", "N", "S"], "channel": ["Web", "Store", "Web"]})
    prepared = prepare_visualization(frame, "Bar Chart", "region", ["channel"])
    fig = build_chart(prepared.dataframe, "Bar Chart", x=prepared.x, y_columns=prepared.y_columns)
    assert fig.data
    assert "channel" in fig.layout.title.text
