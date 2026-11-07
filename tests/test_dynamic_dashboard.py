import numpy as np
import pandas as pd
import pytest

from src.analysis.dashboard_service import build_recommended_chart, column_kpis, generate_insights, recommend_charts
from src.analysis.profiling_service import profile_dataframe


def dashboard_examples():
    return {
        "sales": pd.DataFrame({"item": ["A", "B", "A", "C"], "amount": [12, 14, 12, 30], "period": pd.date_range("2024-01-01", periods=4)}),
        "hr": pd.DataFrame({"employee_id": [10, 11, 12], "department": ["Ops", "Tech", "Ops"], "tenure": [1.2, 3.5, 2.1], "active": [True, False, True]}),
        "healthcare": pd.DataFrame({"patient_id": [1, 2, 3], "measure": [5.1, 7.4, 6.3], "status": ["stable", "review", "stable"]}),
        "financial": pd.DataFrame({"instrument": ["X", "Y", "Z"], "balance": [100.0, 140.0, 90.0], "change": [2.0, -3.0, 1.0]}),
        "numeric_only": pd.DataFrame({"first": [1, 2, 4, 8], "second": [5, 3, 2, 1]}),
        "categorical_only": pd.DataFrame({"kind": ["a", "b", "a"], "flag": [True, False, True]}),
        "datetime": pd.DataFrame({"when": pd.date_range("2022-01-01", periods=8, freq="D"), "reading": np.arange(8)}),
        "missing": pd.DataFrame({"measure": [1.0, np.nan, 3.0], "group": ["a", None, "b"]}),
        "ids": pd.DataFrame({"record_id": np.arange(100, 130), "score": np.linspace(1, 4, 30), "group": ["x", "y"] * 15}),
        "mixed": pd.DataFrame({"mixed_field": pd.Series([1, "two", 3, None], dtype=object), "value": [10, 20, 30, 40]}),
        "tiny": pd.DataFrame({"only": [1]}),
        "large": pd.DataFrame({"metric_a": np.arange(5000), "metric_b": np.arange(5000) * 3, "segment": ["x", "y"] * 2500}),
    }


@pytest.mark.parametrize("name", list(dashboard_examples()))
def test_dashboard_services_adapt_across_diverse_tables(name):
    frame = dashboard_examples()[name]
    original = frame.copy(deep=True)
    profiles = profile_dataframe(frame)
    assert len(profiles) == len(frame.columns)
    recs = recommend_charts(frame, profiles)
    for recommendation in recs:
        figure = build_recommended_chart(frame, recommendation)
        assert hasattr(figure, "to_plotly_json")
    if len(frame.columns):
        assert column_kpis(frame, frame.columns[0])
    generate_insights(frame, profiles)
    pd.testing.assert_frame_equal(frame, original)


def test_profiles_mark_identifier_boolean_mixed_and_constant_columns():
    frame = pd.DataFrame({
        "transaction_id": range(30), "enabled": [True, False] * 15,
        "mixed": pd.Series([1, "text"] * 15, dtype=object), "fixed": [7] * 30,
        "free_text": [f"a long piece of text number {i}" for i in range(30)],
    })
    result = {item.name: item for item in profile_dataframe(frame)}
    assert result["transaction_id"].id_like
    assert result["enabled"].semantic_type == "boolean"
    assert result["mixed"].semantic_type == "mixed"
    assert result["fixed"].constant
    assert result["free_text"].semantic_type == "text"


def test_insights_are_computed_from_current_filtered_rows():
    frame = pd.DataFrame({"reading": [1, 2, 3, 4, 5, 1000, np.nan], "kind": ["a", "a", "b", "b", "a", "b", "b"]})
    insights = generate_insights(frame)
    joined = " ".join(evidence for _, _, evidence in insights)
    assert "100.0%" not in joined
    assert "missing cells" in joined
    assert "IQR" in joined


