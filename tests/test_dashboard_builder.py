import json

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.dashboard.chart_service import (
    DASHBOARD_CHART_TYPES, DashboardChartError, build_dashboard_chart, validate_chart_config,
)
from src.dashboard.ai_dashboard_service import AIDashboardError, analyze_dashboard_patterns, create_ai_dashboard, recommend_dashboard_charts
from src.dashboard.dashboard_store import DashboardStore
from src.dataset.dataset_registry import DatasetRegistry
from src.upload.upload_manager import UploadManager
from tests.helpers import Upload


@pytest.fixture
def chart_frame():
    return pd.DataFrame({
        "period": pd.date_range("2024-01-01", periods=6, freq="D"),
        "category": ["A", "B", "A", "C", "B", "A"],
        "group": ["North", "South", "North", "West", "South", "West"],
        "value": [5, 7, 8, 4, 10, 12],
        "other": [3, 8, 5, 9, 4, 6],
        "size": [10, 12, 14, 11, 16, 18],
        "text": ["A sufficiently descriptive sentence for group one", "A slightly longer descriptive sentence for group two",
                 "Another detailed example sentence for group one", "A compact yet descriptive sentence for group three",
                 "A second descriptive paragraph in group two", "A final descriptive sentence for group one"],
    })


def configs_for_all_types():
    result = {
        "Bar Chart": {"x": "category", "y_columns": ["value"]},
        "Line Chart": {"x": "period", "y_columns": ["value", "other"]},
        "Scatter Plot": {"x": "value", "y_columns": ["other"]},
        "Histogram": {"x": "value"},
        "Box Plot": {"x": None, "y_columns": ["value"], "group_by": "category"},
        "Violin Plot": {"x": None, "y_columns": ["value"], "group_by": "category"},
        "Pie Chart": {"x": "category", "y_columns": ["value"]},
        "Donut Chart": {"x": "category", "y_columns": ["value"]},
        "Area Chart": {"x": "period", "y_columns": ["value", "other"]},
        "Stacked Bar Chart": {"x": "category", "y_columns": ["value"], "group_by": "group"},
        "Grouped Bar Chart": {"x": "category", "y_columns": ["value"], "group_by": "group"},
        "Heatmap": {"x": None, "y_columns": ["value", "other"]},
        "Count Plot": {"x": "category"},
        "Bubble Chart": {"x": "value", "y_columns": ["other"], "size_column": "size"},
        "Treemap": {"x": "category", "y_columns": ["value"], "group_by": "group"},
        "Text Length": {"x": "text"},
        "Text Length by Group": {"x": "text", "group_by": "group"},
    }
    return {chart: {"chart_type": chart, "aggregation": "Mean", **config} for chart, config in result.items()}


@pytest.mark.parametrize("chart_type", DASHBOARD_CHART_TYPES)
def test_dashboard_chart_types_build_interactive_plotly_figures(chart_frame, chart_type):
    config = configs_for_all_types()[chart_type]
    assert validate_chart_config(chart_frame, config) is None
    figure = build_dashboard_chart(chart_frame, config)
    assert hasattr(figure, "to_plotly_json")
    assert figure.data


def test_chart_configuration_rejects_invalid_types_without_encoding(chart_frame):
    error = validate_chart_config(chart_frame, {"chart_type": "Histogram", "x": "category"})
    assert error and "numeric X" in error
    error = validate_chart_config(chart_frame, {"chart_type": "Scatter Plot", "x": "category", "y_columns": ["value"]})
    assert error and "numeric X" in error
    with pytest.raises(DashboardChartError):
        build_dashboard_chart(chart_frame, {"chart_type": "Line Chart", "x": "category", "y_columns": ["value"]})


def test_missing_dataset_columns_are_reported_without_substitution(chart_frame):
    error = validate_chart_config(chart_frame, {"chart_type": "Histogram", "x": "removed_column"})
    assert error and "unavailable" in error


def test_dataset_switch_uses_each_dataset_schema_without_retargeting_saved_configs():
    first = pd.DataFrame({"category": ["A", "B"], "value": [1, 2]})
    second = pd.DataFrame({"kind": ["X", "Y"], "measure": [8, 9]})
    config_a = {"chart_type": "Count Plot", "dataset_id": "dataset-a", "x": "category"}
    config_b = {"chart_type": "Count Plot", "dataset_id": "dataset-b", "x": "kind"}
    assert validate_chart_config(first, config_a) is None
    assert validate_chart_config(second, config_b) is None
    assert validate_chart_config(second, config_a) and "unavailable" in validate_chart_config(second, config_a)
    assert list(build_dashboard_chart(first, config_a).data[0].x) == ["A", "B"]
    assert list(build_dashboard_chart(second, config_b).data[0].x) == ["X", "Y"]


def test_per_chart_filter_recalculates_from_selected_rows_without_mutating_dataset(chart_frame):
    before = chart_frame.copy(deep=True)
    config = {"chart_type": "Count Plot", "x": "category", "filters": {"column": "group", "kind": "categorical", "values": ["North"]}}
    figure = build_dashboard_chart(chart_frame, config)
    assert list(figure.data[0].x) == ["A"]
    assert list(figure.data[0].y) == [2]
    pd.testing.assert_frame_equal(chart_frame, before)


def test_dashboard_store_saves_loads_duplicates_and_deletes_isolated_definitions(tmp_path):
    store = DashboardStore(tmp_path / "dashboards.sqlite")
    first = store.create_dashboard("First")
    first["charts"] = [
        {"chart_id": "chart-1", "dataset_id": "dataset-a", "dataset_name": "a.csv", "chart_type": "Histogram", "x": "value", "filters": {"kind": "categorical", "values": [np.int64(3)]}},
        {"chart_id": "chart-2", "dataset_id": "dataset-b", "dataset_name": "b.csv", "chart_type": "Count Plot", "x": "kind", "filters": {}},
    ]
    saved = store.save_dashboard(first)
    assert store.get_dashboard(saved["dashboard_id"])["charts"] == first["charts"]

    # Editing one chart and saving does not overwrite the other chart.
    edited = store.get_dashboard(saved["dashboard_id"])
    edited["charts"][0]["title"] = "Updated title"
    store.save_dashboard(edited)
    loaded = store.get_dashboard(saved["dashboard_id"])
    assert loaded["charts"][0]["title"] == "Updated title"
    assert loaded["charts"][1]["dataset_id"] == "dataset-b"

    added = store.add_chart(saved["dashboard_id"], {"chart_type": "Histogram", "dataset_id": "dataset-c"})
    assert len(added["charts"]) == 3
    added_chart_id = added["charts"][-1]["chart_id"]
    updated = store.update_chart(saved["dashboard_id"], added_chart_id, {"title": "Edited", "dataset_id": "dataset-c"})
    assert updated["charts"][-1]["title"] == "Edited"
    duplicated_chart = store.duplicate_chart(saved["dashboard_id"], added_chart_id)
    assert len(duplicated_chart["charts"]) == 4
    assert duplicated_chart["charts"][-1]["chart_id"] != added_chart_id
    moved = store.move_chart(saved["dashboard_id"], added_chart_id, -1)
    assert moved["charts"][1]["chart_id"] == added_chart_id
    duplicate = store.duplicate_dashboard(saved["dashboard_id"])
    assert duplicate["dashboard_id"] != saved["dashboard_id"]
    assert len(duplicate["charts"]) == 4
    assert duplicate["charts"][0]["chart_id"] != loaded["charts"][0]["chart_id"]
    store.delete_chart(duplicate["dashboard_id"], duplicate["charts"][0]["chart_id"])
    assert len(store.get_dashboard(saved["dashboard_id"])["charts"]) == 4
    assert len(store.get_dashboard(duplicate["dashboard_id"])["charts"]) == 3
    removed = store.delete_chart(saved["dashboard_id"], added_chart_id)
    assert len(removed["charts"]) == 3
    assert store.delete_dashboard(duplicate["dashboard_id"])
    assert store.get_dashboard(duplicate["dashboard_id"]) is None


def test_ai_dashboard_builder_saves_valid_interactive_charts(tmp_path, chart_frame):
    store = DashboardStore(tmp_path / "ai_dashboards.sqlite")
    before = chart_frame.copy(deep=True)
    result = create_ai_dashboard(store, chart_frame, "dataset-1", "metrics.csv", [
        {"chart_type": "Bar Chart", "x": "category", "y_columns": ["value"], "aggregation": "Mean", "title": "Value by category"},
        {"chart_type": "Histogram", "x": "value", "title": "Value distribution"},
    ], "AI review")
    saved = store.get_dashboard(result["dashboard_id"])
    assert result["chart_count"] == 2
    assert saved["name"] == "AI review"
    assert len(saved["charts"]) == 2
    assert saved["charts"][0]["dataset_id"] == "dataset-1"
    assert build_dashboard_chart(chart_frame, saved["charts"][0]).data
    pd.testing.assert_frame_equal(chart_frame, before)


def test_ai_dashboard_builder_rejects_invalid_chart_before_saving(tmp_path, chart_frame):
    store = DashboardStore(tmp_path / "ai_dashboards.sqlite")
    with pytest.raises(AIDashboardError, match="requires a numeric X"):
        create_ai_dashboard(store, chart_frame, "dataset-1", "metrics.csv", [
            {"chart_type": "Histogram", "x": "category"},
        ])
    assert store.list_dashboards() == []


def test_local_dashboard_recommendations_adapt_to_detected_types(chart_frame):
    charts = recommend_dashboard_charts(chart_frame)
    assert charts
    assert any(chart["chart_type"] == "Bar Chart" and chart["x"] == "category" for chart in charts)
    assert all(validate_chart_config(chart_frame, chart) is None for chart in charts)


def test_dashboard_pattern_engine_reports_measured_rationales_and_text_relationships(chart_frame):
    analysis = analyze_dashboard_patterns(chart_frame)
    assert analysis["profiled_columns"] == len(chart_frame.columns)
    assert analysis["rows"] == len(chart_frame)
    assert analysis["suggested_charts"]
    assert all(item["rationale"] and 0 <= item["score"] <= 1 for item in analysis["suggested_charts"])
    text_by_group = [item for item in analysis["suggested_charts"] if item["chart"]["chart_type"] == "Text Length by Group"]
    assert text_by_group
    assert "eta-squared" in text_by_group[0]["rationale"]


def test_dashboard_builder_starts_empty_and_adds_user_created_charts(tmp_path):
    registry = DatasetRegistry()
    data = b"category,value\na,1\nb,2\na,3\n"
    result = UploadManager().ingest(Upload(data, "builder.csv", "text/csv"), "builder.csv", "text/csv")
    metadata = registry.register("builder.csv", result)
    store = DashboardStore(tmp_path / "ui_dashboards.sqlite")
    app = AppTest.from_file("app.py")
    app.session_state["dataset_registry"] = registry
    app.session_state["selected_dataset_id"] = metadata.dataset_id
    app.session_state["dashboard_store"] = store
    app.run()
    assert not app.exception
    app.button(key=f"open_manual_tools_open_{metadata.dataset_id}").click().run()
    assert not app.exception
    assert any("Your dashboard is empty. Add your first chart." in item.value for item in app.info)

    dashboard_id = app.session_state["active_dashboard_id"]
    app.button(key=f"add_dashboard_chart_{dashboard_id}").click().run()
    assert not app.exception
    editor_id = app.session_state["dashboard_editor"]["editor_id"]
    app.button(key=f"builder_save_chart_{editor_id}").click().run()
    assert not app.exception
    dashboard = store.get_dashboard(dashboard_id)
    assert len(dashboard["charts"]) == 1
    assert dashboard["charts"][0]["dataset_id"] == metadata.dataset_id

    assert "builder" in dashboard["charts"][0]["dataset_name"]
