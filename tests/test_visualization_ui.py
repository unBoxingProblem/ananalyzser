import json

from streamlit.testing.v1 import AppTest

from src.dashboard.dashboard_store import DashboardStore
from src.dataset.dataset_registry import DatasetRegistry
from src.upload.upload_manager import UploadManager
from tests.helpers import Upload


def _eda_spec(app, dataset_id):
    suffix = f"eda_plot_{dataset_id}"
    element = next(item for item in app.get("plotly_chart") if suffix in item.proto.id)
    return json.loads(element.proto.spec)


def test_eda_controls_and_plot_render_for_the_selected_dataset(tmp_path):
    registry = DatasetRegistry()
    result = UploadManager().ingest(
        Upload(
            b"category,sales,visits\nA,10,2\nB,20,4\nA,30,6\n",
            "eda.csv",
            "text/csv",
        ),
        "eda.csv",
        "text/csv",
    )
    metadata = registry.register("eda.csv", result)

    app = AppTest.from_file("app.py")
    app.session_state["dataset_registry"] = registry
    app.session_state["selected_dataset_id"] = metadata.dataset_id
    app.session_state["dashboard_store"] = DashboardStore(tmp_path / "dashboards.sqlite")
    app.run()
    app.button(key=f"open_manual_tools_open_{metadata.dataset_id}").click().run()

    assert not app.exception
    labels = [select.label for select in app.selectbox]
    assert "Select Dataset" in labels
    assert "Select X-axis / Primary Column" in labels
    assert "Select Y columns" in [item.label for item in app.multiselect]
    assert "Select Grouping / Color Column (optional)" in labels
    assert "Visualization Type" in labels
    assert len(app.get("plotly_chart")) >= 1
    assert any(f"eda_plot_{metadata.dataset_id}" in item.proto.id for item in app.get("plotly_chart"))


def test_changing_visualization_widgets_rebuilds_chart_without_generate_button(tmp_path):
    registry = DatasetRegistry()
    result = UploadManager().ingest(
        Upload(b"region,sales,visits,profit\nNorth,10,2,3\nSouth,20,4,5\nNorth,30,6,7\n", "reactive.csv", "text/csv"),
        "reactive.csv", "text/csv",
    )
    metadata = registry.register("reactive.csv", result)
    app = AppTest.from_file("app.py")
    app.session_state["dataset_registry"] = registry
    app.session_state["selected_dataset_id"] = metadata.dataset_id
    app.session_state["dashboard_store"] = DashboardStore(tmp_path / "dashboards.sqlite")
    app.run()
    app.button(key=f"open_manual_tools_open_{metadata.dataset_id}").click().run()
    assert not app.exception
    assert not any("generate" in button.label.lower() for button in app.get("button"))

    initial_spec = _eda_spec(app, metadata.dataset_id)
    assert initial_spec["data"][0]["type"] == "histogram"
    app.selectbox(key=f"eda_chart_type_{metadata.dataset_id}").set_value("Scatter Plot").run()
    initial_spec = _eda_spec(app, metadata.dataset_id)
    assert initial_spec["data"][0]["type"] == "scatter"
    initial_x = initial_spec["data"][0]["x"]
    app.selectbox(key=f"eda_x_{metadata.dataset_id}").set_value("visits").run()
    assert not app.exception
    updated_spec = _eda_spec(app, metadata.dataset_id)
    assert updated_spec["data"][0]["x"] != initial_x

    app.multiselect(key=f"eda_y_columns_{metadata.dataset_id}").set_value(["sales", "profit"]).run()
    assert not app.exception
    assert len(_eda_spec(app, metadata.dataset_id)["data"]) == 2

    app.selectbox(key=f"eda_color_{metadata.dataset_id}").set_value("region").run()
    assert not app.exception
    grouped_spec = _eda_spec(app, metadata.dataset_id)
    assert len(grouped_spec["data"]) > 2
