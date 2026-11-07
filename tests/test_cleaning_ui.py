from streamlit.testing.v1 import AppTest

from src.dashboard.dashboard_store import DashboardStore
from src.dataset.dataset_registry import DatasetRegistry
from src.upload.upload_manager import UploadManager
from tests.helpers import Upload


def test_apply_button_is_required_and_updates_the_current_dataset(tmp_path):
    registry = DatasetRegistry()
    result = UploadManager().ingest(
        Upload(b"a,b\n1,x\n,missing\n2,z\n", "clean.csv", "text/csv"),
        "clean.csv",
        "text/csv",
    )
    metadata = registry.register("clean.csv", result)
    original = registry.get_original_dataframe(metadata.dataset_id)

    app = AppTest.from_file("app.py")
    app.session_state["dataset_registry"] = registry
    app.session_state["selected_dataset_id"] = metadata.dataset_id
    app.session_state["dashboard_store"] = DashboardStore(tmp_path / "dashboards.sqlite")
    app.run()
    app.button(key=f"open_manual_tools_open_{metadata.dataset_id}").click().run()

    assert not app.exception
    assert len(registry.get_dataframe(metadata.dataset_id)) == 3
    assert any(item.label == "Select Cleaning Operation" for item in app.selectbox)
    apply_button = next(button for button in app.button if button.label == "Apply cleaning operation")
    apply_button.click().run()

    assert not app.exception
    assert len(registry.get_dataframe(metadata.dataset_id)) == 2
    assert len(original) == 3
    assert any("removed 1 rows" in message.value for message in app.success)
