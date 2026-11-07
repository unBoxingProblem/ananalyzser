from __future__ import annotations

import hashlib
import logging

import pandas as pd
import streamlit as st

from src.dataset.dataset_registry import DatasetRegistry
from src.ui.cleaning_page import render_cleaning_page
from src.ui.dashboard_page import render_dashboard
from src.ui.ai_analyst_page import render_ai_analyst
from src.ui.visualization_page import render_visualization_explorer
from src.upload.errors import UploadError
from src.upload.upload_manager import UploadManager
from src.upload.validator import configured_max_upload_mb

logger = logging.getLogger(__name__)
PREVIEW_ROW_LIMIT = 100


def _format_bytes(size: int) -> str:
    if size >= 1024**3:
        return f"{size / 1024**3:.2f} GB"
    if size >= 1024**2:
        return f"{size / 1024**2:.2f} MB"
    if size >= 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size} bytes"


def _file_key(file, index: int) -> str:
    file.seek(0)
    prefix = file.read(2048)
    file.seek(0)
    file_id = getattr(file, "file_id", "")
    identity = f"{file_id}|{index}|{file.name}|{getattr(file, 'size', 0)}|".encode() + prefix
    return hashlib.sha1(identity).hexdigest()[:16]


def _get_registry() -> DatasetRegistry:
    if "dataset_registry" not in st.session_state:
        st.session_state.dataset_registry = DatasetRegistry()
    return st.session_state.dataset_registry


def _render_uploads(registry: DatasetRegistry) -> None:
    st.subheader("Upload data files")
    max_mb = configured_max_upload_mb()
    st.caption(f"Supported: CSV, TSV, XLSX, JSON, Parquet · Up to {max_mb} MB per file. Uploads stay in this browser session.")
    uploads = st.file_uploader(
        "Drag files here or browse your device",
        type=["csv", "tsv", "xlsx", "json", "parquet"],
        accept_multiple_files=True,
        help=f"Maximum {max_mb} MB for each file. Files are parsed in memory and are not permanently stored.",
        key="datapilot_uploads",
    )
    if not uploads:
        return
    manager = UploadManager()
    cache = st.session_state.setdefault("upload_inspections", {})
    loaded = st.session_state.setdefault("loaded_upload_keys", {})
    current_keys = set()

    for index, upload in enumerate(uploads):
        key = _file_key(upload, index)
        current_keys.add(key)
        st.markdown(f"##### {upload.name}")
        columns = st.columns([1, 2, 2, 2])
        columns[0].metric("File size", _format_bytes(getattr(upload, "size", 0)))
        inspection = cache.get(key)
        if inspection is None:
            try:
                detected, size, sheet_names = manager.inspect(upload, upload.name, getattr(upload, "type", None))
                inspection = {"format": detected.format, "size": size, "sheets": sheet_names}
                cache[key] = inspection
            except UploadError as exc:
                inspection = {"error": str(exc)}
                cache[key] = inspection
                reported = st.session_state.setdefault("reported_inspection_errors", set())
                if key not in reported:
                    registry.add_error(upload.name, str(exc))
                    reported.add(key)
        if inspection.get("error"):
            columns[1].error(inspection["error"])
            if loaded.get(key) in {meta.dataset_id for meta in registry.list_datasets()}:
                st.success(f"Loaded as dataset `{loaded[key]}`.")
            continue

        fmt = inspection["format"]
        columns[1].caption(f"Detected format: **{fmt.upper()}**")
        columns[2].caption(f"Upload status: **Ready to load**")
        options: dict[str, object] = {}
        if fmt == "xlsx":
            sheets = inspection["sheets"]
            if not sheets:
                st.error("This workbook does not contain any worksheets.")
                continue
            options["sheet_name"] = st.selectbox(
                "Choose a worksheet",
                sheets,
                key=f"sheet_{key}",
            )
        elif fmt in {"csv", "tsv"}:
            options["delimiter"] = st.selectbox(
                "Delimiter",
                ["auto", "comma (,)", "semicolon (;)", "tab (\\t)"],
                index=0 if fmt == "csv" else 3,
                key=f"delimiter_{key}",
            )

        already_loaded = loaded.get(key) in {meta.dataset_id for meta in registry.list_datasets()}
        if already_loaded:
            st.success(f"Successfully ingested as dataset `{loaded[key]}`.")
            if st.button("Load this file again", key=f"reload_{key}"):
                loaded.pop(key, None)
                st.rerun()
        elif st.button("Load dataset", type="primary", key=f"load_{key}"):
            try:
                with st.spinner(f"Validating and loading {upload.name}…"):
                    result = manager.ingest(upload, upload.name, getattr(upload, "type", None), **options)
                    metadata = registry.register(upload.name, result)
                    loaded[key] = metadata.dataset_id
                    st.session_state.selected_dataset_id = metadata.dataset_id
                st.success(f"Successfully ingested {upload.name} as dataset {metadata.dataset_id}.")
                st.rerun()
            except UploadError as exc:
                registry.add_error(upload.name, str(exc))
                logger.info("Upload rejected for %s: %s", upload.name, exc)
                st.error(str(exc))
            except Exception as exc:
                registry.add_error(upload.name, "Unexpected ingestion error. Check the file and try again.")
                logger.exception("Unexpected ingestion error for %s", upload.name)
                st.error("DataPilot could not load this file. Check that it is valid and try again.")

    for stale_key in set(loaded) - current_keys:
        loaded.pop(stale_key, None)
        cache.pop(stale_key, None)


def _render_dataset_preview(registry: DatasetRegistry) -> str | None:
    st.divider()
    st.subheader("Your datasets")
    datasets = registry.list_datasets()
    if not datasets:
        st.info("No datasets loaded yet. Upload a supported file above to get started.")
        return None

    dataset_ids = [item.dataset_id for item in datasets]
    if st.session_state.get("selected_dataset_id") not in dataset_ids:
        st.session_state.selected_dataset_id = dataset_ids[0]
    selected_id = st.selectbox(
        "Select a dataset",
        dataset_ids,
        key="selected_dataset_id",
        format_func=lambda dataset_id: next(
            f"{item.filename} · {item.file_format.upper()} · {item.row_count:,} rows"
            for item in datasets if item.dataset_id == dataset_id
        ),
    )
    metadata = registry.get_metadata(selected_id)
    dataframe = registry.get_dataframe(selected_id)
    is_cleaned = registry.has_cleaning_changes(selected_id)

    st.markdown(f"### {metadata.filename}")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Format", metadata.file_format.upper())
    metric_cols[1].metric("Rows", f"{len(dataframe):,}")
    metric_cols[2].metric("Columns", f"{len(dataframe.columns):,}")
    metric_cols[3].metric("File size", _format_bytes(metadata.file_size_bytes))
    if is_cleaned:
        st.info("Displayed data: cleaned working copy. The original uploaded dataset is preserved; use Reset Cleaning to restore it.")
    else:
        st.caption("Displayed data: original uploaded dataset.")
    st.caption(f"Status: **{metadata.status.title()}** · Dataset ID: `{metadata.dataset_id}`")

    if metadata.warnings:
        for warning in metadata.warnings:
            st.warning(warning)

    dtypes = pd.DataFrame({
        "Column": metadata.columns,
        "Data type": [str(dataframe.dtypes.iloc[index]) for index in range(len(metadata.columns))],
        "Missing values": [int(dataframe.iloc[:, index].isna().sum()) for index in range(len(metadata.columns))],
    })
    left, right = st.columns([3, 1])
    with left:
        st.markdown("#### Columns and data types")
        st.dataframe(dtypes, use_container_width=True, hide_index=True)
    with right:
        st.markdown("#### Column names")
        st.write(metadata.columns)

    st.markdown(f"#### Preview · first {min(PREVIEW_ROW_LIMIT, len(dataframe)):,} rows")
    st.dataframe(dataframe.head(PREVIEW_ROW_LIMIT), use_container_width=True, hide_index=True)
    if dataframe.empty:
        st.info("This dataset has headers but no data rows.")

    if st.button("Remove dataset from this session", key=f"remove_{selected_id}"):
        registry.remove(selected_id)
        for file_key, dataset_id in list(st.session_state.get("loaded_upload_keys", {}).items()):
            if dataset_id == selected_id:
                st.session_state.loaded_upload_keys.pop(file_key, None)
        st.session_state.pop("selected_dataset_id", None)
        st.rerun()

    return selected_id


def _render_ingestion_errors(registry: DatasetRegistry) -> None:
    if not registry.errors:
        return
    with st.expander(f"Recent ingestion errors ({len(registry.errors)})"):
        for error in reversed(registry.errors[-10:]):
            st.error(f"{error.filename} · {error.error_message}")


def render_upload_page() -> None:
    st.set_page_config(page_title="DataPilot | Data ingestion", page_icon="🧭", layout="wide")
    st.markdown("""
    <style>
    .block-container {max-width: 1380px; padding-top: 2rem; padding-bottom: 3rem;}
    .dp-hero {padding: 1.2rem 1.5rem; border-radius: 16px; background: linear-gradient(115deg,#12243a,#1f5268); color: white; margin-bottom: 1.2rem;}
    .dp-hero h1 {margin: 0 0 .35rem 0; font-size: 2rem;}
    .dp-hero p {margin: 0; color: #d1e0ea;}
    </style>
    <div class="dp-hero"><h1>🧭 DataPilot</h1><p>Upload a file, chat with the AI Analyst to get insights, then clean and explore the data yourself if you want.</p></div>
    """, unsafe_allow_html=True)
    registry = _get_registry()
    _render_uploads(registry)
    _render_ingestion_errors(registry)
    selected_id = _render_dataset_preview(registry)
    render_ai_analyst(registry, selected_id)
    if selected_id is not None:
        manual_key = f"manual_tools_open_{selected_id}"
        if not st.session_state.get(manual_key, False):
            if st.button("Open manual data tools", key=f"open_{manual_key}"):
                st.session_state[manual_key] = True
                st.rerun()
            st.caption("Data Cleaning, Explore Data, and Dashboard Builder are hidden until you choose to open them.")
        else:
            st.divider()
            st.subheader("Manual data tools")
            st.caption("Clean data, explore charts, or edit saved dashboards here.")
            if st.button("Hide manual data tools", key=f"close_{manual_key}"):
                st.session_state[manual_key] = False
                st.rerun()
            render_cleaning_page(registry, selected_id)
            render_visualization_explorer(registry, selected_id)
            render_dashboard(registry, selected_id)
