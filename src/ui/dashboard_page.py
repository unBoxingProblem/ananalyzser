from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

import pandas as pd
import streamlit as st

from src.dashboard.chart_service import (
    AGGREGATIONS, DASHBOARD_CHART_TYPES, DashboardChartError,
    build_dashboard_chart, validate_chart_config,
)
from src.dashboard.dashboard_store import DashboardStore
from src.dataset.dataset_registry import DatasetRegistry

logger = logging.getLogger(__name__)


def _get_store() -> DashboardStore:
    if "dashboard_store" not in st.session_state:
        st.session_state.dashboard_store = DashboardStore()
    return st.session_state.dashboard_store


def _numeric_columns(dataframe: pd.DataFrame) -> list[Any]:
    return [column for column in dataframe.columns if pd.api.types.is_numeric_dtype(dataframe[column]) and not pd.api.types.is_bool_dtype(dataframe[column])]


def _categorical_columns(dataframe: pd.DataFrame) -> list[Any]:
    return [column for column in dataframe.columns if not pd.api.types.is_numeric_dtype(dataframe[column]) and not pd.api.types.is_datetime64_any_dtype(dataframe[column])]


def _text_columns(dataframe: pd.DataFrame) -> list[Any]:
    return [column for column in _categorical_columns(dataframe)
            if dataframe[column].dropna().astype(str).str.len().mean() >= 12]


def _default_config(dataset_id: str, filename: str, dataframe: pd.DataFrame) -> dict[str, Any]:
    numeric = _numeric_columns(dataframe)
    categorical = _categorical_columns(dataframe)
    return {
        "chart_id": str(uuid4()), "dataset_id": dataset_id, "dataset_name": filename,
        "chart_type": "Bar Chart", "x": categorical[0] if categorical else (dataframe.columns[0] if len(dataframe.columns) else None),
        "y_columns": [numeric[0]] if numeric else [], "group_by": None, "color_column": None,
        "aggregation": "Mean", "title": "", "sort_order": "Descending", "top_n": 20,
        "bins": 30, "size_column": None, "filters": {},
    }


def _select_optional(label: str, columns: list[Any], current: Any, key: str) -> Any:
    options = [None, *columns]
    return st.selectbox(label, options, index=options.index(current) if current in options else 0,
                        format_func=lambda item: "None" if item is None else str(item), key=key)


def _make_filters(dataframe: pd.DataFrame, columns: list[Any], editor_key: str, dataset_id: str,
                  current: dict[str, Any] | None = None) -> dict[str, Any]:
    current = current or {}
    with st.expander("Optional chart filter", expanded=False):
        selected = _select_optional("Filter column", columns, current.get("column"), f"{editor_key}_{dataset_id}_filter_column")
        if selected is None:
            return {}
        series = dataframe[selected]
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            values = pd.to_numeric(series, errors="coerce").dropna()
            if not len(values):
                return {}
            low, high = float(values.min()), float(values.max())
            if low == high:
                return {"column": selected, "kind": "numeric", "min": low, "max": high}
            if current.get("column") == selected:
                default_range = (max(low, min(high, float(current.get("min", low)))), max(low, min(high, float(current.get("max", high)))))
                default_range = tuple(sorted(default_range))
            else:
                default_range = (low, high)
            chosen = st.slider(f"{selected} range", min_value=low, max_value=high, value=default_range, key=f"{editor_key}_{dataset_id}_filter_numeric")
            return {"column": selected, "kind": "numeric", "min": float(chosen[0]), "max": float(chosen[1])}
        if pd.api.types.is_datetime64_any_dtype(series):
            values = pd.to_datetime(series, errors="coerce").dropna()
            if not len(values):
                return {}
            low, high = values.min().date(), values.max().date()
            default_range = (pd.Timestamp(current.get("from")).date(), pd.Timestamp(current.get("to")).date()) if current.get("column") == selected and current.get("from") else (low, high)
            default_range = tuple(sorted((max(low, min(high, default_range[0])), max(low, min(high, default_range[1])))))
            chosen = st.date_input(f"{selected} date range", default_range, min_value=low, max_value=high, key=f"{editor_key}_{dataset_id}_filter_date")
            if isinstance(chosen, (list, tuple)) and len(chosen) == 2:
                return {"column": selected, "kind": "date", "from": chosen[0].isoformat(), "to": chosen[1].isoformat()}
            return {}
        unique = list(pd.unique(series.dropna()))
        if len(unique) <= 100:
            default_values = current.get("values", unique) if current.get("column") == selected else unique
            chosen = st.multiselect(f"{selected} values", unique, default=[item for item in default_values if item in unique], key=f"{editor_key}_{dataset_id}_filter_categories")
            return {"column": selected, "kind": "categorical", "values": chosen}
        contains = st.text_input(f"{selected} contains", value=current.get("contains", "") if current.get("column") == selected else "", key=f"{editor_key}_{dataset_id}_filter_text")
        return {"column": selected, "kind": "text", "contains": contains}


def _render_chart_editor(registry: DatasetRegistry, datasets: list, editor: dict[str, Any]) -> None:
    st.markdown("#### Chart configuration")
    by_id = {item.dataset_id: item for item in datasets}
    dataset_ids = list(by_id)
    saved_dataset_id = (editor.get("config") or {}).get("dataset_id")
    if saved_dataset_id and saved_dataset_id not in dataset_ids:
        st.warning(f"The chart references ‘{editor.get('dataset_name', saved_dataset_id)}’, which is unavailable. Choose a dataset explicitly to retarget this chart.")
    initial_dataset = saved_dataset_id if saved_dataset_id in dataset_ids else None
    dataset_id = st.selectbox(
        "Dataset", [None, *dataset_ids], index=([None, *dataset_ids].index(initial_dataset)),
        format_func=lambda value: "Choose a dataset" if value is None else by_id[value].filename,
        key=f"builder_dataset_{editor['editor_id']}",
    )
    if dataset_id is None:
        return
    metadata = by_id[dataset_id]
    dataframe = registry.get_dataframe(dataset_id)
    columns = list(dataframe.columns)
    if not columns:
        st.warning("This dataset has no columns to chart.")
        return
    numeric = _numeric_columns(dataframe)
    config = dict(editor.get("config") or _default_config(dataset_id, metadata.filename, dataframe))
    config["dataset_id"] = dataset_id
    config["dataset_name"] = metadata.filename
    type_key = f"builder_chart_type_{editor['editor_id']}"
    chart_type = st.selectbox("Chart Type", DASHBOARD_CHART_TYPES,
                              index=DASHBOARD_CHART_TYPES.index(config["chart_type"]) if config.get("chart_type") in DASHBOARD_CHART_TYPES else 0,
                              key=type_key)
    config["chart_type"] = chart_type
    needs_x = chart_type not in {"Heatmap", "Box Plot", "Violin Plot"}
    if needs_x:
        if chart_type in {"Histogram", "Scatter Plot", "Bubble Chart"}:
            allowed_x = numeric
        elif chart_type in {"Text Length", "Text Length by Group"}:
            allowed_x = _text_columns(dataframe)
        else:
            allowed_x = columns
        current_x = config.get("x")
        if current_x not in allowed_x:
            current_x = (allowed_x[0] if allowed_x else None)
        x_options = ([None] if chart_type == "Count Plot" or not allowed_x else []) + allowed_x
        x = st.selectbox("X-axis", x_options, index=x_options.index(current_x) if current_x in x_options else 0,
                         format_func=lambda item: "None" if item is None else str(item),
                         key=f"builder_x_{editor['editor_id']}_{dataset_id}")
        config["x"] = x
    else:
        config["x"] = None

    if chart_type == "Heatmap":
        chosen_y = st.multiselect("Y columns", numeric, default=[item for item in config.get("y_columns", []) if item in numeric],
                                  key=f"builder_y_{editor['editor_id']}_{dataset_id}")
    elif chart_type in {"Histogram", "Count Plot", "Text Length", "Text Length by Group"}:
        chosen_y = []
    else:
        y_options = columns if chart_type in {"Bar Chart", "Stacked Bar Chart", "Grouped Bar Chart", "Pie Chart", "Donut Chart", "Treemap"} else numeric
        allow_optional = chart_type in {"Bar Chart", "Stacked Bar Chart", "Grouped Bar Chart", "Pie Chart", "Donut Chart", "Treemap"}
        current_y = next((item for item in config.get("y_columns", []) if item in y_options), None)
        y_selector_options = [None] + y_options
        chosen = st.selectbox("Y-axis column", y_selector_options,
                              index=y_selector_options.index(current_y) if current_y in y_selector_options else 0,
                              format_func=lambda item: "Count rows" if item is None else str(item),
                              key=f"builder_y_{editor['editor_id']}_{dataset_id}")
        chosen_y = [chosen] if chosen is not None else []
    config["y_columns"] = chosen_y

    group = _select_optional("Group By", columns, config.get("group_by"), f"builder_group_{editor['editor_id']}_{dataset_id}")
    color = _select_optional("Color column", columns, config.get("color_column"), f"builder_color_{editor['editor_id']}_{dataset_id}")
    config["group_by"], config["color_column"] = group, color

    relevant_aggregation = chart_type in {"Bar Chart", "Stacked Bar Chart", "Grouped Bar Chart", "Pie Chart", "Donut Chart", "Treemap", "Count Plot"}
    if relevant_aggregation:
        aggregations = list(AGGREGATIONS)
        config["aggregation"] = st.selectbox("Aggregation", aggregations,
                                             index=aggregations.index(config.get("aggregation", "Mean")) if config.get("aggregation", "Mean") in aggregations else 1,
                                             key=f"builder_aggregation_{editor['editor_id']}_{dataset_id}")
    config["title"] = st.text_input("Chart title", value=str(config.get("title") or ""),
                                    key=f"builder_title_{editor['editor_id']}_{dataset_id}")
    if chart_type in {"Bar Chart", "Stacked Bar Chart", "Grouped Bar Chart", "Pie Chart", "Donut Chart", "Treemap", "Count Plot"}:
        config["sort_order"] = st.selectbox("Sort order", ["Descending", "Ascending", "Original"],
                                            index=["Descending", "Ascending", "Original"].index(config.get("sort_order", "Descending")),
                                            key=f"builder_sort_{editor['editor_id']}_{dataset_id}")
        config["top_n"] = st.number_input("Top N categories", min_value=1, max_value=500, value=int(config.get("top_n", 20)),
                                          key=f"builder_topn_{editor['editor_id']}_{dataset_id}")
    if chart_type == "Histogram":
        config["bins"] = st.slider("Histogram bins", 5, 100, int(config.get("bins", 30)), key=f"builder_bins_{editor['editor_id']}_{dataset_id}")
    if chart_type == "Bubble Chart":
        config["size_column"] = _select_optional("Bubble size column", numeric, config.get("size_column"), f"builder_size_{editor['editor_id']}_{dataset_id}")
    config["filters"] = _make_filters(dataframe, columns, editor["editor_id"], dataset_id, config.get("filters"))

    warning = validate_chart_config(dataframe, config)
    if warning:
        st.warning(warning)
    else:
        try:
            figure = build_dashboard_chart(dataframe, config)
            st.plotly_chart(figure, use_container_width=True, key=f"builder_preview_{editor['editor_id']}_{dataset_id}")
        except DashboardChartError as exc:
            st.warning(str(exc))

    config["dataset_id"] = dataset_id
    config["dataset_name"] = metadata.filename
    editor["config"] = config
    button_label = "Save Chart" if editor.get("mode") == "edit" else "Add to Dashboard"
    if st.button(button_label, type="primary", key=f"builder_save_chart_{editor['editor_id']}", disabled=bool(warning)):
        store = _get_store()
        dashboard = store.get_dashboard(st.session_state.active_dashboard_id)
        if dashboard is None:
            st.error("The selected dashboard could not be loaded.")
            return
        if editor.get("mode") == "edit":
            chart_id = editor["config"].get("chart_id")
            store.update_chart(dashboard["dashboard_id"], chart_id, config)
        else:
            store.add_chart(dashboard["dashboard_id"], config)
        st.session_state.dashboard_editor = None
        st.rerun()


def render_dashboard(registry: DatasetRegistry, dataset_id: str | None) -> None:
    st.divider()
    st.subheader("Dashboard Builder")
    st.caption("Build and save a dashboard from charts you configure. Each chart keeps its own dataset and settings.")
    store = _get_store()
    dashboards = store.list_dashboards()
    if not dashboards:
        dashboards = [store.create_dashboard("My Dashboard")]
    dashboard_ids = [item["dashboard_id"] for item in dashboards]
    active_id = st.session_state.get("active_dashboard_id")
    if active_id not in dashboard_ids:
        active_id = dashboard_ids[0]
        st.session_state.active_dashboard_id = active_id
    active = store.get_dashboard(active_id)
    if active is None:
        st.error("This saved dashboard is unavailable.")
        return

    left, right = st.columns([3, 1])
    with left:
        active_id = st.selectbox("Load Dashboard", dashboard_ids, index=dashboard_ids.index(active_id),
                                 key="active_dashboard_id", format_func=lambda value: next(item["name"] for item in dashboards if item["dashboard_id"] == value))
    active = store.get_dashboard(active_id)
    if active is None:
        st.error("This saved dashboard is unavailable.")
        return
    with right:
        st.caption(f"Saved · updated {active['updated_at'][:19].replace('T', ' ')} UTC")

    name_col, save_col = st.columns([3, 2])
    with name_col:
        new_name = st.text_input("Dashboard name", value=active["name"], key=f"dashboard_name_{active_id}")
    with save_col:
        st.write("")
        save_buttons = st.columns(2)
        save_requested = save_buttons[0].button("Save Dashboard", key=f"save_dashboard_{active_id}")
        changes_requested = save_buttons[1].button("Save Changes", key=f"save_changes_{active_id}")
        if save_requested or changes_requested:
            active["name"] = new_name
            store.save_dashboard(active)
            st.success("Dashboard changes saved.")
            st.rerun()

    buttons = st.columns(4)
    if buttons[0].button("Add Chart", type="primary", key=f"add_dashboard_chart_{active_id}", disabled=not registry.list_datasets()):
        items = registry.list_datasets()
        initial_id = dataset_id if dataset_id in {item.dataset_id for item in items} else items[0].dataset_id
        metadata = registry.get_metadata(initial_id)
        config = _default_config(initial_id, metadata.filename, registry.get_dataframe(initial_id))
        st.session_state.dashboard_editor = {"editor_id": str(uuid4()), "mode": "new", "config": config}
    duplicate_name = buttons[1].text_input("Copy name", value=f"{active['name']} (copy)", key=f"copy_name_{active_id}", label_visibility="collapsed")
    if buttons[2].button("Duplicate Dashboard", key=f"duplicate_dashboard_{active_id}"):
        copied = store.duplicate_dashboard(active_id, duplicate_name)
        st.session_state.active_dashboard_id = copied["dashboard_id"]
        st.rerun()
    if buttons[3].button("Delete Dashboard", key=f"delete_dashboard_{active_id}"):
        store.delete_dashboard(active_id)
        st.session_state.active_dashboard_id = None
        st.rerun()

    create_col, create_button_col = st.columns([4, 1])
    with create_col:
        new_dashboard_name = st.text_input("New dashboard name", key="new_dashboard_name", placeholder="e.g. Weekly Review")
    with create_button_col:
        st.write("")
        if st.button("Create New Dashboard", key="create_dashboard"):
            created = store.create_dashboard(new_dashboard_name or "Untitled Dashboard")
            st.session_state.active_dashboard_id = created["dashboard_id"]
            st.rerun()

    editor = st.session_state.get("dashboard_editor")
    if editor:
        with st.container(border=True):
            _render_chart_editor(registry, registry.list_datasets(), editor)

    charts = active["charts"]
    if not charts:
        st.info("Your dashboard is empty. Add your first chart.")
    else:
        for start in range(0, len(charts), 2):
            cards = st.columns(2)
            for offset, (column, chart_config) in enumerate(zip(cards, charts[start:start + 2])):
                with column, st.container(border=True):
                    title = chart_config.get("title") or chart_config.get("chart_type", "Chart")
                    st.markdown(f"### {title}")
                    referenced_id = chart_config.get("dataset_id")
                    available_ids = {item.dataset_id for item in registry.list_datasets()}
                    if referenced_id not in available_ids:
                        st.warning(f"Dataset ‘{chart_config.get('dataset_name', referenced_id)}’ is unavailable in this session. Select the original dataset when editing; DataPilot will not substitute another dataset.")
                    else:
                        dataframe = registry.get_dataframe(referenced_id)
                        issue = validate_chart_config(dataframe, chart_config)
                        if issue:
                            st.warning(issue)
                        else:
                            try:
                                figure = build_dashboard_chart(dataframe, chart_config)
                                st.plotly_chart(figure, use_container_width=True,
                                                key=f"dashboard_saved_chart_{active_id}_{chart_config.get('chart_id', start)}")
                            except DashboardChartError as exc:
                                st.warning(str(exc))
                    st.caption(f"{chart_config.get('chart_type', 'Chart')} · {chart_config.get('dataset_name', 'Dataset')}")
                    actions = st.columns(5)
                    chart_id = chart_config.get("chart_id")
                    if actions[0].button("Edit", key=f"edit_chart_{active_id}_{chart_id}"):
                        st.session_state.dashboard_editor = {"editor_id": str(uuid4()), "mode": "edit", "config": dict(chart_config)}
                        st.rerun()
                    if actions[1].button("Duplicate", key=f"duplicate_chart_{active_id}_{chart_id}"):
                        store.duplicate_chart(active_id, chart_id)
                        st.rerun()
                    if actions[2].button("Delete", key=f"delete_chart_{active_id}_{chart_id}"):
                        store.delete_chart(active_id, chart_id)
                        st.rerun()
                    if actions[3].button("Move up", key=f"move_chart_up_{active_id}_{chart_id}", disabled=start + offset == 0):
                        store.move_chart(active_id, chart_id, -1)
                        st.rerun()
                    if actions[4].button("Move down", key=f"move_chart_down_{active_id}_{chart_id}", disabled=start + offset >= len(charts) - 1):
                        store.move_chart(active_id, chart_id, 1)
                        st.rerun()
