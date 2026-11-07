from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import streamlit as st

from src.dataset.dataset_registry import DatasetRegistry
from src.visualization.chart_builders import (
    CHART_TYPES, MAX_CORRELATION_COLUMNS, MAX_PAIR_COLUMNS, ChartConfigurationError,
    build_chart, numeric_columns,
)
from src.visualization.transformation_service import prepare_visualization

logger = logging.getLogger(__name__)

CHART_CATEGORIES = {
    "Histogram": "Distribution Analysis", "Box Plot": "Distribution Analysis",
    "Violin Plot": "Distribution Analysis", "Density Plot": "Distribution Analysis",
    "ECDF Plot": "Distribution Analysis", "Bar Chart": "Categorical Analysis",
    "Count Plot": "Categorical Analysis", "Pie Chart": "Categorical Analysis",
    "Stacked Bar Chart": "Categorical Analysis", "Scatter Plot": "Relationship Analysis",
    "Bubble Chart": "Relationship Analysis", "Correlation Heatmap": "Relationship Analysis",
    "Pair Plot": "Relationship Analysis", "Grouped Box Plot": "Relationship Analysis",
    "Line Chart": "Time Series Analysis", "Rolling Average Plot": "Time Series Analysis",
    "Seasonal Decomposition": "Time Series Analysis", "Missing Values Bar Chart": "Data Quality Analysis",
    "Missing Values Heatmap": "Data Quality Analysis", "Outlier Analysis Plot": "Data Quality Analysis",
}


def _sync_current_dataset() -> None:
    selected = st.session_state.get("eda_dataset_id")
    if selected:
        st.session_state.selected_dataset_id = selected


def _column_selector(label: str, columns: list[Any], key: str, *, optional: bool = False, default: Any = None):
    options = ([None] if optional else []) + columns
    if not options:
        st.selectbox(label, [None], format_func=lambda _: "No columns available", disabled=True, key=key)
        return None
    return st.selectbox(label, options, index=options.index(default) if default in options else 0,
                        format_func=lambda column: "None" if column is None else str(column), key=key)


def _is_multi_series(chart_type: str) -> bool:
    return chart_type in {
        "Histogram", "Box Plot", "Violin Plot", "Density Plot", "ECDF Plot", "Bar Chart",
        "Stacked Bar Chart", "Scatter Plot", "Bubble Chart", "Line Chart", "Rolling Average Plot",
        "Grouped Box Plot", "Outlier Analysis Plot",
    }


def render_visualization_explorer(registry: DatasetRegistry, current_dataset_id: str | None) -> None:
    st.divider()
    st.subheader("Explore Data")
    st.caption("Choose a chart and columns. Plotly charts update automatically when you change a selection.")
    datasets = registry.list_datasets()
    if not datasets:
        st.info("Load a dataset to explore its distributions, relationships, time series, and data quality.")
        return
    dataset_ids = [item.dataset_id for item in datasets]
    if current_dataset_id not in dataset_ids:
        current_dataset_id = dataset_ids[0]
    if st.session_state.get("eda_dataset_id") not in dataset_ids or st.session_state.get("eda_dataset_id") != current_dataset_id:
        st.session_state.eda_dataset_id = current_dataset_id
    st.markdown("**Step 1: Select Dataset**")
    dataset_id = st.selectbox(
        "Select Dataset", dataset_ids, key="eda_dataset_id",
        format_func=lambda value: next(f"{item.filename} · {len(registry.get_dataframe(item.dataset_id)):,} rows" for item in datasets if item.dataset_id == value),
        on_change=_sync_current_dataset, label_visibility="collapsed",
    )
    dataframe = registry.get_dataframe(dataset_id)
    columns = list(dataframe.columns)
    nums = numeric_columns(dataframe)
    if not columns:
        st.warning("This dataset has no columns to visualize.")
        return

    st.markdown("**Step 2: Select Visualization Type**")
    chart_type = st.selectbox("Visualization Type", CHART_TYPES,
                              format_func=lambda value: f"{CHART_CATEGORIES[value]} · {value}",
                              key=f"eda_chart_type_{dataset_id}")
    st.markdown("**Step 3: Select X-axis / Primary Column**")
    default_x = nums[0] if nums else columns[0]
    x = _column_selector("Select X-axis / Primary Column", columns, f"eda_x_{dataset_id}", optional=True, default=default_x)
    st.markdown("**Step 4: Select one or more Y columns**")
    y_options = [column for column in columns if column != x]
    default_y = [column for column in nums if column != x][:2]
    y_key = f"eda_y_columns_{dataset_id}"
    # Remove a newly selected X from the existing widget state before constructing the widget.
    if y_key in st.session_state:
        st.session_state[y_key] = [column for column in st.session_state[y_key] if column in y_options]
    y_columns = st.multiselect("Select Y columns", y_options, default=default_y, key=y_key,
                               help="Select multiple numerical measures to compare. Categorical Y values are counted where suitable.")
    st.markdown("**Step 5: Select Grouping / Color Column**")
    color = _column_selector("Select Grouping / Color Column (optional)", columns,
                             f"eda_color_{dataset_id}", optional=True, default=None)

    config: dict[str, Any] = {"x": x, "y_columns": list(y_columns), "color": color, "aggregation": "mean"}
    transform_options: dict[str, Any] = {}
    categorical_charts = {"Bar Chart", "Count Plot", "Pie Chart", "Stacked Bar Chart"}
    if chart_type in categorical_charts and x in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe[x]):
        st.caption("Numerical X values need categories for this chart. DataPilot bins them temporarily; your dataset is unchanged.")
        transform_options["numeric_x_binning"] = st.selectbox(
            "Numerical X treatment", ["Equal-width bins", "Equal-frequency bins", "Exact numeric categories"],
            key=f"eda_binning_{dataset_id}")
        if transform_options["numeric_x_binning"] != "Exact numeric categories":
            transform_options["bins"] = st.slider("Number of bins", 2, 50, 10, key=f"eda_cat_bins_{dataset_id}")
    if chart_type == "Histogram":
        config["bins"] = st.slider("Number of bins", 5, 100, 30, key=f"eda_bins_{dataset_id}")
    if chart_type in {"Bar Chart", "Pie Chart", "Stacked Bar Chart"}:
        config["aggregation"] = st.selectbox("Aggregation", ["mean", "sum", "median", "min", "max"], key=f"eda_aggregation_{dataset_id}",
                                                help="Applied to numerical Y columns. Without Y, counts are shown.")
    if chart_type == "Bubble Chart":
        config["size"] = _column_selector("Bubble size (optional numerical column)", nums, f"eda_size_{dataset_id}", optional=True)
    if chart_type == "Correlation Heatmap":
        corr_options = list(dict.fromkeys([*nums, *columns]))
        default_corr = nums[:MAX_CORRELATION_COLUMNS]
        transform_options["correlation_columns"] = st.multiselect("Additional correlation columns", corr_options, default=default_corr,
                                                                  key=f"eda_corr_cols_{dataset_id}", max_selections=MAX_CORRELATION_COLUMNS)
        encode_columns = [column for column in transform_options["correlation_columns"] if column not in nums]
        if encode_columns:
            transform_options["encode_columns"] = encode_columns
            transform_options["categorical_encoding"] = st.selectbox("Categorical encoding for heatmap", ["one-hot", "frequency", "label"],
                                                                       format_func=lambda v: {"one-hot": "One-hot", "frequency": "Frequency", "label": "Label (explicit)"}[v],
                                                                       key=f"eda_corr_encoding_{dataset_id}")
    if chart_type == "Pair Plot":
        transform_options["pair_columns"] = st.multiselect("Numerical columns for pair plot", nums,
                                                             default=nums[:min(4, MAX_PAIR_COLUMNS)], max_selections=MAX_PAIR_COLUMNS,
                                                             key=f"eda_pair_cols_{dataset_id}")
    if chart_type in {"Line Chart", "Rolling Average Plot", "Seasonal Decomposition"}:
        config["aggregation"] = st.selectbox("Combine repeated X values using", ["mean", "sum", "median"], key=f"eda_time_aggregation_{dataset_id}")
        if chart_type == "Rolling Average Plot":
            max_window = max(2, min(60, len(dataframe)))
            config["rolling_window"] = st.slider("Rolling window (observations)", 2, max_window, min(7, max_window), key=f"eda_rolling_window_{dataset_id}")
        elif chart_type == "Seasonal Decomposition":
            config["seasonal_period"] = st.slider("Observations per season", 2, 24, 12, key=f"eda_seasonal_period_{dataset_id}")
            config["seasonal_model"] = st.selectbox("Decomposition model", ["additive", "multiplicative"], key=f"eda_seasonal_model_{dataset_id}")
    if chart_type == "Outlier Analysis Plot":
        config["outlier_method"] = st.selectbox("Outlier rule", ["IQR", "Z-score"], key=f"eda_outlier_method_{dataset_id}")
    if len(y_columns) > 1 and chart_type in {"Line Chart", "Rolling Average Plot", "Scatter Plot", "Bubble Chart"}:
        transform_options["normalization"] = st.selectbox("Y-series scale", ["None", "Z-score", "Min-max"], key=f"eda_normalization_{dataset_id}",
                                                           help="Normalize selected numerical series temporarily to compare different scales.")
    if chart_type == "Scatter Plot" and y_columns:
        transform_options["categorical_encoding"] = st.selectbox("Categorical Y treatment", ["Automatic one-hot", "One-hot", "None"], key=f"eda_scatter_encoding_{dataset_id}")
        transform_options["categorical_encoding"] = {"Automatic one-hot": "Automatic", "One-hot": "One-hot", "None": "None"}[transform_options["categorical_encoding"]]

    prepared = prepare_visualization(dataframe, chart_type, x, list(y_columns), color=color, **transform_options)
    for notice in prepared.notices:
        st.info(notice)
    if prepared.transformations and prepared.preview is not None:
        with st.expander("Preview temporary visualization data"):
            st.dataframe(prepared.preview, use_container_width=True)
    if prepared.error:
        st.warning(prepared.error)
        return

    config.update({"x": prepared.x, "y_columns": prepared.y_columns, "color": prepared.color,
                   "correlation_columns": prepared.correlation_columns, "pair_columns": prepared.pair_columns})
    try:
        figure = build_chart(prepared.dataframe, chart_type, **config)
        st.plotly_chart(figure, use_container_width=True, key=f"eda_plot_{dataset_id}")
    except ChartConfigurationError as exc:
        st.warning(str(exc))
    except Exception:
        logger.exception("Could not build %s chart for dataset %s", chart_type, dataset_id)
        st.error("DataPilot could not build this chart from the selected columns. Try a different chart configuration.")
