from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


DASHBOARD_CHART_TYPES = [
    "Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Box Plot", "Violin Plot",
    "Pie Chart", "Donut Chart", "Area Chart", "Stacked Bar Chart", "Grouped Bar Chart",
    "Heatmap", "Count Plot", "Bubble Chart", "Treemap", "Text Length", "Text Length by Group",
]
AGGREGATIONS = {"Sum": "sum", "Mean": "mean", "Median": "median", "Count": "count",
                "Min": "min", "Max": "max", "Standard Deviation": "std"}
MAX_CHART_POINTS = 10000


class DashboardChartError(ValueError):
    """Invalid dashboard chart configuration or incompatible data."""


def _numeric(dataframe: pd.DataFrame, column: Any) -> bool:
    return column in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe[column]) and not pd.api.types.is_bool_dtype(dataframe[column])


def _categorical(dataframe: pd.DataFrame, column: Any) -> bool:
    return column in dataframe.columns and not _numeric(dataframe, column) and not pd.api.types.is_datetime64_any_dtype(dataframe[column])


def _text_column(dataframe: pd.DataFrame, column: Any) -> bool:
    if column not in dataframe.columns or not (pd.api.types.is_object_dtype(dataframe[column]) or pd.api.types.is_string_dtype(dataframe[column])):
        return False
    values = dataframe[column].dropna().astype(str)
    return bool(len(values) and values.str.len().mean() >= 12 and values.str.contains(r"\s", regex=True).mean() >= .3)


def validate_chart_config(dataframe: pd.DataFrame, config: dict[str, Any]) -> str | None:
    chart = config.get("chart_type")
    if chart not in DASHBOARD_CHART_TYPES:
        return "Choose a supported chart type."
    x, ys = config.get("x"), list(config.get("y_columns") or [])
    missing = [column for column in [x, *ys, config.get("group_by"), config.get("color_column"), config.get("size_column")] if column is not None and column not in dataframe.columns]
    if missing:
        return f"Required column(s) are unavailable in the selected dataset: {', '.join(map(str, dict.fromkeys(missing)))}. Edit this chart to choose available columns."
    if chart in {"Text Length", "Text Length by Group"}:
        if x is None or not _text_column(dataframe, x):
            return f"{chart} requires a text column with prose-like values."
        if chart == "Text Length by Group":
            group = config.get("group_by")
            if group is None or not _categorical(dataframe, group):
                return "Text Length by Group requires a categorical Group By column."
        elif config.get("group_by") is not None and not _categorical(dataframe, config["group_by"]):
            return "Text Length color grouping requires a categorical column."
        return None
    numeric_charts = {"Histogram", "Scatter Plot", "Line Chart", "Area Chart", "Bubble Chart"}
    cat_x_charts = {"Bar Chart", "Stacked Bar Chart", "Grouped Bar Chart", "Pie Chart", "Donut Chart", "Treemap", "Count Plot"}
    if chart in numeric_charts and x is None:
        return "Select an X column."
    if chart in cat_x_charts and x is None:
        return "Select a category column for the X-axis."
    if chart in cat_x_charts and chart != "Count Plot" and x is not None and not _categorical(dataframe, x):
        return "This chart needs a categorical X column. Choose a text/category column; numeric values are not silently encoded."
    if chart in {"Histogram", "Scatter Plot", "Bubble Chart"} and x is not None and not _numeric(dataframe, x):
        return f"{chart} requires a numeric X column. Choose a numerical column."
    if chart in {"Scatter Plot", "Bubble Chart", "Line Chart", "Area Chart", "Box Plot", "Violin Plot"} and not ys:
        return f"Select at least one numerical Y column for the {chart.lower()}."
    invalid_y = [column for column in ys if not _numeric(dataframe, column)]
    if invalid_y and chart not in {"Bar Chart", "Stacked Bar Chart", "Grouped Bar Chart", "Pie Chart", "Donut Chart", "Treemap", "Count Plot"}:
        return f"Y column(s) must be numerical for this chart: {', '.join(map(str, invalid_y))}. Choose numeric columns."
    if chart == "Heatmap":
        if len(ys) < 2:
            return "Select at least two numerical columns for the heatmap."
        if invalid_y:
            return "Heatmaps require numerical columns only."
    if chart == "Count Plot" and not _categorical(dataframe, x):
        return "Count Plot needs a categorical X column. Choose a low-cardinality category or text column."
    if chart in {"Line Chart", "Area Chart"} and x is not None:
        is_ordered = pd.api.types.is_numeric_dtype(dataframe[x]) or pd.api.types.is_datetime64_any_dtype(dataframe[x]) or (
            isinstance(dataframe[x].dtype, pd.CategoricalDtype) and dataframe[x].dtype.ordered
        )
        if not is_ordered:
            return "Line and Area charts need a datetime, numeric, or explicitly ordered X column."
    size = config.get("size_column")
    if chart == "Bubble Chart" and size is not None and not _numeric(dataframe, size):
        return "Bubble size must be a numerical column."
    if chart == "Treemap" and config.get("group_by") is not None and config["group_by"] == x:
        return "Choose different columns for X and Group By."
    return None


def _sample(frame: pd.DataFrame, limit: int = MAX_CHART_POINTS) -> pd.DataFrame:
    if len(frame) <= limit:
        return frame
    return frame.iloc[np.linspace(0, len(frame) - 1, limit, dtype=int)]


def _filtered_frame(dataframe: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    filters = config.get("filters") or {}
    column = filters.get("column")
    if not column:
        return dataframe
    if column not in dataframe.columns:
        raise DashboardChartError(f"Saved filter column ‘{column}’ is unavailable. Edit this chart to update its filter.")
    series = dataframe[column]
    kind = filters.get("kind")
    if kind == "categorical":
        return dataframe[series.isin(filters.get("values", [])) | series.isna()]
    if kind == "numeric":
        numeric = pd.to_numeric(series, errors="coerce")
        return dataframe[numeric.between(filters.get("min"), filters.get("max")) | numeric.isna()]
    if kind == "text":
        term = filters.get("contains", "")
        return dataframe[series.astype("string").str.contains(term, case=False, na=False, regex=False)] if term else dataframe
    if kind == "date":
        dates = pd.to_datetime(series, errors="coerce")
        low = pd.Timestamp(filters.get("from"))
        high = pd.Timestamp(filters.get("to")) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        return dataframe[dates.between(low, high) | dates.isna()]
    return dataframe


def build_dashboard_chart(dataframe: pd.DataFrame, config: dict[str, Any]) -> go.Figure:
    message = validate_chart_config(dataframe, config)
    if message:
        raise DashboardChartError(message)
    frame = _filtered_frame(dataframe, config)
    if frame.empty:
        raise DashboardChartError("No rows match this chart’s saved filter.")
    chart = config["chart_type"]
    x = config.get("x")
    ys = list(config.get("y_columns") or [])
    group = config.get("group_by")
    if chart in {"Box Plot", "Violin Plot"} and group is None and _categorical(frame, x):
        group = x
    color = config.get("color_column") or group
    title = config.get("title") or chart
    aggregation = AGGREGATIONS.get(config.get("aggregation", "Mean"), "mean")
    top_n = max(1, int(config.get("top_n", 20)))
    sort_order = config.get("sort_order", "Original")

    if chart == "Text Length":
        length = frame[x].fillna("").astype("string").str.len()
        work = pd.DataFrame({"Characters": length})
        group = config.get("group_by") or config.get("color_column")
        if group:
            work[group] = frame[group].astype("string").fillna("(missing)")
        return px.histogram(work, x="Characters", color=group, nbins=int(config.get("bins", 30)),
                            title=title, labels={"Characters": f"Character count · {x}"})
    if chart == "Text Length by Group":
        work = pd.DataFrame({"Group": frame[config["group_by"]].astype("string").fillna("(missing)"),
                             "Characters": frame[x].fillna("").astype("string").str.len()})
        return px.box(work, x="Group", y="Characters", color="Group", points="outliers", title=title,
                      labels={"Group": config["group_by"], "Characters": f"Character count · {x}"})

    if chart in {"Histogram", "Box Plot", "Violin Plot", "Scatter Plot", "Bubble Chart", "Line Chart", "Area Chart"}:
        plot_columns = list(dict.fromkeys(column for column in [x, *ys, group, color, config.get("size_column")] if column is not None))
        work = _sample(frame[plot_columns].copy())
        if chart == "Histogram":
            return px.histogram(work, x=x, nbins=int(config.get("bins", 30)), color=color, title=title)
        if chart in {"Box Plot", "Violin Plot"}:
            y = ys[0]
            if chart == "Box Plot":
                return px.box(work, x=group, y=y, color=color, points="outliers", title=title)
            return px.violin(work, x=group, y=y, color=color, box=True, points="outliers", title=title)
        if chart == "Scatter Plot":
            return px.scatter(work, x=x, y=ys[0], color=color, symbol=group if group and group != color else None, title=title)
        if chart == "Bubble Chart":
            return px.scatter(work, x=x, y=ys[0], color=color, symbol=group if group and group != color else None,
                              size=config.get("size_column"), title=title, size_max=36)
        if chart == "Line Chart":
            return px.line(work.sort_values(x), x=x, y=ys, color=color,
                           line_group=group if group and group != color else None, title=title)
        if chart == "Area Chart":
            return px.area(work.sort_values(x), x=x, y=ys, color=color,
                           line_group=group if group and group != color else None, title=title)

    if chart in {"Bar Chart", "Stacked Bar Chart", "Grouped Bar Chart", "Count Plot", "Pie Chart", "Donut Chart", "Treemap"}:
        selected = list(dict.fromkeys([x, *ys] + [item for item in [group, color] if item and item not in {x, *ys}]))
        work = frame[selected].copy()
        valid_ys = [column for column in ys if _numeric(work, column)]
        category_ys = [column for column in ys if not _numeric(work, column)]
        if chart == "Count Plot" or not valid_ys:
            keys = list(dict.fromkeys([x] + [item for item in [group, color] if item and item != x] + [item for item in category_ys if item != x and item not in {group, color}]))
            aggregated = work.groupby(keys, dropna=False, observed=False).size().reset_index(name="Count")
            value_column = "Count"
        else:
            measure = valid_ys[0]
            group_columns = list(dict.fromkeys([x] + [item for item in [group, color] if item and item != x]))
            aggregated = work.groupby(group_columns, dropna=False, observed=False)[measure].agg(aggregation).reset_index()
            value_column = measure
        if chart in {"Pie Chart", "Donut Chart"}:
            if value_column != "Count" and (aggregated[value_column] < 0).any():
                raise DashboardChartError("Pie and Donut charts require non-negative values. Choose Count aggregation or another chart type.")
            aggregated = aggregated.sort_values(value_column, ascending=False).head(top_n)
            pie_name = x
            if category_ys:
                pie_name = "__dashboard_category"
                aggregated[pie_name] = aggregated[x].astype(str) + " · " + aggregated[category_ys[0]].astype(str)
            return px.pie(aggregated, names=pie_name, values=value_column, color=color, hole=0.48 if chart == "Donut Chart" else 0, title=title)
        if chart == "Treemap":
            path = ([group] if group and group != x else []) + [item for item in category_ys if item not in {x, group}] + [x]
            return px.treemap(aggregated, path=path, values=value_column, title=title)
        if sort_order != "Original":
            category_order = aggregated.groupby(x, dropna=False)[value_column].sum().sort_values(ascending=sort_order == "Ascending").index
            aggregated = aggregated[aggregated[x].isin(category_order[:top_n])]
            aggregated = aggregated.sort_values(value_column, ascending=sort_order == "Ascending")
        else:
            category_counts = aggregated.groupby(x, dropna=False)[value_column].sum().sort_values(ascending=False).head(top_n).index
            aggregated = aggregated[aggregated[x].isin(category_counts)]
        color = color if color in aggregated.columns else (group if group in aggregated.columns else (category_ys[0] if category_ys and category_ys[0] in aggregated.columns else None))
        mode = "stack" if chart == "Stacked Bar Chart" else "group"
        return px.bar(aggregated, x=x, y=value_column, color=color, barmode=mode, title=title)

    if chart == "Heatmap":
        return px.imshow(_sample(frame[ys]).corr(numeric_only=True), text_auto=".2f", zmin=-1, zmax=1,
                         color_continuous_scale="RdBu_r", title=title)
    raise DashboardChartError(f"The {chart} chart is not implemented.")
