from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
from statsmodels.tsa.seasonal import seasonal_decompose


CHART_TYPES = [
    "Histogram",
    "Box Plot",
    "Violin Plot",
    "Density Plot",
    "ECDF Plot",
    "Bar Chart",
    "Count Plot",
    "Pie Chart",
    "Stacked Bar Chart",
    "Scatter Plot",
    "Bubble Chart",
    "Correlation Heatmap",
    "Pair Plot",
    "Grouped Box Plot",
    "Line Chart",
    "Rolling Average Plot",
    "Seasonal Decomposition",
    "Missing Values Bar Chart",
    "Missing Values Heatmap",
    "Outlier Analysis Plot",
]

MAX_SCATTER_POINTS = 5000
MAX_MATRIX_POINTS = 1500
MAX_PAIR_COLUMNS = 5
MAX_CORRELATION_COLUMNS = 30


class ChartConfigurationError(ValueError):
    """A user-facing chart configuration or data suitability problem."""


def numeric_columns(dataframe: pd.DataFrame) -> list[Any]:
    return list(dataframe.select_dtypes(include="number").columns)


def categorical_columns(dataframe: pd.DataFrame) -> list[Any]:
    numbers = set(numeric_columns(dataframe))
    return [column for column in dataframe.columns if column not in numbers and not pd.api.types.is_datetime64_any_dtype(dataframe[column])]


def _require_column(dataframe: pd.DataFrame, column: Any, label: str = "X-axis") -> Any:
    if column is None or column not in dataframe.columns:
        raise ChartConfigurationError(f"Select a valid {label} column.")
    return column


def _require_numeric(dataframe: pd.DataFrame, column: Any, label: str) -> Any:
    _require_column(dataframe, column, label)
    if not pd.api.types.is_numeric_dtype(dataframe[column]):
        raise ChartConfigurationError(f"{label} must be numerical. Choose one of: {', '.join(map(str, numeric_columns(dataframe))) or 'no numeric columns found'}.")
    return column


def _require_categorical(dataframe: pd.DataFrame, column: Any, label: str = "X-axis") -> Any:
    _require_column(dataframe, column, label)
    if pd.api.types.is_numeric_dtype(dataframe[column]):
        raise ChartConfigurationError(f"{label} must be categorical for this chart. Choose a text or category column.")
    return column


def _sample_rows(dataframe: pd.DataFrame, limit: int, *, ordered: bool = False) -> pd.DataFrame:
    if len(dataframe) <= limit:
        return dataframe
    if ordered:
        indices = np.linspace(0, len(dataframe) - 1, num=limit, dtype=int)
        return dataframe.iloc[indices]
    return dataframe.sample(n=limit, random_state=17)


def _datetime_series(series: pd.Series) -> pd.Series | None:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")
    if pd.api.types.is_numeric_dtype(series):
        return None
    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    present = int(series.notna().sum())
    if present and parsed.notna().sum() / present >= 0.8:
        return parsed
    return None


def _aggregate_categories(
    dataframe: pd.DataFrame,
    x: Any,
    y: Any | None,
    color: Any | None,
    aggregation: str,
) -> tuple[pd.DataFrame, Any]:
    group_columns = [x] + ([color] if color is not None and color != x else [])
    if y is None:
        aggregated = dataframe.groupby(group_columns, dropna=False, observed=False).size().reset_index(name="Count")
        return aggregated, "Count"
    _require_numeric(dataframe, y, "Y-axis")
    aggregated = dataframe.groupby(group_columns, dropna=False, observed=False)[y].agg(aggregation).reset_index()
    return aggregated, y


def _time_series_frame(
    dataframe: pd.DataFrame,
    x: Any,
    y: Any,
    color: Any | None,
    aggregation: str,
) -> tuple[pd.DataFrame, bool]:
    _require_column(dataframe, x, "X-axis")
    _require_numeric(dataframe, y, "Y-axis")
    parsed_dates = _datetime_series(dataframe[x])
    work_columns = [x, y] + ([color] if color is not None and color not in {x, y} else [])
    work = dataframe[work_columns].copy()
    is_datetime = parsed_dates is not None
    if parsed_dates is not None:
        work[x] = parsed_dates
        work = work.dropna(subset=[x]).sort_values(x)
    elif pd.api.types.is_numeric_dtype(dataframe[x]):
        work = work.sort_values(x)
    else:
        # For non-date labels, preserve their first-seen order in the source data.
        order = pd.unique(work[x].dropna())
        work[x] = pd.Categorical(work[x], categories=order, ordered=True)
    keys = [x] + ([color] if color is not None and color not in {x, y} else [])
    work = work.groupby(keys, dropna=False, observed=True)[y].agg(aggregation).reset_index()
    if work.empty:
        raise ChartConfigurationError("No usable X/Y values are available for this time series.")
    return work, is_datetime


def _build_correlation(dataframe: pd.DataFrame, columns: list[Any] | None) -> go.Figure:
    numeric = numeric_columns(dataframe)
    selected = list(columns) if columns else numeric
    invalid = [column for column in selected if column not in numeric]
    if invalid:
        raise ChartConfigurationError("Correlation heatmaps only accept numerical columns.")
    if len(selected) < 2:
        raise ChartConfigurationError("Select at least two numerical columns for correlation analysis.")
    if len(selected) > MAX_CORRELATION_COLUMNS:
        raise ChartConfigurationError(f"Select no more than {MAX_CORRELATION_COLUMNS} columns for this heatmap.")
    correlations = dataframe[selected].corr(numeric_only=True)
    return px.imshow(
        correlations,
        text_auto=".2f",
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title="Correlation heatmap",
        labels={"x": "Column", "y": "Column", "color": "Correlation"},
    )


def _build_missing_heatmap(dataframe: pd.DataFrame) -> go.Figure:
    if dataframe.empty or dataframe.shape[1] == 0:
        raise ChartConfigurationError("A missing-value heatmap needs at least one row and one column.")
    columns = list(dataframe.columns[:50])
    sample = _sample_rows(dataframe[columns], 1000, ordered=True)
    values = sample.isna().astype(int)
    values.index = np.arange(1, len(values) + 1)
    return px.imshow(
        values,
        aspect="auto",
        color_continuous_scale=[[0, "#dfe8ee"], [1, "#e45756"]],
        zmin=0,
        zmax=1,
        title="Missing values by row and column (1,000-row preview maximum)",
        labels={"x": "Column", "y": "Preview row", "color": "Missing"},
    )


def _build_outliers(dataframe: pd.DataFrame, column: Any, method: str, color: Any | None) -> go.Figure:
    _require_numeric(dataframe, column, "Outlier column")
    values = dataframe[column]
    valid = values.notna()
    if method == "Z-score":
        mean, std = values.mean(), values.std(ddof=0)
        if not std or np.isnan(std):
            mask = pd.Series(False, index=dataframe.index)
        else:
            mask = ((values - mean).abs() / std) > 3
        bounds = (mean - 3 * std, mean + 3 * std)
        threshold_label = "±3 standard deviations"
    else:
        q1, q3 = values.quantile([0.25, 0.75])
        spread = q3 - q1
        bounds = (q1 - 1.5 * spread, q3 + 1.5 * spread)
        mask = (values < bounds[0]) | (values > bounds[1])
        threshold_label = "1.5 × IQR"
    plot_columns = [column] + ([color] if color is not None and color in dataframe.columns else [])
    work = dataframe.loc[valid, plot_columns].copy()
    work["Row"] = np.flatnonzero(valid.to_numpy()) + 1
    work["Status"] = np.where(mask.loc[valid], "Outlier", "Within range")
    work = _sample_rows(work, MAX_SCATTER_POINTS)
    figure = px.scatter(
        work,
        x="Row",
        y=column,
        color="Status",
        hover_data=[color] if color is not None and color in work.columns else None,
        color_discrete_map={"Outlier": "#d62728", "Within range": "#2878b5"},
        title=f"Outlier analysis · {column} · {method}",
    )
    figure.add_hline(y=float(bounds[0]), line_dash="dash", annotation_text=f"Lower {threshold_label}")
    figure.add_hline(y=float(bounds[1]), line_dash="dash", annotation_text=f"Upper {threshold_label}")
    return figure


def build_chart(
    dataframe: pd.DataFrame,
    chart_type: str,
    *,
    x: Any | None = None,
    y: Any | None = None,
    color: Any | None = None,
    aggregation: str = "mean",
    bins: int = 30,
    size: Any | None = None,
    correlation_columns: list[Any] | None = None,
    pair_columns: list[Any] | None = None,
    rolling_window: int = 7,
    seasonal_period: int = 12,
    seasonal_model: str = "additive",
    outlier_method: str = "IQR",
    y_columns: list[Any] | None = None,
) -> go.Figure:
    """Build a Plotly figure from a dataset and explicit chart configuration."""
    if chart_type not in CHART_TYPES:
        raise ChartConfigurationError(f"Unknown visualization type: {chart_type}")
    if dataframe.empty and chart_type not in {"Correlation Heatmap", "Missing Values Bar Chart", "Missing Values Heatmap"}:
        raise ChartConfigurationError("This dataset has no rows to visualize.")

    selected_y = list(dict.fromkeys(column for column in (y_columns or ([y] if y is not None else [])) if column is not None))
    if y_columns is not None and selected_y:
        if chart_type in {"Histogram", "Box Plot", "Violin Plot", "Density Plot", "ECDF Plot", "Grouped Box Plot", "Outlier Analysis Plot"} and len(selected_y) == 1:
            x = selected_y[0]
        elif chart_type in {"Scatter Plot", "Bubble Chart", "Line Chart", "Rolling Average Plot", "Seasonal Decomposition"} and len(selected_y) == 1:
            y = selected_y[0]
        elif chart_type in {"Bar Chart", "Stacked Bar Chart", "Pie Chart"} and len(selected_y) == 1:
            y = selected_y[0]
    # Multi-series configurations are composed from the same validated single-series
    # builders so their aggregation, sampling, and chart-specific checks stay aligned.
    if len(selected_y) > 1:
        if chart_type in {"Seasonal Decomposition", "Pie Chart"}:
            raise ChartConfigurationError(f"{chart_type} supports one Y series at a time. Select a single Y column.")
        if chart_type in {"Bar Chart", "Stacked Bar Chart"}:
            numeric_flags = [column in numeric_columns(dataframe) for column in selected_y]
            if any(numeric_flags) and not all(numeric_flags):
                raise ChartConfigurationError("Select either numerical measures or categorical columns for a multi-series bar chart; mixed Y types have no single meaningful aggregation.")
            category_color = color is not None and color not in [x, *selected_y]
            long = dataframe[[x, *selected_y] + ([color] if category_color else [])].melt(
                id_vars=[x] + ([color] if color is not None and color not in [x, *selected_y] else []),
                value_vars=selected_y, var_name="Measure", value_name="Value",
            )
            group_columns = [x, "Measure"] + ([color] if category_color else [])
            if all(numeric_flags):
                aggregated = long.groupby(group_columns, dropna=False, observed=False)["Value"].agg(aggregation).reset_index()
                if category_color:
                    aggregated["Series"] = aggregated["Measure"].astype(str) + " · " + aggregated[color].astype(str)
                    color_column = "Series"
                else:
                    color_column = "Measure"
                return px.bar(aggregated, x=x, y="Value", color=color_column,
                              barmode="stack" if chart_type == "Stacked Bar Chart" else "group",
                              title=f"{aggregation.title()} by {x}")
            counts = long.groupby([x, "Measure", "Value"] + ([color] if category_color else []), dropna=False, observed=False).size().reset_index(name="Count")
            if category_color:
                counts["Series"] = counts["Measure"].astype(str) + " · " + counts[color].astype(str)
                color_column = "Series"
            else:
                color_column = "Value"
            return px.bar(counts, x=x, y="Count", color=color_column,
                          facet_col="Measure", title=f"Category counts by {x}")
        if chart_type in {"Histogram", "Box Plot", "Violin Plot", "Density Plot", "ECDF Plot", "Grouped Box Plot", "Outlier Analysis Plot"}:
            figure = go.Figure()
            for column in selected_y:
                part = build_chart(dataframe, chart_type, x=column, color=color, bins=bins, outlier_method=outlier_method)
                for trace in part.data:
                    trace.name = f"{column}" + (f" · {trace.name}" if trace.name else "")
                    trace.legendgroup = str(column)
                    figure.add_trace(trace)
            figure.update_layout(title=f"{chart_type} · {', '.join(map(str, selected_y))}")
            return figure
        if chart_type in {"Scatter Plot", "Bubble Chart", "Line Chart", "Rolling Average Plot"}:
            figure = go.Figure()
            for column in selected_y:
                part = build_chart(dataframe, chart_type, x=x, y=column, color=color, size=size,
                                   aggregation=aggregation, rolling_window=rolling_window)
                for trace in part.data:
                    trace.name = f"{column}" + (f" · {trace.name}" if trace.name else "")
                    trace.legendgroup = str(column)
                    figure.add_trace(trace)
            figure.update_layout(title=f"{chart_type} · {', '.join(map(str, selected_y))}")
            return figure

    if chart_type == "Histogram":
        _require_numeric(dataframe, x, "X-axis")
        return px.histogram(dataframe, x=x, nbins=bins, title=f"Distribution of {x}")

    if chart_type in {"Box Plot", "Grouped Box Plot"}:
        _require_numeric(dataframe, x, "Primary column")
        group = color
        if chart_type == "Grouped Box Plot" and group is None:
            raise ChartConfigurationError("Choose a grouping column for the grouped box plot.")
        return px.box(dataframe, x=group, y=x, color=group, points="outliers", title=f"Box plot · {x}")

    if chart_type == "Violin Plot":
        _require_numeric(dataframe, x, "Primary column")
        return px.violin(dataframe, x=color, y=x, color=color, box=True, points="outliers", title=f"Violin plot · {x}")

    if chart_type == "Density Plot":
        _require_numeric(dataframe, x, "X-axis")
        values = dataframe[x].dropna().to_numpy(dtype=float)
        if len(values) < 2 or np.ptp(values) == 0:
            raise ChartConfigurationError("Density estimation needs at least two different numerical values.")
        if len(values) > 5000:
            values = np.random.default_rng(17).choice(values, 5000, replace=False)
        density = gaussian_kde(values)
        domain = np.linspace(float(values.min()), float(values.max()), 250)
        figure = go.Figure(go.Scatter(x=domain, y=density(domain), mode="lines", fill="tozeroy", name="Density"))
        figure.update_layout(title=f"Kernel density · {x}", xaxis_title=str(x), yaxis_title="Density")
        return figure

    if chart_type == "ECDF Plot":
        _require_numeric(dataframe, x, "X-axis")
        return px.ecdf(dataframe, x=x, color=color, title=f"Empirical cumulative distribution · {x}")

    if chart_type in {"Bar Chart", "Count Plot", "Stacked Bar Chart"}:
        _require_categorical(dataframe, x, "X-axis")
        if chart_type == "Stacked Bar Chart" and color is None:
            raise ChartConfigurationError("Choose a grouping/color column for a stacked bar chart.")
        if chart_type == "Count Plot":
            aggregated, value_name = _aggregate_categories(dataframe, x, None, color, "count")
            title = f"Count of {x}"
        elif y is not None and y in dataframe.columns and not pd.api.types.is_numeric_dtype(dataframe[y]):
            if chart_type == "Stacked Bar Chart":
                raise ChartConfigurationError("For stacked bars, choose a numerical Y measure; categorical Y values are not encoded as numbers.")
            group_cols = [x] + ([color] if color is not None and color != x else []) + ([y] if y not in {x, color} else [])
            aggregated = dataframe.groupby(group_cols, dropna=False, observed=False).size().reset_index(name="Count")
            figure = px.bar(aggregated, x=x, y="Count", color=y if color is None else color,
                            facet_col=y if color is not None and y != color and aggregated[y].nunique() <= 6 else None,
                            title=f"Count of {y} by {x}")
            return figure
        else:
            aggregated, value_name = _aggregate_categories(dataframe, x, y, color, aggregation)
            title = f"{aggregation.title()} {y}" if y is not None else f"Count of {x}"
        mode = "stack" if chart_type == "Stacked Bar Chart" else "group"
        return px.bar(aggregated, x=x, y=value_name, color=color, barmode=mode, title=title)

    if chart_type == "Pie Chart":
        _require_categorical(dataframe, x, "Category column")
        aggregated, value_name = _aggregate_categories(dataframe, x, y, color, aggregation)
        return px.pie(aggregated, names=x, values=value_name, color=color, title=f"Share by {x}")

    if chart_type in {"Scatter Plot", "Bubble Chart"}:
        _require_numeric(dataframe, x, "X-axis")
        _require_numeric(dataframe, y, "Y-axis")
        if chart_type == "Bubble Chart" and size is not None:
            _require_numeric(dataframe, size, "Bubble size")
        columns = [column for column in [x, y, color, size] if column is not None]
        work = _sample_rows(dataframe[columns], MAX_SCATTER_POINTS)
        return px.scatter(
            work,
            x=x,
            y=y,
            color=color,
            size=size if chart_type == "Bubble Chart" else None,
            title=f"{chart_type} · {y} vs {x}",
            size_max=36,
        )

    if chart_type == "Correlation Heatmap":
        return _build_correlation(dataframe, correlation_columns)

    if chart_type == "Pair Plot":
        columns = list(pair_columns or [])
        if len(columns) < 2:
            raise ChartConfigurationError("Select at least two numerical columns for a pair plot.")
        if len(columns) > MAX_PAIR_COLUMNS:
            raise ChartConfigurationError(f"Select no more than {MAX_PAIR_COLUMNS} numerical columns for the pair plot.")
        if any(column not in numeric_columns(dataframe) for column in columns):
            raise ChartConfigurationError("Pair plots only accept numerical columns.")
        work_columns = columns + ([color] if color is not None and color not in columns else [])
        work = _sample_rows(dataframe[work_columns], MAX_MATRIX_POINTS)
        return px.scatter_matrix(work, dimensions=columns, color=color, title="Pair plot")

    if chart_type in {"Line Chart", "Rolling Average Plot", "Seasonal Decomposition"}:
        _require_column(dataframe, x, "X-axis")
        _require_numeric(dataframe, y, "Y-axis")
        work, is_datetime = _time_series_frame(dataframe, x, y, color, aggregation)
        if chart_type == "Line Chart":
            work = _sample_rows(work, MAX_SCATTER_POINTS, ordered=True)
            figure = px.line(work, x=x, y=y, color=color, markers=True, title=f"{y} over {x}")
            if not is_datetime and not pd.api.types.is_numeric_dtype(dataframe[x]):
                figure.update_xaxes(categoryorder="array", categoryarray=list(work[x].astype(str)))
            return figure
        if chart_type == "Rolling Average Plot":
            window = max(1, int(rolling_window))
            if color is not None and color in work.columns:
                work["Rolling average"] = work.groupby(color, observed=True)[y].transform(lambda series: series.rolling(window, min_periods=1).mean())
            else:
                work["Rolling average"] = work[y].rolling(window, min_periods=1).mean()
            work = _sample_rows(work, MAX_SCATTER_POINTS, ordered=True)
            figure = go.Figure()
            for group_value, group_df in (work.groupby(color, dropna=False, observed=True) if color is not None and color in work.columns else [(None, work)]):
                group_label = "" if group_value is None else f" · {group_value}"
                figure.add_trace(go.Scatter(x=group_df[x], y=group_df[y], mode="lines", name=f"{y}{group_label}"))
                figure.add_trace(go.Scatter(x=group_df[x], y=group_df["Rolling average"], mode="lines", name=f"{window}-point average{group_label}"))
            figure.update_layout(title=f"Rolling average of {y} · window {window}", xaxis_title=str(x), yaxis_title=str(y))
            return figure
        if seasonal_period < 2:
            raise ChartConfigurationError("Seasonal period must be at least 2 observations.")
        if len(work) < seasonal_period * 2:
            raise ChartConfigurationError(f"Seasonal decomposition needs at least two full periods ({seasonal_period * 2} observations).")
        series = work.set_index(x)[y].astype(float)
        if seasonal_model == "multiplicative" and (series <= 0).any():
            raise ChartConfigurationError("Multiplicative decomposition requires strictly positive values. Choose additive mode or another series.")
        try:
            decomposition = seasonal_decompose(series, model=seasonal_model, period=seasonal_period, extrapolate_trend="freq")
        except (ValueError, ZeroDivisionError) as exc:
            raise ChartConfigurationError(f"Seasonal decomposition could not be computed: {exc}") from exc
        figure = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=("Observed", "Trend", "Seasonal", "Residual"))
        for row, values, label in [
            (1, decomposition.observed, "Observed"),
            (2, decomposition.trend, "Trend"),
            (3, decomposition.seasonal, "Seasonal"),
            (4, decomposition.resid, "Residual"),
        ]:
            figure.add_trace(go.Scatter(x=values.index, y=values, mode="lines", name=label), row=row, col=1)
        figure.update_layout(height=800, title=f"Seasonal decomposition · period {seasonal_period}")
        return figure

    if chart_type == "Missing Values Bar Chart":
        counts = dataframe.isna().sum().rename_axis("Column").reset_index(name="Missing values")
        return px.bar(counts, x="Column", y="Missing values", title="Missing values by column")

    if chart_type == "Missing Values Heatmap":
        return _build_missing_heatmap(dataframe)

    if chart_type == "Outlier Analysis Plot":
        return _build_outliers(dataframe, x, outlier_method, color)

    raise ChartConfigurationError(f"The {chart_type} chart is not implemented.")
