from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.visualization.chart_builders import _datetime_series


TIME_CHARTS = {"Line Chart", "Rolling Average Plot", "Seasonal Decomposition"}
CATEGORICAL_X_CHARTS = {"Bar Chart", "Count Plot", "Pie Chart", "Stacked Bar Chart"}
NUMERIC_SERIES_CHARTS = {
    "Histogram", "Box Plot", "Violin Plot", "Density Plot", "ECDF Plot",
    "Scatter Plot", "Bubble Chart", "Line Chart", "Rolling Average Plot",
    "Seasonal Decomposition", "Grouped Box Plot", "Outlier Analysis Plot",
}


class TransformationCompatibilityError(ValueError):
    """The chosen columns cannot be transformed safely for the selected chart."""


@dataclass
class PreparedVisualization:
    dataframe: pd.DataFrame
    x: Any | None
    y_columns: list[Any]
    color: Any | None
    correlation_columns: list[Any] = field(default_factory=list)
    pair_columns: list[Any] = field(default_factory=list)
    notices: list[str] = field(default_factory=list)
    transformations: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    preview: pd.DataFrame | None = None


class TemporaryTransformationService:
    """Prepare chart inputs without mutating or registering transformed data."""

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.original = dataframe
        self.working = dataframe
        self.notices: list[str] = []
        self.transformations: list[dict[str, Any]] = []

    def _add_column(self, values: pd.Series, source: Any, suffix: str, method: str) -> Any:
        base = f"__datapilot_{str(source)}_{suffix}"
        name = base
        suffix_num = 2
        while name in self.working.columns:
            name = f"{base}_{suffix_num}"
            suffix_num += 1
        if self.working is self.original:
            self.working = self.original.copy(deep=False)
        self.working[name] = values
        self.transformations.append({"source": source, "column": name, "method": method})
        return name

    def numeric_column(self, column: Any, label: str) -> Any:
        if column not in self.working.columns:
            raise TransformationCompatibilityError(f"Selected {label} column is unavailable in this dataset.")
        series = self.working[column]
        if pd.api.types.is_numeric_dtype(series):
            return column
        converted = pd.to_numeric(series, errors="coerce")
        nonmissing = int(series.notna().sum())
        converted_count = int(converted.notna().sum())
        if nonmissing and converted_count == nonmissing:
            result = self._add_column(converted, column, "numeric", "numeric conversion")
            self.notices.append(f"Column ‘{column}’ was temporarily converted from {series.dtype} to numerical values for this chart.")
            return result
        if converted_count:
            failed = nonmissing - converted_count
            raise TransformationCompatibilityError(
                f"Column ‘{column}’ has {failed} non-numeric value(s). DataPilot did not drop them; clean or choose another column."
            )
        raise TransformationCompatibilityError(f"Column ‘{column}’ is categorical and has no numerical values for the {label}.")

    def datetime_column(self, column: Any) -> Any:
        if column not in self.working.columns:
            raise TransformationCompatibilityError("Select a valid X column for the time series.")
        parsed = _datetime_series(self.working[column])
        if parsed is None:
            return column
        series = self.working[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            return column
        nonmissing = int(series.notna().sum())
        parsed_count = int(parsed.notna().sum())
        if parsed_count == nonmissing:
            result = self._add_column(parsed, column, "datetime", "datetime parsing")
            self.notices.append(f"Column ‘{column}’ was temporarily parsed as datetime for the time-series X-axis.")
            return result
        failed = nonmissing - parsed_count
        raise TransformationCompatibilityError(
            f"Column ‘{column}’ contains {failed} value(s) that could not be parsed as dates. DataPilot kept all rows; clean the dates or use an ordered X column."
        )

    def encode_column(self, column: Any, method: str) -> list[Any]:
        series = self.working[column]
        nonmissing = series.dropna()
        if method == "one-hot":
            dummies = pd.get_dummies(series, prefix=str(column), prefix_sep=" = ", dummy_na=False, dtype=int)
            if dummies.shape[1] == 0:
                raise TransformationCompatibilityError(f"Column ‘{column}’ has no categories to encode.")
            generated = []
            for dummy_column in dummies.columns:
                name = self._add_column(dummies[dummy_column], column, f"onehot_{dummy_column}", "one-hot encoding")
                generated.append(name)
            missing_count = int(series.isna().sum())
            note = f"Column ‘{column}’ was temporarily one-hot encoded into {len(generated)} indicator column(s)."
            if missing_count:
                note += f" {missing_count} missing value(s) remain unencoded and were not dropped."
            self.notices.append(note)
            return generated
        if method == "frequency":
            frequencies = series.map(nonmissing.value_counts(normalize=True))
            name = self._add_column(frequencies.astype(float), column, "frequency", "frequency encoding")
            self.notices.append(f"Column ‘{column}’ was temporarily frequency-encoded as category proportions; missing values were preserved.")
            return [name]
        if method == "label":
            categories = list(pd.unique(nonmissing))
            mapping = {category: index for index, category in enumerate(categories)}
            encoded = series.map(mapping).astype(float)
            name = self._add_column(encoded, column, "label", "explicit label encoding")
            self.notices.append(f"Column ‘{column}’ was temporarily label-encoded in first-seen order at your request. Use only if that order is meaningful.")
            return [name]
        raise TransformationCompatibilityError("Choose one-hot, frequency, or explicit label encoding for categorical columns.")

    def bin_numeric_column(self, column: Any, method: str, bins: int) -> Any:
        numeric = self.numeric_column(column, "categorical X")
        values = self.working[numeric]
        try:
            if method == "Equal-frequency bins":
                binned = pd.qcut(values, q=bins, duplicates="drop")
            elif method == "Exact numeric categories":
                result = self._add_column(values.astype("string"), column, "categories", "exact numeric categories")
                self.notices.append(f"Numerical X column ‘{column}’ was temporarily treated as exact text categories; no arbitrary numeric encoding was applied.")
                return result
            else:
                binned = pd.cut(values, bins=bins, include_lowest=True)
        except ValueError as exc:
            raise TransformationCompatibilityError(f"Could not bin ‘{column}’: {exc}") from exc
        result = self._add_column(binned, column, "bins", method.lower())
        found = int(binned.nunique(dropna=True))
        self.notices.append(f"Numerical column ‘{column}’ was temporarily converted to {found} {method.lower()} for the categorical chart.")
        return result

    def normalize_column(self, column: Any, method: str) -> Any:
        series = self.working[column]
        if method == "Z-score":
            standard_deviation = series.std(ddof=0)
            values = (series - series.mean()) / standard_deviation if standard_deviation else series * 0
        elif method == "Min-max":
            spread = series.max() - series.min()
            values = (series - series.min()) / spread if spread else series * 0
        else:
            return column
        result = self._add_column(values, column, method.lower().replace("-", ""), method.lower() + " normalization")
        return result

    def prepare(
        self,
        chart_type: str,
        x: Any | None,
        y_columns: list[Any],
        *,
        categorical_encoding: str = "Automatic",
        encode_columns: list[Any] | None = None,
        numeric_x_binning: str = "Equal-width bins",
        bins: int = 10,
        normalization: str = "None",
        correlation_columns: list[Any] | None = None,
        pair_columns: list[Any] | None = None,
    ) -> PreparedVisualization:
        y_columns = list(dict.fromkeys(column for column in y_columns if column is not None))
        correlation_columns = list(dict.fromkeys(correlation_columns or []))
        pair_columns = list(dict.fromkeys(pair_columns or []))
        x_result = x
        y_result = list(y_columns)
        correlation_result = list(correlation_columns)
        pair_result = list(pair_columns)

        try:
            if chart_type in TIME_CHARTS and x is not None:
                x_result = self.datetime_column(x)

            if chart_type in CATEGORICAL_X_CHARTS and x is not None and x in self.working.columns and pd.api.types.is_numeric_dtype(self.working[x]):
                x_result = self.bin_numeric_column(x, numeric_x_binning, bins)

            if chart_type in NUMERIC_SERIES_CHARTS:
                converted_y: list[Any] = []
                for column in y_result:
                    if column not in self.working.columns:
                        continue
                    if pd.api.types.is_numeric_dtype(self.working[column]):
                        converted_y.append(column)
                        continue
                    if chart_type == "Scatter Plot" and categorical_encoding in {"Automatic", "One-hot"}:
                        converted_y.extend(self.encode_column(column, "one-hot"))
                        continue
                    try:
                        converted_y.append(self.numeric_column(column, "Y-axis"))
                    except TransformationCompatibilityError:
                        raise
                y_result = converted_y

                if chart_type == "Histogram" and x is not None and not y_result:
                    # Legacy single-column configuration uses X as the measured variable.
                    x_result = self.numeric_column(x, "X-axis")

                if chart_type in {"Scatter Plot", "Bubble Chart"} and x is not None:
                    if x not in self.working.columns:
                        raise TransformationCompatibilityError("Select a valid X column for this chart.")
                    if pd.api.types.is_numeric_dtype(self.working[x]):
                        x_result = x
                    else:
                        try:
                            x_result = self.numeric_column(x, "X-axis")
                        except TransformationCompatibilityError:
                            raise TransformationCompatibilityError(
                                f"X column ‘{x}’ is categorical. A continuous scatter X-axis would be misleading; use Bar Chart or Box Plot for category comparisons."
                            ) from None

                if chart_type in {"Box Plot", "Violin Plot", "Grouped Box Plot", "Density Plot", "ECDF Plot", "Outlier Analysis Plot"} and x is not None and not y_result:
                    x_result = self.numeric_column(x, "Primary column")

            if chart_type == "Correlation Heatmap":
                requested = list(dict.fromkeys(([x] if x is not None else []) + y_result + correlation_result))
                encoded = set(encode_columns or [])
                transformed_columns: list[Any] = []
                for column in requested:
                    if column not in self.working.columns:
                        continue
                    if pd.api.types.is_numeric_dtype(self.working[column]):
                        transformed_columns.append(column)
                    elif column in encoded and categorical_encoding not in {"Automatic", "None"}:
                        transformed_columns.extend(self.encode_column(column, categorical_encoding.lower()))
                    else:
                        raise TransformationCompatibilityError(
                            f"Column ‘{column}’ is categorical. Select it for explicit one-hot, frequency, or label encoding, or remove it from the heatmap."
                        )
                correlation_result = list(dict.fromkeys(transformed_columns))

            if chart_type == "Pair Plot":
                pair_result = list(dict.fromkeys(([x] if x is not None else []) + y_result + pair_result))
                nonnumeric = [column for column in pair_result if column in self.working.columns and not pd.api.types.is_numeric_dtype(self.working[column])]
                if nonnumeric:
                    raise TransformationCompatibilityError(f"Pair plots need numerical columns. ‘{nonnumeric[0]}’ is categorical; select numerical columns only.")

            if normalization != "None" and len(y_result) > 1:
                normalized: list[Any] = []
                for column in y_result:
                    if not pd.api.types.is_numeric_dtype(self.working[column]):
                        raise TransformationCompatibilityError("Normalization is available only for numerical comparison columns.")
                    normalized.append(self.normalize_column(column, normalization))
                y_result = normalized
                self.notices.append(f"Selected Y series were temporarily normalized using {normalization}. This changes their scale; the original values remain untouched.")

            preview_columns = list(dict.fromkeys(
                [column for column in [x_result, *y_result] if column is not None and column in self.working.columns]
            ))
            preview = self.working[preview_columns].head(8) if self.transformations else None
            return PreparedVisualization(
                dataframe=self.working,
                x=x_result,
                y_columns=y_result,
                color=None,
                correlation_columns=correlation_result,
                pair_columns=pair_result,
                notices=self.notices,
                transformations=self.transformations,
                preview=preview,
            )
        except TransformationCompatibilityError as exc:
            return PreparedVisualization(
                dataframe=self.working,
                x=x_result,
                y_columns=y_result,
                color=None,
                correlation_columns=correlation_result,
                pair_columns=pair_result,
                notices=self.notices,
                transformations=self.transformations,
                error=str(exc),
            )


def color_column(dataframe: pd.DataFrame, selected: Any | None) -> Any | None:
    return selected if selected in dataframe.columns else None


def prepare_visualization(
    dataframe: pd.DataFrame,
    chart_type: str,
    x: Any | None,
    y_columns: list[Any],
    *,
    color: Any | None = None,
    **options: Any,
) -> PreparedVisualization:
    service = TemporaryTransformationService(dataframe)
    prepared = service.prepare(chart_type, x, y_columns, **options)
    prepared.color = color_column(prepared.dataframe, color)
    return prepared
