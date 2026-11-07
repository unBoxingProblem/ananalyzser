from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.profiling_service import ColumnProfile, profile_dataframe


@dataclass(frozen=True)
class ChartRecommendation:
    chart_type: str
    columns: tuple[Any, ...]
    reason: str


def dataset_summary(dataframe: pd.DataFrame) -> dict[str, int | float]:
    return {
        "rows": int(len(dataframe)),
        "columns": int(len(dataframe.columns)),
        "missing_cells": int(dataframe.isna().to_numpy().sum()),
        "duplicate_rows": int(dataframe.duplicated().sum()) if len(dataframe) else 0,
        "memory_bytes": int(dataframe.memory_usage(index=True, deep=True).sum()),
    }


def column_kpis(dataframe: pd.DataFrame, column: Any, profile: ColumnProfile | None = None) -> list[tuple[str, str]]:
    if column not in dataframe.columns:
        return []
    series = dataframe[column]
    if profile is None:
        profile = next(item for item in profile_dataframe(dataframe) if item.name == column)
    semantic = profile.semantic_type
    if semantic == "numeric":
        values = pd.to_numeric(series, errors="coerce").dropna()
        if not len(values):
            return [("Non-missing values", "0")]
        return [("Mean", f"{values.mean():,.4g}"), ("Median", f"{values.median():,.4g}"),
                ("Minimum", f"{values.min():,.4g}"), ("Maximum", f"{values.max():,.4g}"),
                ("Standard deviation", f"{values.std():,.4g}" if len(values) > 1 else "N/A")]
    if semantic == "datetime":
        parsed = pd.to_datetime(series, errors="coerce", format="mixed").dropna()
        if not len(parsed):
            return [("Parsed dates", "0")]
        return [("Earliest", parsed.min().strftime("%Y-%m-%d %H:%M")),
                ("Latest", parsed.max().strftime("%Y-%m-%d %H:%M")),
                ("Unique dates", f"{parsed.nunique():,}")]
    if semantic == "boolean":
        normalized = series.astype("boolean") if pd.api.types.is_bool_dtype(series) else series.astype(str).str.strip().str.lower().map(
            {"true": True, "yes": True, "y": True, "1": True, "false": False, "no": False, "n": False, "0": False}
        )
        true_count = int(normalized.eq(True).sum())
        false_count = int(normalized.eq(False).sum())
        valid = true_count + false_count
        return [("True", f"{true_count:,} ({true_count / valid:.1%})" if valid else "0"),
                ("False", f"{false_count:,} ({false_count / valid:.1%})" if valid else "0"),
                ("Missing / unrecognized", f"{len(series) - valid:,}")]
    counts = series.value_counts(dropna=True)
    if not len(counts):
        return [("Unique values", "0"), ("Dominant value", "N/A"), ("Dominant frequency", "0")]
    top = counts.index[0]
    return [("Unique values", f"{series.nunique(dropna=True):,}"),
            ("Dominant value", str(top)[:80]),
            ("Dominant frequency", f"{int(counts.iloc[0]):,} ({counts.iloc[0] / max(1, series.notna().sum()):.1%})")]


def recommend_charts(dataframe: pd.DataFrame, profiles: list[ColumnProfile] | None = None, *, limit: int = 8) -> list[ChartRecommendation]:
    profiles = profiles if profiles is not None else profile_dataframe(dataframe)
    useful = [item for item in profiles if not item.constant and not item.id_like and item.semantic_type != "identifier"]
    numerics = [item for item in useful if item.semantic_type == "numeric"]
    categories = [item for item in useful if item.semantic_type in {"categorical", "boolean"} and not item.high_cardinality]
    dates = [item for item in useful if item.semantic_type == "datetime"]
    texts = [item for item in useful if item.semantic_type == "text"]
    recs: list[ChartRecommendation] = []
    for item in numerics[:2]:
        recs.append(ChartRecommendation("Histogram", (item.name,), f"Inspect the distribution of {item.name}; {item.missing:,} values are missing." if item.missing else f"Inspect the distribution of {item.name}."))
    for item in categories[:2]:
        recs.append(ChartRecommendation("Count Plot", (item.name,), f"Compare frequencies across {item.unique:,} observed categories."))
    if numerics and categories:
        category = min(categories, key=lambda item: item.unique)
        recs.append(ChartRecommendation("Box Plot", (numerics[0].name, category.name), f"Compare {numerics[0].name} across levels of {category.name}."))
    if numerics and dates:
        recs.append(ChartRecommendation("Line Chart", (dates[0].name, numerics[0].name), f"Inspect how {numerics[0].name} changes over {dates[0].name}."))
    if len(numerics) >= 2:
        recs.append(ChartRecommendation("Correlation Heatmap", tuple(item.name for item in numerics[:20]), "Review pairwise numerical associations; correlation does not establish causation."))
    for item in texts[:1]:
        recs.append(ChartRecommendation("Text Length", (item.name,), f"Inspect the character-length distribution in {item.name}."))
    if not recs and len(dataframe.columns) and dataframe.isna().to_numpy().any():
        recs.append(ChartRecommendation("Missing Values Bar Chart", (), "Review missingness across the available columns."))
    return recs[:limit]


def build_recommended_chart(dataframe: pd.DataFrame, recommendation: ChartRecommendation):
    import plotly.express as px

    chart = recommendation.chart_type
    if chart == "Histogram":
        column = recommendation.columns[0]
        values = pd.to_numeric(dataframe[column], errors="coerce")
        work = pd.DataFrame({str(column): values})
        work = _chart_sample(work)
        return px.histogram(work, x=str(column), title=f"Distribution of {column}")
    if chart == "Count Plot":
        column = recommendation.columns[0]
        counts = dataframe[column].value_counts(dropna=False).head(30).rename_axis(str(column)).reset_index(name="Count")
        return px.bar(counts, x=str(column), y="Count", title=f"Frequency of {column}")
    if chart == "Box Plot":
        numeric, category = recommendation.columns
        work = dataframe[[category]].copy()
        work[str(numeric)] = pd.to_numeric(dataframe[numeric], errors="coerce")
        work = _chart_sample(work)
        return px.box(work, x=category, y=str(numeric), points="outliers", title=f"{numeric} by {category}")
    if chart == "Line Chart":
        date_column, numeric = recommendation.columns
        work = dataframe[[date_column, numeric]].copy()
        work[date_column] = pd.to_datetime(work[date_column], errors="coerce", format="mixed")
        work = work.dropna(subset=[date_column]).sort_values(date_column)
        if len(work) > 5000:
            work = work.iloc[np.linspace(0, len(work) - 1, 5000, dtype=int)]
        return px.line(work, x=date_column, y=numeric, title=f"{numeric} over {date_column}")
    if chart == "Correlation Heatmap":
        columns = list(recommendation.columns)
        work = _chart_sample(dataframe[columns])
        return px.imshow(work.corr(numeric_only=True), text_auto=".2f", zmin=-1, zmax=1,
                         color_continuous_scale="RdBu_r", title="Numerical correlation overview")
    if chart == "Text Length":
        column = recommendation.columns[0]
        lengths = dataframe[column].dropna().astype(str).str.len()
        return px.histogram(pd.DataFrame({"Character count": lengths}), x="Character count", title=f"Text length in {column}")
    from src.visualization.chart_builders import build_chart
    return build_chart(dataframe, chart, x=recommendation.columns[0] if recommendation.columns else None)


def _chart_sample(dataframe: pd.DataFrame, limit: int = 10000) -> pd.DataFrame:
    if len(dataframe) <= limit:
        return dataframe
    return dataframe.iloc[np.linspace(0, len(dataframe) - 1, limit, dtype=int)]


def generate_insights(dataframe: pd.DataFrame, profiles: list[ColumnProfile] | None = None) -> list[tuple[str, str, str]]:
    """Return (kind, label, evidence) triples computed from the current DataFrame."""
    profiles = profiles if profiles is not None else profile_dataframe(dataframe)
    insights: list[tuple[str, str, str]] = []
    if not len(dataframe):
        return [("observation", "No rows", "The current filtered dataset contains no rows.")]
    summary = dataset_summary(dataframe)
    if summary["missing_cells"]:
        total_cells = max(1, len(dataframe) * len(dataframe.columns))
        top = max(profiles, key=lambda item: item.missing_fraction, default=None)
        evidence = f"{summary['missing_cells']:,} missing cells ({summary['missing_cells'] / total_cells:.1%} of all cells)."
        if top and top.missing:
            evidence += f" Highest missing share: {top.name} ({top.missing:,}/{top.rows:,}, {top.missing_fraction:.1%})."
        insights.append(("potential issue", "Missing values", evidence))
    if summary["duplicate_rows"]:
        insights.append(("potential issue", "Duplicate rows", f"{summary['duplicate_rows']:,} completely duplicated row(s) in the current data."))
    for item in profiles:
        if item.constant:
            insights.append(("potential issue", "Constant column", f"{item.name} has {item.unique} distinct non-missing value(s)."))
        if item.high_cardinality and not item.id_like and item.semantic_type in {"categorical", "text"}:
            insights.append(("observation", "High-cardinality column", f"{item.name} has {item.unique:,} distinct values across {item.rows - item.missing:,} present rows."))
        if item.mixed_type:
            insights.append(("potential issue", "Mixed-type column", f"{item.name} contains multiple underlying value types; review parsing and normalization before numeric analysis."))
        if item.semantic_type in {"categorical", "boolean"} and item.unique and not item.id_like:
            counts = dataframe[item.name].value_counts(dropna=True)
            if len(counts):
                insights.append(("observation", "Most frequent category", f"{item.name}: {counts.index[0]} occurs {int(counts.iloc[0]):,} times ({counts.iloc[0] / max(1, dataframe[item.name].notna().sum()):.1%})."))
        if item.semantic_type == "numeric" and not item.id_like and dataframe[item.name].notna().sum() >= 3:
            skew = float(dataframe[item.name].skew())
            if np.isfinite(skew):
                insights.append(("observation", "Numerical skewness", f"{item.name} has skewness {skew:.2f}. This describes distribution shape, not a cause."))
            values = dataframe[item.name].dropna()
            q1, q3 = values.quantile([0.25, 0.75])
            spread = q3 - q1
            candidates = int(((values < q1 - 1.5 * spread) | (values > q3 + 1.5 * spread)).sum()) if spread else 0
            if candidates:
                insights.append(("hypothesis", "Potential outlier values", f"{item.name}: {candidates:,} values fall beyond 1.5×IQR fences. Review context before treating them as errors."))
        if item.semantic_type == "datetime":
            values = pd.to_datetime(dataframe[item.name], errors="coerce", format="mixed").dropna()
            if len(values):
                insights.append(("observation", "Date range", f"{item.name}: {values.min()} to {values.max()} ({len(values):,} parseable rows)."))

    numeric = [item.name for item in profiles if item.semantic_type == "numeric" and not item.id_like and not item.constant][:30]
    if len(numeric) >= 2:
        corr = dataframe[numeric].corr(numeric_only=True)
        pairs = []
        for i, left in enumerate(numeric):
            for right in numeric[i + 1:]:
                value = corr.loc[left, right]
                if pd.notna(value):
                    pairs.append((abs(float(value)), left, right, float(value)))
        for _, left, right, value in sorted(pairs, reverse=True)[:3]:
            insights.append(("observation", "Numerical association", f"{left} and {right} have Pearson correlation {value:.2f}. Correlation does not show causation."))
    return insights[:20]
