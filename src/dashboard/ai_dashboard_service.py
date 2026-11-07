from __future__ import annotations

from typing import Any
from uuid import uuid4

import pandas as pd
import numpy as np

from src.dashboard.chart_service import AGGREGATIONS, DASHBOARD_CHART_TYPES, DashboardChartError, build_dashboard_chart, validate_chart_config
from src.dashboard.dashboard_store import DashboardStore
from src.analysis.profiling_service import profile_dataframe


class AIDashboardError(ValueError):
    """A dashboard request could not be converted into valid saved charts."""


def create_ai_dashboard(store: DashboardStore, dataframe: pd.DataFrame, dataset_id: str,
                        dataset_name: str, charts: list[dict[str, Any]],
                        dashboard_name: str | None = None,
                        planning_notes: list[str] | None = None) -> dict[str, Any]:
    """Validate, render-check, and persist a model-proposed interactive dashboard."""
    if not isinstance(charts, list) or not charts:
        raise AIDashboardError("A dashboard needs at least one chart configuration.")
    if len(charts) > 8:
        raise AIDashboardError("A dashboard request can contain at most eight charts.")

    allowed = {"chart_type", "x", "y_columns", "group_by", "color_column", "aggregation", "title",
               "sort_order", "top_n", "bins", "size_column"}
    prepared: list[dict[str, Any]] = []
    for index, proposed in enumerate(charts, start=1):
        if not isinstance(proposed, dict):
            raise AIDashboardError(f"Chart {index} must be an object with chart settings.")
        unknown = set(proposed) - allowed
        if unknown:
            raise AIDashboardError(f"Chart {index} includes unsupported setting(s): {', '.join(sorted(unknown))}.")
        chart_type = proposed.get("chart_type")
        if chart_type not in DASHBOARD_CHART_TYPES:
            raise AIDashboardError(f"Chart {index} needs a supported chart_type.")
        aggregation = proposed.get("aggregation", "Mean")
        if aggregation not in AGGREGATIONS:
            raise AIDashboardError(f"Chart {index} uses an unsupported aggregation.")
        config = {
            "chart_id": str(uuid4()), "dataset_id": dataset_id, "dataset_name": dataset_name,
            "chart_type": chart_type, "x": proposed.get("x"),
            "y_columns": proposed.get("y_columns") or [], "group_by": proposed.get("group_by"),
            "color_column": proposed.get("color_column") or proposed.get("group_by"),
            "aggregation": aggregation, "title": str(proposed.get("title") or f"{chart_type} · {dataset_name}")[:120],
            "sort_order": proposed.get("sort_order", "Descending"),
            "top_n": int(proposed.get("top_n", 20)), "bins": int(proposed.get("bins", 30)),
            "size_column": proposed.get("size_column"), "filters": {},
        }
        if not isinstance(config["y_columns"], list):
            raise AIDashboardError(f"Chart {index} y_columns must be a list.")
        issue = validate_chart_config(dataframe, config)
        if issue:
            raise AIDashboardError(f"Chart {index} ({chart_type}): {issue}")
        try:
            # Build the figure before saving, so invalid combinations never create a partial dashboard.
            build_dashboard_chart(dataframe, config)
        except DashboardChartError as exc:
            raise AIDashboardError(f"Chart {index} ({chart_type}): {exc}") from exc
        prepared.append(config)

    created = store.create_dashboard(dashboard_name or f"{dataset_name} · AI Dashboard")
    created["charts"] = prepared
    saved = store.save_dashboard(created)
    return {
        "operation": "dashboard_created", "dashboard_id": saved["dashboard_id"],
        "dashboard_name": saved["name"], "dataset_id": dataset_id,
        "dataset_name": dataset_name, "chart_count": len(prepared), "charts": prepared,
        "planning_notes": list(planning_notes or []),
    }


def analyze_dashboard_patterns(dataframe: pd.DataFrame, limit: int = 8) -> dict[str, Any]:
    """Rank dashboard charts using measured distributions and relationships."""
    profiles = profile_dataframe(dataframe)
    numeric = [str(item.name) for item in profiles if item.semantic_type == "numeric" and not item.id_like and not item.constant]
    categorical = [str(item.name) for item in profiles if item.semantic_type in {"categorical", "boolean"}
                   and not item.id_like and not item.high_cardinality and not item.constant]
    dates = [str(item.name) for item in profiles if item.semantic_type == "datetime"
             and pd.api.types.is_datetime64_any_dtype(dataframe[item.name])]
    text_columns = []
    for item in profiles:
        if item.id_like or item.name not in dataframe.columns or item.semantic_type in {"numeric", "datetime", "boolean", "identifier"}:
            continue
        values = dataframe[item.name].dropna().astype(str)
        if len(values) and values.str.len().mean() >= 24 and values.str.contains(r"\s", regex=True).mean() >= .3:
            text_columns.append(str(item.name))
    # Prose fields may be profiled as categorical; never treat a text body as a
    # grouping dimension (which would create one group per sentence/record).
    categorical = [column for column in categorical if column not in text_columns]

    candidates: list[dict[str, Any]] = []
    observations: list[str] = []

    for date in dates:
        for metric in numeric[:3]:
            candidates.append({"score": .95, "family": "time", "rationale": f"{date} is datetime and {metric} is numeric, so a time series can show how the measure changes over time.",
                               "chart": {"chart_type": "Line Chart", "x": date, "y_columns": [metric], "title": f"{metric} over time"}})

    strongest_category_pair: tuple[float, str, str, float] | None = None
    for category in categorical:
        for metric in numeric:
            pair = dataframe[[category, metric]].dropna()
            if len(pair) < 6 or pair[category].nunique() < 2 or pair[category].nunique() > 30:
                continue
            values = pd.to_numeric(pair[metric], errors="coerce")
            pair = pair.loc[values.notna()].copy()
            values = pd.to_numeric(pair[metric], errors="coerce")
            overall = float(values.mean()) if len(values) else 0.0
            total = float(((values - overall) ** 2).sum())
            group_means = pair.assign(_metric=values).groupby(category, observed=True)["_metric"].agg(["mean", "count"])
            between = float((group_means["count"] * (group_means["mean"] - overall) ** 2).sum())
            effect = between / total if total > 0 else 0.0
            pair_score = .5 + min(.45, effect * .7)
            if strongest_category_pair is None or effect > strongest_category_pair[0]:
                strongest_category_pair = (effect, category, metric, pair_score)

    if strongest_category_pair:
        effect, category, metric, score = strongest_category_pair
        candidates.append({"score": score, "family": "group_metric",
            "rationale": f"Pandas compared {metric} across {dataframe[category].nunique()} values of {category}; the between-group effect size (eta-squared) is {effect:.2f}.",
            "chart": {"chart_type": "Bar Chart", "x": category, "y_columns": [metric], "aggregation": "Mean",
                      "title": f"Average {metric} by {category}"}})
        if effect >= .08 and dataframe[category].nunique() <= 12:
            candidates.append({"score": score - .04, "family": "group_distribution",
                "rationale": f"A box plot shows the within-group spread of {metric}, which a group average can hide.",
                "chart": {"chart_type": "Box Plot", "x": None, "y_columns": [metric], "group_by": category,
                          "title": f"{metric} spread by {category}"}})

    if len(numeric) >= 2:
        correlations = dataframe[numeric].corr(numeric_only=True).abs().where(lambda matrix: np.triu(np.ones(matrix.shape), k=1).astype(bool)).stack()
        if len(correlations):
            first, second = correlations.idxmax()
            strength = float(correlations.loc[(first, second)])
            if strength >= .25:
                signed = float(dataframe[[first, second]].corr(numeric_only=True).loc[first, second])
                candidates.append({"score": .55 + .4 * strength, "family": "relationship",
                    "rationale": f"The strongest numeric pair is {first} and {second} (Pearson r={signed:.2f}); a scatter plot exposes its shape and outliers.",
                    "chart": {"chart_type": "Scatter Plot", "x": first, "y_columns": [second],
                              "title": f"{first} vs {second}"}})
            if len(numeric) >= 3 and strength >= .3:
                candidates.append({"score": .58 + .3 * strength, "family": "correlation_matrix",
                    "rationale": f"The dataset has {len(numeric)} usable numeric measures and at least one notable pairwise correlation (|r|={strength:.2f}).",
                    "chart": {"chart_type": "Heatmap", "y_columns": numeric[:12], "title": "Numeric correlation patterns"}})

    for text_column in text_columns:
        if categorical:
            for category in categorical:
                sizes = dataframe.groupby(category, dropna=False, observed=True).size()
                if len(sizes) < 2 or len(sizes) > 30:
                    continue
                lengths = dataframe[text_column].fillna("").astype("string").str.len()
                group_frame = pd.DataFrame({"group": dataframe[category], "length": lengths}).dropna()
                means = group_frame.groupby("group", observed=True)["length"].mean()
                total = float(((group_frame["length"] - group_frame["length"].mean()) ** 2).sum())
                between = float((group_frame.groupby("group", observed=True).size() * (means - group_frame["length"].mean()) ** 2).sum())
                effect = between / total if total else 0.0
                candidates.append({"score": .58 + min(.32, effect * .8), "family": "text_by_group",
                    "rationale": f"Pandas compared character counts in {text_column} across {len(means)} values of {category}; eta-squared={effect:.2f}.",
                    "chart": {"chart_type": "Text Length by Group", "x": text_column, "group_by": category,
                              "title": f"{text_column} length by {category}"}})
        lengths = dataframe[text_column].dropna().astype("string").str.len()
        if len(lengths):
            median = float(lengths.median())
            skew = float(lengths.skew()) if len(lengths) >= 3 else 0.0
            candidates.append({"score": .53 + min(.25, abs(skew) * .1), "family": "text_distribution",
                "rationale": f"Text in {text_column} has median length {median:.0f} characters and length skewness {skew:.2f}.",
                "chart": {"chart_type": "Text Length", "x": text_column, "title": f"Text length distribution · {text_column}"}})

    if numeric:
        stats = []
        for metric in numeric:
            values = pd.to_numeric(dataframe[metric], errors="coerce").dropna()
            if len(values) < 3:
                continue
            skew = float(values.skew()) if len(values) >= 3 else 0.0
            q1, q3 = values.quantile([.25, .75])
            spread = float(q3 - q1)
            outlier_rate = float((((values < q1 - 1.5 * spread) | (values > q3 + 1.5 * spread)).mean()) if spread else 0.0)
            stats.append((abs(skew) + outlier_rate, metric, skew, outlier_rate))
        for score, metric, skew, outlier_rate in sorted(stats, reverse=True)[:2]:
            candidates.append({"score": .5 + min(.25, score * .08), "family": f"distribution:{metric}",
                "rationale": f"{metric} has skewness {skew:.2f} and {outlier_rate:.1%} of values beyond the 1.5×IQR fences; its distribution merits inspection.",
                "chart": {"chart_type": "Histogram", "x": metric, "title": f"Distribution of {metric}"}})

    if not numeric and not text_columns:
        for category in categorical:
            counts = dataframe[category].value_counts(dropna=False)
            if len(counts) < 2:
                continue
            coefficient = float(counts.std(ddof=0) / counts.mean()) if counts.mean() else 0.0
            if coefficient >= .05:
                candidates.append({"score": .48 + min(.3, coefficient * .1), "family": f"category:{category}",
                    "rationale": f"Counts for {category} vary across categories (coefficient of variation {coefficient:.2f}).",
                    "chart": {"chart_type": "Count Plot", "x": category, "title": f"Records by {category}"}})
        if not candidates and categorical:
            category = max(categorical, key=lambda name: int(dataframe[name].nunique(dropna=True)))
            candidates.append({"score": .45, "family": f"category:{category}",
                "rationale": f"{category} is the most informative low-cardinality field available for a category count.",
                "chart": {"chart_type": "Count Plot", "x": category, "title": f"Records by {category}"}})

    candidates.sort(key=lambda item: item["score"], reverse=True)
    selected: list[dict[str, Any]] = []
    families: set[str] = set()
    signatures: set[tuple[Any, ...]] = set()
    for candidate in candidates:
        signature = (candidate["chart"]["chart_type"], candidate["chart"].get("x"),
                     tuple(candidate["chart"].get("y_columns", [])), candidate["chart"].get("group_by"))
        if candidate["family"] in families or signature in signatures:
            continue
        families.add(candidate["family"])
        signatures.add(signature)
        selected.append(candidate)
        if len(selected) >= limit:
            break
    if not selected:
        observations.append("No chart was recommended because the loaded columns do not support a useful plot without additional data or a user-selected transformation.")
    else:
        observations.append(f"Reviewed {len(profiles)} columns, tested group effects and numeric associations, then selected {len(selected)} non-duplicate chart(s) with the strongest supported patterns.")
    return {"suggested_charts": selected, "insights": observations,
            "profiled_columns": len(profiles), "rows": len(dataframe)}


def recommend_dashboard_charts(dataframe: pd.DataFrame, limit: int = 8) -> list[dict[str, Any]]:
    return [item["chart"] for item in analyze_dashboard_patterns(dataframe, limit)["suggested_charts"]]
