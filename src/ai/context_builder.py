from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.profiling_service import profile_dataframe


SENSITIVE_NAME = re.compile(r"email|e-mail|phone|mobile|ssn|social.?security|address|full.?name|first.?name|last.?name|birth|dob|patient|diagnos|medical|account|passport|tax.?id|salary|wage|token|password|secret|credit.?card", re.I)
EMAIL_VALUE = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
PHONE_VALUE = re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{7,}\d)(?!\w)")
CONTEXT_CHAR_LIMIT = 14000


def sensitive_columns(dataframe: pd.DataFrame) -> list[Any]:
    flagged = []
    for column in dataframe.columns:
        if SENSITIVE_NAME.search(str(column)):
            flagged.append(column)
            continue
        sample = dataframe[column].dropna().astype(str).head(100)
        if len(sample) and (sample.str.contains(EMAIL_VALUE).mean() >= 0.1 or sample.str.contains(PHONE_VALUE).mean() >= 0.25):
            flagged.append(column)
    return flagged


def _safe_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)):
        text = str(value)
        if EMAIL_VALUE.search(text) or PHONE_VALUE.search(text):
            return "[redacted: sensitive-looking value]"
        return text[:180] if isinstance(value, str) else value
    return str(value)[:180]


def build_dataset_context(dataframe: pd.DataFrame, dataset_name: str, *, is_cleaned: bool = False,
                          sample_rows: int = 5, max_columns: int = 30) -> dict[str, Any]:
    """Build a compact, calculated context; raw rows are sampled and sensitive values masked."""
    sample_rows = max(0, min(int(sample_rows), 5))
    max_columns = max(1, min(int(max_columns), 30))
    profiles = profile_dataframe(dataframe)
    sensitive = set(sensitive_columns(dataframe))
    schema = []
    statistics: dict[str, Any] = {}
    for item in profiles[:max_columns]:
        schema.append({"name": str(item.name), "pandas_dtype": item.dtype,
                       "detected_type": item.semantic_type, "id_like": item.id_like,
                       "missing": item.missing, "unique": item.unique,
                       "constant": item.constant, "high_cardinality": item.high_cardinality})
        column = item.name
        series = dataframe[column]
        if item.semantic_type == "numeric" and column not in sensitive:
            values = pd.to_numeric(series, errors="coerce").dropna()
            if len(values):
                statistics[str(column)] = {key: _safe_value(value) for key, value in values.describe(percentiles=[.25, .5, .75]).to_dict().items()}
                statistics[str(column)]["skewness"] = _safe_value(values.skew()) if len(values) >= 3 else None
        elif item.semantic_type == "datetime":
            dates = pd.to_datetime(series, errors="coerce", format="mixed").dropna()
            if len(dates):
                statistics[str(column)] = {"earliest": dates.min().isoformat(), "latest": dates.max().isoformat(), "parseable": int(len(dates))}
        elif item.semantic_type in {"categorical", "boolean"}:
            counts = series.value_counts(dropna=True)
            if column in sensitive:
                statistics[str(column)] = {"unique": int(series.nunique(dropna=True)), "frequencies": "omitted: sensitive-looking column"}
            else:
                statistics[str(column)] = {"unique": int(series.nunique(dropna=True)),
                                           "top_frequencies": [{"value": _safe_value(value), "count": int(count)} for value, count in counts.head(5).items()]}

    selected_columns = list(dataframe.columns[:max_columns])
    sample = dataframe[selected_columns].head(sample_rows)
    sample_records = []
    for _, row in sample.iterrows():
        sample_records.append({str(column): ("[redacted: sensitive-looking column]" if column in sensitive else _safe_value(row[column])) for column in selected_columns})

    context = {
        "dataset_name": str(dataset_name)[:160], "rows": int(len(dataframe)), "columns": int(len(dataframe.columns)),
        "schema": schema, "missing_counts": {str(column): int(dataframe[column].isna().sum()) for column in selected_columns},
        "duplicate_rows": int(dataframe.duplicated().sum()) if len(dataframe) else 0,
        "memory_bytes": int(dataframe.memory_usage(index=True, deep=True).sum()),
        "statistics": statistics, "sample_rows": sample_records,
        "sample_note": "At most five rows and thirty columns; sensitive-looking fields and values are redacted.",
        "cleaning_state": "current working data differs from original upload; detailed operation history is not stored" if is_cleaned else "original uploaded data",
        "sensitive_columns": [str(column) for column in sensitive],
    }
    encoded = json.dumps(context, ensure_ascii=False, default=str)
    if len(encoded) > CONTEXT_CHAR_LIMIT:
        context["sample_rows"] = []
        context["statistics"] = {key: value for key, value in list(statistics.items())[:12]}
        context["schema"] = schema[:20]
        encoded = json.dumps(context, ensure_ascii=False, default=str)
        if len(encoded) > CONTEXT_CHAR_LIMIT:
            context["schema"] = context["schema"][:10]
            context["statistics"] = {}
    return context
