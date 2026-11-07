from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


ID_NAME_PATTERN = re.compile(r"(^|[^a-z])(id|uuid|guid|identifier|postal|postcode|zipcode|zip|accountnumber|accountno|code)([^a-z]|$)", re.I)
BOOLEAN_VALUES = {"true", "false", "yes", "no", "y", "n", "0", "1"}


@dataclass(frozen=True)
class ColumnProfile:
    name: Any
    dtype: str
    semantic_type: str
    rows: int
    missing: int
    missing_fraction: float
    unique: int
    unique_fraction: float
    id_like: bool
    high_cardinality: bool
    constant: bool
    mixed_type: bool
    examples: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _name_suggests_id(name: Any) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "", str(name).lower())
    return bool(ID_NAME_PATTERN.search(str(name))) or normalized.endswith(("id", "uuid", "guid", "zipcode", "postalcode", "accountno"))


def _mixed_python_types(series: pd.Series) -> bool:
    present = series.dropna()
    if not len(present) or pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
        return False
    kinds = {type(value).__name__ for value in present.iloc[:2000]}
    return len(kinds) > 1


def _values_suggest_identifier(series: pd.Series, unique_fraction: float, unique: int) -> bool:
    if len(series) < 10 or unique < 10 or unique_fraction < 0.98:
        return False
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series.dropna(), errors="coerce")
        if pd.api.types.is_integer_dtype(series) and len(values) > 1 and values.is_monotonic_increasing:
            differences = values.diff().dropna()
            return bool(len(differences) and differences.eq(1).all())
        return False
    sample = series.dropna().astype(str).head(200)
    return bool(len(sample) and sample.str.fullmatch(r"(?=.*\d)[A-Za-z0-9_-]+|[0-9a-fA-F]{8}-[0-9a-fA-F-]{27,}").all())


def _datetime_parse_ratio(series: pd.Series) -> tuple[pd.Series | None, float]:
    if pd.api.types.is_datetime64_any_dtype(series):
        parsed = pd.to_datetime(series, errors="coerce")
        return parsed, 1.0
    if pd.api.types.is_numeric_dtype(series):
        return None, 0.0
    present = series.dropna()
    if not len(present):
        return None, 0.0
    sample = present.iloc[: min(len(present), 5000)]
    try:
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        ratio = float(parsed.notna().mean())
    except (ValueError, TypeError, OverflowError):
        return None, 0.0
    # Avoid treating arbitrary short integer-like strings as dates.
    has_date_hint = sample.astype(str).str.contains(r"[-/:T]|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b", case=False, regex=True).mean() >= 0.5
    if ratio >= 0.85 and has_date_hint:
        full = pd.to_datetime(series, errors="coerce", format="mixed")
        return full, float(full.notna().sum() / max(1, series.notna().sum()))
    return None, ratio


def profile_column(name: Any, series: pd.Series) -> ColumnProfile:
    rows = len(series)
    missing = int(series.isna().sum())
    present = series.dropna()
    unique = int(present.nunique(dropna=True))
    unique_fraction = unique / max(1, len(present))
    constant = unique <= 1
    dtype = str(series.dtype)
    mixed_type = _mixed_python_types(series)
    is_bool = pd.api.types.is_bool_dtype(series)
    if not is_bool and len(present) and not pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_datetime64_any_dtype(series):
        normalized = present.astype(str).str.strip().str.lower()
        is_bool = normalized.nunique() <= 2 and set(normalized.unique()).issubset(BOOLEAN_VALUES) and normalized.nunique() > 0

    parsed_dates, date_ratio = _datetime_parse_ratio(series)
    parsed_numeric: pd.Series | None = None
    numeric_ratio = 0.0
    if len(present) and not pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_datetime64_any_dtype(series) and not mixed_type:
        candidate = pd.to_numeric(series, errors="coerce")
        numeric_ratio = float(candidate.notna().sum() / max(1, series.notna().sum()))
        if numeric_ratio >= 0.9:
            parsed_numeric = candidate
    id_like = _name_suggests_id(name) or _values_suggest_identifier(series, unique_fraction, unique)
    high_cardinality = unique > max(20, int(np.sqrt(max(1, rows)))) or (rows >= 50 and unique_fraction >= 0.5)

    if mixed_type:
        semantic = "mixed"
    elif is_bool:
        semantic = "boolean"
    elif parsed_dates is not None and date_ratio >= 0.85:
        semantic = "datetime"
    elif parsed_numeric is not None:
        semantic = "identifier" if id_like else "numeric"
    elif pd.api.types.is_numeric_dtype(series):
        semantic = "identifier" if id_like else "numeric"
    else:
        text_values = present.astype(str) if len(present) else pd.Series(dtype=str)
        average_length = float(text_values.str.len().mean()) if len(text_values) else 0.0
        semantic = "identifier" if id_like else ("text" if average_length >= 40 or (high_cardinality and average_length >= 24) else "categorical")

    examples = tuple(str(value)[:100] for value in present.drop_duplicates().head(4).tolist())
    return ColumnProfile(name, dtype, semantic, rows, missing, missing / max(1, rows), unique,
                         unique_fraction, id_like, high_cardinality, constant, mixed_type, examples)


def profile_dataframe(dataframe: pd.DataFrame) -> list[ColumnProfile]:
    """Profile every column using dtype, value, cardinality, and name heuristics."""
    return [profile_column(column, dataframe[column]) for column in dataframe.columns]


def profile_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([item.as_dict() for item in profile_dataframe(dataframe)])
