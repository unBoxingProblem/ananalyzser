from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd


class CleaningOperation(str, Enum):
    DROP_MISSING = "drop_missing"
    DROP_DUPLICATES = "drop_duplicates"
    DROP_EMPTY_ROWS = "drop_empty_rows"


class MissingMode(str, Enum):
    ANY = "any"
    ALL = "all"


DESCRIPTIONS = {
    CleaningOperation.DROP_MISSING: "Remove rows with one or more missing cells, or choose to remove only rows where every cell is missing.",
    CleaningOperation.DROP_DUPLICATES: "Remove completely identical rows. The first occurrence is kept; no individual columns are used for matching.",
    CleaningOperation.DROP_EMPTY_ROWS: "Remove rows where every column is missing. Rows with at least one populated cell are kept.",
}


@dataclass
class CleaningPreview:
    operation: CleaningOperation
    rows_before: int
    rows_to_remove: int
    rows_after: int
    removed_rows_preview: pd.DataFrame


def _rows_to_remove_mask(
    dataframe: pd.DataFrame,
    operation: CleaningOperation,
    missing_mode: MissingMode = MissingMode.ANY,
) -> pd.Series:
    if operation == CleaningOperation.DROP_MISSING:
        return dataframe.isna().any(axis=1) if missing_mode == MissingMode.ANY else dataframe.isna().all(axis=1)
    if operation == CleaningOperation.DROP_EMPTY_ROWS:
        if dataframe.shape[1] == 0:
            return pd.Series(False, index=dataframe.index)
        return dataframe.isna().all(axis=1)
    if operation == CleaningOperation.DROP_DUPLICATES:
        return dataframe.duplicated(keep="first")
    raise ValueError(f"Unsupported cleaning operation: {operation}")


def preview_cleaning(
    dataframe: pd.DataFrame,
    operation: CleaningOperation,
    *,
    missing_mode: MissingMode = MissingMode.ANY,
    preview_limit: int = 10,
) -> CleaningPreview:
    mask = _rows_to_remove_mask(dataframe, operation, missing_mode)
    count = int(mask.sum())
    return CleaningPreview(
        operation=operation,
        rows_before=len(dataframe),
        rows_to_remove=count,
        rows_after=len(dataframe) - count,
        removed_rows_preview=dataframe.loc[mask].head(max(0, preview_limit)),
    )


def apply_cleaning(
    dataframe: pd.DataFrame,
    operation: CleaningOperation,
    *,
    missing_mode: MissingMode = MissingMode.ANY,
) -> tuple[pd.DataFrame, int]:
    """Return a cleaned frame without mutating the caller's DataFrame."""
    preview = preview_cleaning(dataframe, operation, missing_mode=missing_mode, preview_limit=0)
    if preview.rows_to_remove == 0:
        return dataframe, 0
    if operation == CleaningOperation.DROP_MISSING:
        cleaned = dataframe.dropna(how="any" if missing_mode == MissingMode.ANY else "all")
    elif operation == CleaningOperation.DROP_EMPTY_ROWS:
        cleaned = dataframe.dropna(how="all")
    elif operation == CleaningOperation.DROP_DUPLICATES:
        cleaned = dataframe.drop_duplicates(keep="first")
    else:
        raise ValueError(f"Unsupported cleaning operation: {operation}")
    return cleaned, preview.rows_to_remove
