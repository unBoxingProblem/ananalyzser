from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.upload.errors import NormalizationError


@dataclass
class NormalizedDataFrame:
    dataframe: pd.DataFrame
    original_columns: list[object]
    dtypes: dict[str, str]
    warnings: list[str]


class DataFrameNormalizer:
    def normalize(self, dataframe: pd.DataFrame) -> NormalizedDataFrame:
        if not isinstance(dataframe, pd.DataFrame):
            raise NormalizationError("The selected parser did not return a Pandas DataFrame.")
        original_columns = list(dataframe.columns)
        display_names = [str(column) for column in original_columns]
        duplicate_names = sorted({name for name in display_names if display_names.count(name) > 1})
        if dataframe.columns.has_duplicates or duplicate_names:
            names = ", ".join(repr(name) for name in duplicate_names)
            raise NormalizationError(f"Duplicate column names prevent reliable dataset selection ({names}). Rename them in the source file and retry.")
        dtypes = {str(column): str(dtype) for column, dtype in dataframe.dtypes.items()}
        warnings: list[str] = []
        if dataframe.shape == (0, 0):
            warnings.append("The file has no data rows or columns.")
        elif len(dataframe) == 0:
            warnings.append("The file contains column headers but no data rows.")
        elif len(dataframe.columns) == 0:
            warnings.append("The file contains rows but no columns.")
        return NormalizedDataFrame(dataframe, original_columns, dtypes, warnings)
