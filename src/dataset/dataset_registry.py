from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd

from src.ingestion.dataframe_normalizer import NormalizedDataFrame
from src.upload.upload_manager import IngestionResult
from .dataset_metadata import DatasetMetadata, IngestionErrorRecord


class DatasetRegistry:
    """Session-scoped metadata and DataFrame storage without cloning frames."""

    def __init__(self) -> None:
        self._metadata: dict[str, DatasetMetadata] = {}
        self._dataframes: dict[str, pd.DataFrame] = {}
        self._original_dataframes: dict[str, pd.DataFrame] = {}
        self._previous_dataframes: dict[str, pd.DataFrame | None] = {}
        self._errors: list[IngestionErrorRecord] = []

    def register(self, filename: str, result: IngestionResult) -> DatasetMetadata:
        frame = result.normalized.dataframe
        dataset_id = str(uuid4())
        original_columns = result.normalized.original_columns
        columns = [str(column) for column in original_columns]
        missing = {str(column): int(frame.iloc[:, index].isna().sum()) for index, column in enumerate(original_columns)}
        details: dict[str, str | int | float | bool | list[str]] = {
            key: value for key, value in result.parse_result.details.items()
            if isinstance(value, (str, int, float, bool)) or (isinstance(value, list) and all(isinstance(item, str) for item in value))
        }
        metadata = DatasetMetadata(
            dataset_id=dataset_id,
            filename=filename,
            file_format=result.detected.format,
            file_size_bytes=result.size_bytes,
            row_count=len(frame),
            column_count=len(frame.columns),
            columns=columns,
            dtypes=result.normalized.dtypes,
            missing_values=missing,
            details=details,
            warnings=result.parse_result.warnings + result.normalized.warnings,
        )
        self._metadata[dataset_id] = metadata
        self._dataframes[dataset_id] = frame
        self._original_dataframes[dataset_id] = frame
        self._previous_dataframes[dataset_id] = None
        return metadata

    def add_error(self, filename: str, message: str) -> IngestionErrorRecord:
        record = IngestionErrorRecord(
            filename=filename,
            error_message=message,
            occurred_at=datetime.now(timezone.utc).isoformat(),
        )
        self._errors.append(record)
        return record

    def list_datasets(self) -> list[DatasetMetadata]:
        return list(self._metadata.values())

    def get_metadata(self, dataset_id: str) -> DatasetMetadata:
        try:
            return self._metadata[dataset_id]
        except KeyError as exc:
            raise KeyError(f"No dataset is registered with ID {dataset_id}.") from exc

    def get_dataframe(self, dataset_id: str) -> pd.DataFrame:
        try:
            return self._dataframes[dataset_id]
        except KeyError as exc:
            raise KeyError(f"No dataset is registered with ID {dataset_id}.") from exc

    def get_original_dataframe(self, dataset_id: str) -> pd.DataFrame:
        try:
            return self._original_dataframes[dataset_id]
        except KeyError as exc:
            raise KeyError(f"No dataset is registered with ID {dataset_id}.") from exc

    def set_working_dataframe(self, dataset_id: str, dataframe: pd.DataFrame) -> None:
        if dataset_id not in self._metadata:
            raise KeyError(f"No dataset is registered with ID {dataset_id}.")
        self._previous_dataframes[dataset_id] = self._dataframes[dataset_id]
        self._dataframes[dataset_id] = dataframe

    def has_cleaning_changes(self, dataset_id: str) -> bool:
        return self.get_dataframe(dataset_id) is not self.get_original_dataframe(dataset_id)

    def can_undo_cleaning(self, dataset_id: str) -> bool:
        if dataset_id not in self._metadata:
            raise KeyError(f"No dataset is registered with ID {dataset_id}.")
        return self._previous_dataframes.get(dataset_id) is not None

    def undo_cleaning(self, dataset_id: str) -> bool:
        if not self.can_undo_cleaning(dataset_id):
            return False
        previous = self._previous_dataframes[dataset_id]
        self._dataframes[dataset_id] = previous  # type: ignore[assignment]
        self._previous_dataframes[dataset_id] = None
        return True

    def reset_cleaning(self, dataset_id: str) -> bool:
        if dataset_id not in self._metadata:
            raise KeyError(f"No dataset is registered with ID {dataset_id}.")
        changed = self.has_cleaning_changes(dataset_id)
        self._dataframes[dataset_id] = self._original_dataframes[dataset_id]
        self._previous_dataframes[dataset_id] = None
        return changed

    def remove(self, dataset_id: str) -> bool:
        if dataset_id not in self._metadata:
            return False
        del self._metadata[dataset_id]
        del self._dataframes[dataset_id]
        del self._original_dataframes[dataset_id]
        del self._previous_dataframes[dataset_id]
        return True

    @property
    def errors(self) -> list[IngestionErrorRecord]:
        return list(self._errors)
