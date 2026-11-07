from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.dataset.dataset_registry import DatasetRegistry
from .cleaning_operations import CleaningOperation, CleaningPreview, MissingMode, apply_cleaning, preview_cleaning


@dataclass
class CleaningApplyResult:
    operation: CleaningOperation
    rows_before: int
    rows_after: int
    rows_removed: int


class CleaningService:
    """Apply row-level cleaning actions to the selected dataset's working frame."""

    def __init__(self, registry: DatasetRegistry) -> None:
        self.registry = registry

    def preview(
        self,
        dataset_id: str,
        operation: CleaningOperation,
        *,
        missing_mode: MissingMode = MissingMode.ANY,
        preview_limit: int = 10,
    ) -> CleaningPreview:
        return preview_cleaning(
            self.registry.get_dataframe(dataset_id),
            operation,
            missing_mode=missing_mode,
            preview_limit=preview_limit,
        )

    def apply(
        self,
        dataset_id: str,
        operation: CleaningOperation,
        *,
        missing_mode: MissingMode = MissingMode.ANY,
    ) -> CleaningApplyResult:
        current = self.registry.get_dataframe(dataset_id)
        cleaned, removed = apply_cleaning(current, operation, missing_mode=missing_mode)
        if removed == 0:
            raise ValueError("No rows match this operation. The working dataset was not changed.")
        self.registry.set_working_dataframe(dataset_id, cleaned)
        return CleaningApplyResult(
            operation=operation,
            rows_before=len(current),
            rows_after=len(cleaned),
            rows_removed=removed,
        )

    def undo(self, dataset_id: str) -> bool:
        return self.registry.undo_cleaning(dataset_id)

    def reset(self, dataset_id: str) -> bool:
        return self.registry.reset_cleaning(dataset_id)
