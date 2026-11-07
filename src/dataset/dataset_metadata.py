from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DatasetMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    filename: str
    file_format: Literal["csv", "tsv", "xlsx", "json", "parquet"]
    file_size_bytes: int = Field(ge=0)
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    columns: list[str]
    dtypes: dict[str, str]
    missing_values: dict[str, int]
    status: Literal["loaded"] = "loaded"
    details: dict[str, str | int | float | bool | list[str]] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class IngestionErrorRecord(BaseModel):
    filename: str
    error_message: str
    occurred_at: str
