import io

import pandas as pd
import pytest

from src.dataset.dataset_registry import DatasetRegistry
from src.ingestion.dataframe_normalizer import DataFrameNormalizer
from src.upload.errors import NormalizationError
from src.upload.upload_manager import UploadManager
from tests.helpers import Upload


def test_normalizer_preserves_names_dtypes_and_empty_rows():
    frame = pd.DataFrame({"Original name": pd.Series([], dtype="int64")})
    result = DataFrameNormalizer().normalize(frame)
    assert result.original_columns == ["Original name"]
    assert result.dtypes["Original name"] == "int64"
    assert result.warnings


def test_normalizer_rejects_duplicate_columns_without_renaming():
    frame = pd.DataFrame([[1, 2]], columns=["same", "same"])
    with pytest.raises(NormalizationError, match="Duplicate column names"):
        DataFrameNormalizer().normalize(frame)
    assert list(frame.columns) == ["same", "same"]


def test_upload_manager_and_registry_multiple_dataset_lifecycle():
    manager = UploadManager()
    first = manager.ingest(Upload(b"name,value\na,1\n", "first.csv", "text/csv"), "first.csv", "text/csv")
    second = manager.ingest(Upload(b'{"name":"b","value":2}', "second.json", "application/json"), "second.json", "application/json")
    registry = DatasetRegistry()
    one = registry.register("first.csv", first)
    two = registry.register("second.json", second)
    assert one.dataset_id != two.dataset_id
    assert len(registry.list_datasets()) == 2
    assert registry.get_metadata(one.dataset_id).row_count == 1
    stored_frame = registry.get_dataframe(one.dataset_id)
    assert stored_frame is first.normalized.dataframe
    assert registry.get_dataframe(two.dataset_id).iloc[0]["name"] == "b"
    assert registry.remove(one.dataset_id)
    assert not registry.remove(one.dataset_id)
    with pytest.raises(KeyError):
        registry.get_dataframe(one.dataset_id)


def test_registry_records_ingestion_errors_without_file_contents():
    registry = DatasetRegistry()
    record = registry.add_error("bad.csv", "The CSV rows do not have a consistent number of fields.")
    assert record.filename == "bad.csv"
    assert registry.errors == [record]
