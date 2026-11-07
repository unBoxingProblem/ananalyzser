import pandas as pd
import pytest

from src.cleaning.cleaning_operations import CleaningOperation, MissingMode, apply_cleaning, preview_cleaning
from src.cleaning.cleaning_service import CleaningService
from src.dataset.dataset_registry import DatasetRegistry
from src.upload.upload_manager import UploadManager
from tests.helpers import Upload


def make_registry(csv_text: bytes) -> tuple[DatasetRegistry, str]:
    ingested = UploadManager().ingest(Upload(csv_text, "test.csv", "text/csv"), "test.csv", "text/csv")
    registry = DatasetRegistry()
    metadata = registry.register("test.csv", ingested)
    return registry, metadata.dataset_id


def test_remove_rows_containing_any_missing_values():
    frame = pd.DataFrame({"a": [1, None, None, 4], "b": ["x", "y", None, "z"]})
    cleaned, removed = apply_cleaning(frame, CleaningOperation.DROP_MISSING, missing_mode=MissingMode.ANY)
    assert removed == 2
    assert cleaned.to_dict("records") == [{"a": 1.0, "b": "x"}, {"a": 4.0, "b": "z"}]
    assert frame.isna().any(axis=1).sum() == 2  # The input frame was not mutated.


def test_remove_rows_containing_only_missing_values():
    frame = pd.DataFrame({"a": [1, None, None, 4], "b": ["x", "y", None, "z"]})
    preview = preview_cleaning(frame, CleaningOperation.DROP_MISSING, missing_mode=MissingMode.ALL)
    cleaned, removed = apply_cleaning(frame, CleaningOperation.DROP_MISSING, missing_mode=MissingMode.ALL)
    assert preview.rows_to_remove == 1
    assert removed == 1
    assert len(cleaned) == 3
    assert cleaned.iloc[1].isna().any()  # Partially populated rows remain.


def test_remove_completely_empty_rows_is_distinct_option():
    frame = pd.DataFrame({"a": [1, None, None], "b": ["x", None, "z"]})
    cleaned, removed = apply_cleaning(frame, CleaningOperation.DROP_EMPTY_ROWS)
    assert removed == 1
    assert len(cleaned) == 2
    assert not cleaned.iloc[0].isna().any()


def test_remove_duplicates_keeps_first_occurrence():
    frame = pd.DataFrame({"item": ["first", "other", "first"], "qty": [1, 2, 1]})
    preview = preview_cleaning(frame, CleaningOperation.DROP_DUPLICATES)
    cleaned, removed = apply_cleaning(frame, CleaningOperation.DROP_DUPLICATES)
    assert preview.rows_to_remove == 1
    assert removed == 1
    assert cleaned["item"].tolist() == ["first", "other"]
    assert cleaned.iloc[0]["item"] == "first"


def test_no_missing_values_or_duplicates_is_a_noop():
    frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    cleaned, removed = apply_cleaning(frame, CleaningOperation.DROP_MISSING)
    assert removed == 0
    assert cleaned is frame
    cleaned, removed = apply_cleaning(frame, CleaningOperation.DROP_DUPLICATES)
    assert removed == 0
    assert cleaned is frame


def test_empty_dataframe_operations_are_safe():
    frame = pd.DataFrame(columns=["a", "b"])
    for operation in CleaningOperation:
        preview = preview_cleaning(frame, operation)
        cleaned, removed = apply_cleaning(frame, operation)
        assert preview.rows_before == 0
        assert preview.rows_to_remove == 0
        assert removed == 0
        assert cleaned is frame


def test_service_undo_restores_previous_working_frame():
    registry, dataset_id = make_registry(b"a,b\n1,x\n,missing\n2,z\n")
    original = registry.get_original_dataframe(dataset_id)
    service = CleaningService(registry)
    result = service.apply(dataset_id, CleaningOperation.DROP_MISSING)
    assert result.rows_removed == 1
    assert registry.get_dataframe(dataset_id) is not original
    assert registry.get_original_dataframe(dataset_id) is original
    assert registry.get_dataframe(dataset_id).equals(original.dropna())
    assert service.undo(dataset_id)
    assert registry.get_dataframe(dataset_id) is original
    assert not registry.has_cleaning_changes(dataset_id)
    assert not service.undo(dataset_id)


def test_service_reset_restores_original_after_cleaning():
    registry, dataset_id = make_registry(b"a,b\n1,x\n1,x\n2,z\n")
    original = registry.get_original_dataframe(dataset_id)
    service = CleaningService(registry)
    service.apply(dataset_id, CleaningOperation.DROP_DUPLICATES)
    assert len(registry.get_dataframe(dataset_id)) == 2
    assert service.reset(dataset_id)
    assert registry.get_dataframe(dataset_id) is original
    assert not registry.has_cleaning_changes(dataset_id)
    assert not service.reset(dataset_id)


def test_cleaning_one_dataset_does_not_modify_another():
    manager = UploadManager()
    registry = DatasetRegistry()
    dataset_a = registry.register("a.csv", manager.ingest(Upload(b"a,b\n1,x\n,missing\n", "a.csv", "text/csv"), "a.csv", "text/csv"))
    dataset_b = registry.register("b.csv", manager.ingest(Upload(b"a,b\n9,y\n8,z\n", "b.csv", "text/csv"), "b.csv", "text/csv"))
    frame_b = registry.get_dataframe(dataset_b.dataset_id)
    CleaningService(registry).apply(dataset_a.dataset_id, CleaningOperation.DROP_MISSING)
    assert len(registry.get_dataframe(dataset_a.dataset_id)) == 1
    assert registry.get_dataframe(dataset_b.dataset_id) is frame_b
    assert registry.get_dataframe(dataset_b.dataset_id).to_dict("records") == [{"a": 9, "b": "y"}, {"a": 8, "b": "z"}]


def test_service_refuses_to_apply_when_no_rows_would_be_removed():
    registry, dataset_id = make_registry(b"a,b\n1,x\n2,y\n")
    with pytest.raises(ValueError, match="No rows match"):
        CleaningService(registry).apply(dataset_id, CleaningOperation.DROP_MISSING)
