import io
import zipfile

import pandas as pd
import pytest

from src.upload.errors import FileValidationError, UnsupportedFormatError
from src.upload.file_detector import detect_file_format
from src.upload.validator import validate_file_size
from tests.helpers import Upload


def make_xlsx() -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        pd.DataFrame({"a": [1]}).to_excel(writer, index=False)
    return buf.getvalue()


def test_detect_supported_format_by_extension_and_signature():
    upload = Upload(make_xlsx(), "sales.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    detected = detect_file_format(upload, upload.name, upload.type)
    assert detected.format == "xlsx"
    assert detected.filename == "sales.xlsx"


def test_detect_rejects_unsupported_format():
    with pytest.raises(UnsupportedFormatError, match="CSV, TSV, XLSX, JSON, or Parquet"):
        detect_file_format(Upload(b"hello", "script.py"), "script.py")


def test_detect_rejects_corrupt_excel_signature():
    with pytest.raises(FileValidationError, match="corrupted"):
        detect_file_format(Upload(b"not a workbook", "broken.xlsx"), "broken.xlsx")


def test_detect_rejects_parquet_without_signature():
    with pytest.raises(FileValidationError, match="signature"):
        detect_file_format(Upload(b"not parquet", "broken.parquet"), "broken.parquet")


def test_empty_file_is_rejected():
    with pytest.raises(FileValidationError, match="empty"):
        validate_file_size(Upload(b"", "empty.csv"), "empty.csv", max_upload_mb=1)


def test_file_size_limit_is_enforced_with_actionable_message():
    upload = Upload(b"x" * (1024 * 1024 + 1), "large.csv")
    with pytest.raises(FileValidationError, match="split it into smaller files"):
        validate_file_size(upload, upload.name, max_upload_mb=1)


def test_detect_rejects_binary_data_named_csv():
    with pytest.raises(FileValidationError, match="text-based CSV"):
        detect_file_format(Upload(b"a\x00b", "binary.csv"), "binary.csv")
