from __future__ import annotations

import os
import zipfile
from dataclasses import dataclass
from typing import BinaryIO

from .errors import FileValidationError, UnsupportedFormatError

SUPPORTED_FORMATS = {"csv", "tsv", "xlsx", "json", "parquet"}
EXTENSION_TO_FORMAT = {fmt: fmt for fmt in SUPPORTED_FORMATS}
MIME_TO_FORMAT = {
    "text/csv": "csv",
    "text/tab-separated-values": "tsv",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/json": "json",
    "text/json": "json",
    "application/vnd.apache.parquet": "parquet",
}


@dataclass(frozen=True)
class DetectedFile:
    filename: str
    format: str
    mime_type: str | None = None


def _size(upload: BinaryIO) -> int:
    size = getattr(upload, "size", None)
    if isinstance(size, int):
        return size
    current = upload.tell()
    upload.seek(0, os.SEEK_END)
    size = upload.tell()
    upload.seek(current)
    return size


def detect_file_format(upload: BinaryIO, filename: str, mime_type: str | None = None) -> DetectedFile:
    """Detect a supported format using its extension and available file signature."""
    ext = os.path.splitext(filename)[1].lower().lstrip(".")
    if ext not in SUPPORTED_FORMATS:
        raise UnsupportedFormatError(
            f"Unsupported file type for ‘{filename}’. Upload CSV, TSV, XLSX, JSON, or Parquet."
        )
    fmt = EXTENSION_TO_FORMAT[ext]
    normalized_mime = (mime_type or "").split(";", 1)[0].strip().lower()
    mime_format = MIME_TO_FORMAT.get(normalized_mime)
    if mime_format is not None and mime_format != fmt:
        raise FileValidationError(
            f"The file extension suggests {fmt.upper()}, but the browser identifies it as {mime_format.upper()}. "
            "Check the file type and upload the original export."
        )
    if _size(upload) == 0:
        raise FileValidationError(f"‘{filename}’ is empty. Choose a file that contains data.")

    upload.seek(0)
    header = upload.read(8192)
    upload.seek(0)
    if not header:
        raise FileValidationError(f"‘{filename}’ is empty. Choose a file that contains data.")

    if fmt == "xlsx":
        try:
            upload.seek(0)
            with zipfile.ZipFile(upload) as archive:
                names = set(archive.namelist())
                if "[Content_Types].xml" not in names or "xl/workbook.xml" not in names:
                    raise FileValidationError("This file has an .xlsx extension but is not a valid Excel workbook.")
        except zipfile.BadZipFile as exc:
            raise FileValidationError("This Excel workbook is corrupted or is not an XLSX file.") from exc
        finally:
            upload.seek(0)
    elif fmt == "parquet":
        if len(header) < 4 or header[:4] != b"PAR1":
            raise FileValidationError("This file has a .parquet extension but its Parquet signature is missing.")
        upload.seek(-4, os.SEEK_END)
        footer = upload.read(4)
        upload.seek(0)
        if footer != b"PAR1":
            raise FileValidationError("This Parquet file is incomplete or corrupted.")
    elif fmt in {"csv", "tsv", "json"}:
        if b"\x00" in header:
            raise FileValidationError(f"‘{filename}’ does not look like a text-based {fmt.upper()} file.")

    return DetectedFile(filename=filename, format=fmt, mime_type=mime_type)
