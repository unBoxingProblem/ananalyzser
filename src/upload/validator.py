from __future__ import annotations

import os
from typing import BinaryIO

from .errors import FileValidationError

DEFAULT_MAX_UPLOAD_MB = 100


def configured_max_upload_mb() -> int:
    raw = os.getenv("DATAPILOT_MAX_UPLOAD_MB", str(DEFAULT_MAX_UPLOAD_MB))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_MAX_UPLOAD_MB
    return value if value > 0 else DEFAULT_MAX_UPLOAD_MB


def get_file_size(upload: BinaryIO) -> int:
    size = getattr(upload, "size", None)
    if isinstance(size, int):
        return size
    pos = upload.tell()
    upload.seek(0, os.SEEK_END)
    size = upload.tell()
    upload.seek(pos)
    return size


def validate_file_size(upload: BinaryIO, filename: str, max_upload_mb: int | None = None) -> int:
    limit_mb = max_upload_mb if max_upload_mb is not None else configured_max_upload_mb()
    if limit_mb <= 0:
        raise ValueError("The upload size limit must be a positive number of megabytes.")
    size = get_file_size(upload)
    if size == 0:
        raise FileValidationError(f"‘{filename}’ is empty. Choose a file that contains data.")
    limit = limit_mb * 1024 * 1024
    if size > limit:
        raise FileValidationError(
            f"‘{filename}’ is {size / 1024 / 1024:.1f} MB; the limit is {limit_mb} MB per file. "
            "Reduce the file or split it into smaller files and upload them separately."
        )
    return size
