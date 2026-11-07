from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import BinaryIO

from src.ingestion.base_parser import ParseResult
from src.ingestion.dataframe_normalizer import DataFrameNormalizer, NormalizedDataFrame
from src.ingestion.excel_parser import ExcelParser
from src.ingestion.parser_registry import ParserRegistry
from .errors import UploadError
from .file_detector import DetectedFile, detect_file_format
from .validator import configured_max_upload_mb, validate_file_size

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    detected: DetectedFile
    size_bytes: int
    parse_result: ParseResult
    normalized: NormalizedDataFrame


class UploadManager:
    def __init__(self, registry: ParserRegistry | None = None, normalizer: DataFrameNormalizer | None = None) -> None:
        self.parser_registry = registry or ParserRegistry()
        self.normalizer = normalizer or DataFrameNormalizer()

    def inspect(self, upload: BinaryIO, filename: str, mime_type: str | None = None) -> tuple[DetectedFile, int, list[str]]:
        size = validate_file_size(upload, filename, configured_max_upload_mb())
        detected = detect_file_format(upload, filename, mime_type)
        sheet_names: list[str] = []
        if detected.format == "xlsx":
            parser = self.parser_registry.parsers.get("xlsx")
            if not isinstance(parser, ExcelParser):
                raise UploadError("The Excel parser is not available.")
            sheet_names = parser.sheet_names(upload)
        upload.seek(0)
        return detected, size, sheet_names

    def ingest(
        self,
        upload: BinaryIO,
        filename: str,
        mime_type: str | None = None,
        **options: object,
    ) -> IngestionResult:
        detected, size, _ = self.inspect(upload, filename, mime_type)
        try:
            parsed = self.parser_registry.parse(upload, detected, **options)
            normalized = self.normalizer.normalize(parsed.dataframe)
            return IngestionResult(
                detected=detected,
                size_bytes=size,
                parse_result=parsed,
                normalized=normalized,
            )
        except UploadError:
            logger.exception("Ingestion failed for %s (%s)", filename, detected.format)
            raise
        except Exception as exc:
            logger.exception("Unexpected ingestion failure for %s (%s)", filename, detected.format)
            raise UploadError(f"Could not ingest ‘{filename}’. Check that the file is valid {detected.format.upper()} data.") from exc
