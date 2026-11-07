from __future__ import annotations

import logging
from typing import BinaryIO

from src.upload.errors import ParsingError, UnsupportedFormatError, UploadError
from src.upload.file_detector import DetectedFile
from .base_parser import FileParser, ParseResult
from .csv_parser import CSVParser
from .excel_parser import ExcelParser
from .json_parser import JSONParser
from .parquet_parser import ParquetParser
from .tsv_parser import TSVParser

logger = logging.getLogger(__name__)

PARSER_REGISTRY: dict[str, FileParser] = {
    "csv": CSVParser(),
    "tsv": TSVParser(),
    "xlsx": ExcelParser(),
    "json": JSONParser(),
    "parquet": ParquetParser(),
}


class ParserRegistry:
    def __init__(self, parsers: dict[str, FileParser] | None = None) -> None:
        self.parsers = parsers if parsers is not None else PARSER_REGISTRY

    def parse(self, upload: BinaryIO, detected: DetectedFile, **options: object) -> ParseResult:
        parser = self.parsers.get(detected.format)
        if parser is None:
            raise UnsupportedFormatError(f"No parser is available for {detected.format.upper()} files.")
        try:
            upload.seek(0)
            result = parser.parse(upload, **options)
            upload.seek(0)
            return result
        except UploadError:
            raise
        except MemoryError as exc:
            logger.exception("Memory exhaustion parsing uploaded file %s (%s)", detected.filename, detected.format)
            raise ParsingError("This file needs more memory than the current app can provide. Reduce or split it and retry.") from exc
        except Exception as exc:
            logger.exception("Unexpected parsing error for uploaded file %s (%s)", detected.filename, detected.format)
            raise ParsingError(f"Could not read ‘{detected.filename}’. Check that the file is valid {detected.format.upper()} data.") from exc
