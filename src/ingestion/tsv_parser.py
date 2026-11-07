from __future__ import annotations

from typing import BinaryIO

from .csv_parser import CSVParser
from .base_parser import ParseResult


class TSVParser:
    def parse(self, upload: BinaryIO, **options: object) -> ParseResult:
        result = CSVParser().parse(upload, delimiter="\t", **options)
        result.details["delimiter"] = "tab"
        return result
