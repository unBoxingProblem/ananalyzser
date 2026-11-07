from __future__ import annotations

import json
from typing import Any, BinaryIO

import pandas as pd

from src.upload.errors import ParsingError
from .base_parser import ParseResult


class _DuplicateJSONField(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONField(f"The JSON object repeats field ‘{key}’.")
        result[key] = value
    return result


def _reject_nested(records: list[dict[str, Any]]) -> None:
    for row_number, record in enumerate(records, start=1):
        for key, value in record.items():
            if isinstance(value, (dict, list)):
                raise ParsingError(
                    f"JSON field ‘{key}’ contains nested data (record {row_number}). "
                    "DataPilot does not flatten nested JSON automatically; export a flat table and retry."
                )


def _as_records(document: Any) -> list[dict[str, Any]]:
    if isinstance(document, list):
        if any(not isinstance(row, dict) for row in document):
            raise ParsingError("JSON arrays must contain objects with named fields to load as a table.")
        records = document
    elif isinstance(document, dict):
        if not document:
            raise ParsingError("The JSON object is empty and does not contain tabular records.")
        if set(document) == {"records"} and isinstance(document["records"], list):
            records = document["records"]
            if any(not isinstance(row, dict) for row in records):
                raise ParsingError("The ‘records’ array must contain JSON objects.")
        elif all(isinstance(value, dict) for value in document.values()):
            records = [{"_record_id": key, **value} for key, value in document.items()]
        elif all(not isinstance(value, (dict, list)) for value in document.values()):
            records = [document]
        else:
            raise ParsingError("This JSON structure is nested and cannot be represented as a flat table without changing it.")
    else:
        raise ParsingError("JSON must contain an array of objects or a record-oriented object to load as a table.")
    _reject_nested(records)
    return records


class JSONParser:
    def parse(self, upload: BinaryIO, **_: object) -> ParseResult:
        upload.seek(0)
        raw = upload.read()
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ParsingError("JSON must use UTF-8 encoding. Save it as UTF-8 and try again.") from exc
        warnings: list[str] = []
        try:
            document = json.loads(text, object_pairs_hook=_unique_object)
        except _DuplicateJSONField as exc:
            raise ParsingError(str(exc)) from exc
        except json.JSONDecodeError:
            # JSON Lines is parsed record by record so errors can identify the source line.
            records = []
            for line_number, line in enumerate(text.splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line, object_pairs_hook=_unique_object)
                except _DuplicateJSONField as exc:
                    raise ParsingError(f"Invalid JSON on line {line_number}: {exc}") from exc
                except json.JSONDecodeError as exc:
                    raise ParsingError(f"Invalid JSON on line {line_number}. Check the file syntax and try again.") from exc
                if not isinstance(row, dict):
                    raise ParsingError(f"JSON Lines entry {line_number} is not an object with named fields.")
                records.append(row)
            if not records:
                raise ParsingError("The JSON file contains no readable records.")
            warnings.append("Detected JSON Lines format.")
        else:
            records = _as_records(document)
        _reject_nested(records)
        try:
            frame = pd.DataFrame.from_records(records)
        except Exception as exc:
            raise ParsingError("These JSON records could not be represented as a table.") from exc
        return ParseResult(frame, warnings, {"record_count": len(frame)})
