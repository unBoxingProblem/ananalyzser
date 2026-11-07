from __future__ import annotations

import csv
import io
from typing import BinaryIO

import pandas as pd
from pandas.errors import EmptyDataError, ParserError

from src.upload.errors import ParsingError
from .base_parser import ParseResult

DELIMITER_OPTIONS = {"auto": None, "comma (,)": ",", "semicolon (;)": ";", "tab (\\t)": "\t"}


def _decode_sample(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ParsingError("The CSV text encoding is not supported. Save it as UTF-8 and try again.")


def _read_header(upload: BinaryIO, delimiter: str, encoding: str) -> list[str] | None:
    upload.seek(0)
    sample = upload.read(128 * 1024)
    upload.seek(0)
    text = sample.decode(encoding, errors="replace")
    try:
        return next(csv.reader(io.StringIO(text), delimiter=delimiter, strict=True), None)
    except csv.Error as exc:
        raise ParsingError(f"Could not read the CSV header: {exc}.") from exc


def _check_duplicate_headers(headers: list[str] | None) -> None:
    if headers is None:
        raise ParsingError("The CSV file does not contain a header row.")
    stripped = [str(value) for value in headers]
    blank_positions = [str(index + 1) for index, name in enumerate(stripped) if not name.strip()]
    if blank_positions:
        raise ParsingError(f"Blank column names were found in header position(s) {', '.join(blank_positions)}. Add names in the source file and retry.")
    duplicates = sorted({name for name in stripped if stripped.count(name) > 1})
    if duplicates:
        names = ", ".join(repr(name) for name in duplicates)
        raise ParsingError(f"Duplicate column names were found ({names}). Rename them in the source file and upload again.")


def _validate_csv_structure(upload: BinaryIO, delimiter: str, encoding: str) -> None:
    """Stream rows once to reject inconsistent widths before Pandas can shift fields."""
    upload.seek(0)
    text_stream = io.TextIOWrapper(upload, encoding=encoding, newline="")
    try:
        reader = csv.reader(text_stream, delimiter=delimiter, strict=True)
        headers = next(reader, None)
        _check_duplicate_headers(headers)
        expected_fields = len(headers or [])
        for row in reader:
            if not row or (len(row) == 1 and row[0] == ""):
                continue
            if len(row) != expected_fields:
                raise ParsingError(
                    f"Malformed CSV near line {reader.line_num}: expected {expected_fields} fields but found {len(row)}. "
                    "Correct the row or export the file again; DataPilot will not skip it."
                )
    except csv.Error as exc:
        raise ParsingError(f"Malformed CSV near line {getattr(locals().get('reader'), 'line_num', '?')}: check quoting and delimiters. {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ParsingError("The CSV text encoding could not be decoded. Save it as UTF-8 or Windows-1252 and retry.") from exc
    finally:
        try:
            text_stream.detach()
        except ValueError:
            pass
        upload.seek(0)


class CSVParser:
    def parse(self, upload: BinaryIO, *, delimiter: str = "auto", **_: object) -> ParseResult:
        upload.seek(0)
        sample = upload.read(128 * 1024)
        upload.seek(0)
        try:
            text_sample = sample.decode("utf-8-sig")
            encoding = "utf-8-sig"
        except UnicodeDecodeError:
            encoding = "cp1252"
            text_sample = _decode_sample(sample)
        chosen = DELIMITER_OPTIONS.get(delimiter, delimiter if delimiter in {",", ";", "\t"} else None)
        if chosen is None:
            try:
                chosen = csv.Sniffer().sniff(text_sample, delimiters=",;\t").delimiter
            except csv.Error:
                chosen = ","
        if chosen not in {",", ";", "\t"}:
            raise ParsingError("Choose comma, semicolon, tab, or automatic delimiter detection.")
        _validate_csv_structure(upload, chosen, encoding)
        upload.seek(0)
        try:
            frame = pd.read_csv(upload, sep=chosen, encoding=encoding, on_bad_lines="error")
        except EmptyDataError as exc:
            raise ParsingError("The CSV file has no readable header or data rows.") from exc
        except (ParserError, UnicodeDecodeError, ValueError) as exc:
            raise ParsingError(
                "Could not parse the CSV. Check the selected delimiter, encoding, and that every row has a consistent number of fields."
            ) from exc
        except MemoryError as exc:
            raise ParsingError("This file needs more memory than the current app can provide. Reduce or split the file and retry.") from exc
        warning = []
        if encoding == "cp1252":
            warning.append("The file was read using Windows-1252 because it is not valid UTF-8.")
        return ParseResult(frame, warning, {"delimiter": "tab" if chosen == "\t" else chosen, "encoding": encoding})
