from __future__ import annotations

from typing import BinaryIO

import pandas as pd
from openpyxl import load_workbook

from src.upload.errors import ParsingError
from .base_parser import ParseResult


class ExcelParser:
    def sheet_names(self, upload: BinaryIO) -> list[str]:
        upload.seek(0)
        try:
            with pd.ExcelFile(upload, engine="openpyxl") as book:
                return list(book.sheet_names)
        except (ImportError, ModuleNotFoundError) as exc:
            raise ParsingError("Excel parsing requires openpyxl. Install the project dependencies and retry.") from exc
        except Exception as exc:
            raise ParsingError("This Excel workbook could not be opened. It may be corrupted or password-protected.") from exc
        finally:
            upload.seek(0)

    def _check_duplicate_headers(self, upload: BinaryIO, sheet_name: str) -> None:
        upload.seek(0)
        try:
            book = load_workbook(upload, read_only=True, data_only=True)
            try:
                if sheet_name not in book.sheetnames:
                    raise ParsingError(f"The workbook does not contain a sheet named ‘{sheet_name}’.")
                sheet = book[sheet_name]
                row = next(sheet.iter_rows(min_row=1, max_row=1, values_only=True), None)
                if row is None:
                    raise ParsingError(f"The selected sheet ‘{sheet_name}’ is empty.")
                headers = ["" if value is None else str(value) for value in row]
                last_header = max((index for index, value in enumerate(headers) if value.strip()), default=-1)
                headers = headers[: last_header + 1]
                if last_header < 0:
                    raise ParsingError(f"The selected sheet ‘{sheet_name}’ has no column headers.")
                blank_positions = [str(index + 1) for index, name in enumerate(headers) if not name.strip()]
                if blank_positions:
                    raise ParsingError(f"Blank column names were found in sheet ‘{sheet_name}’ at position(s) {', '.join(blank_positions)}.")
                duplicates = sorted({name for name in headers if headers.count(name) > 1})
                if duplicates:
                    joined = ", ".join(repr(name) for name in duplicates)
                    raise ParsingError(f"Duplicate column names were found in sheet ‘{sheet_name}’ ({joined}). Rename them in Excel and upload again.")
            finally:
                book.close()
        except ParsingError:
            raise
        except Exception as exc:
            raise ParsingError(f"Could not inspect the header row in sheet ‘{sheet_name}’.") from exc
        finally:
            upload.seek(0)

    def parse(self, upload: BinaryIO, *, sheet_name: str | None = None, **_: object) -> ParseResult:
        sheets = self.sheet_names(upload)
        if not sheets:
            raise ParsingError("This workbook does not contain any worksheets.")
        selected = sheet_name or sheets[0]
        if selected not in sheets:
            raise ParsingError(f"The workbook does not contain a sheet named ‘{selected}’.")
        self._check_duplicate_headers(upload, selected)
        upload.seek(0)
        try:
            frame = pd.read_excel(upload, sheet_name=selected, engine="openpyxl")
        except (ImportError, ModuleNotFoundError) as exc:
            raise ParsingError("Excel parsing requires openpyxl. Install the project dependencies and retry.") from exc
        except MemoryError as exc:
            raise ParsingError("This workbook needs more memory than the current app can provide. Reduce or split it and retry.") from exc
        except Exception as exc:
            raise ParsingError(f"Could not read sheet ‘{selected}’. Check that the workbook is not corrupted or password-protected.") from exc
        if frame.shape == (0, 0):
            raise ParsingError(f"The selected sheet ‘{selected}’ is empty or has no header row.")
        return ParseResult(frame, details={"sheet_name": selected, "sheet_names": sheets})
