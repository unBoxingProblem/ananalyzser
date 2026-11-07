from __future__ import annotations

from typing import BinaryIO

import pandas as pd

from src.upload.errors import ParsingError
from .base_parser import ParseResult


class ParquetParser:
    def parse(self, upload: BinaryIO, **_: object) -> ParseResult:
        upload.seek(0)
        try:
            frame = pd.read_parquet(upload, engine="pyarrow")
        except (ImportError, ModuleNotFoundError) as exc:
            raise ParsingError("Parquet parsing requires PyArrow. Install the project dependencies and retry.") from exc
        except MemoryError as exc:
            raise ParsingError("This Parquet file needs more memory than the current app can provide. Reduce or split it and retry.") from exc
        except Exception as exc:
            raise ParsingError("This Parquet file could not be read. It may be corrupted or use an unsupported encoding.") from exc
        return ParseResult(frame)
