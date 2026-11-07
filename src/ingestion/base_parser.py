from __future__ import annotations

from dataclasses import dataclass, field
from typing import BinaryIO, Protocol

import pandas as pd


@dataclass
class ParseResult:
    dataframe: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    details: dict[str, object] = field(default_factory=dict)


class FileParser(Protocol):
    def parse(self, upload: BinaryIO, **options: object) -> ParseResult: ...
