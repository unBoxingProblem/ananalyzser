import io

import pandas as pd
import pytest

from src.ingestion.csv_parser import CSVParser
from src.ingestion.excel_parser import ExcelParser
from src.ingestion.json_parser import JSONParser
from src.ingestion.parquet_parser import ParquetParser
from src.ingestion.tsv_parser import TSVParser
from src.upload.errors import ParsingError
from tests.helpers import Upload


def workbook_bytes() -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame({"product": ["A", "B"], "amount": [10, 20]}).to_excel(writer, sheet_name="Sales", index=False)
        pd.DataFrame({"region": ["North"]}).to_excel(writer, sheet_name="Regions", index=False)
    return buffer.getvalue()


def test_csv_ingestion():
    result = CSVParser().parse(Upload(b"name,amount\nA,1\nB,2\n", "a.csv"))
    assert result.dataframe.to_dict("records") == [{"name": "A", "amount": 1}, {"name": "B", "amount": 2}]


def test_csv_utf8_bom_and_windows_1252():
    bom = CSVParser().parse(Upload("name\nJos\u00e9\n".encode("utf-8-sig"), "bom.csv"))
    assert bom.dataframe.iloc[0, 0] == "Jos\u00e9"
    legacy = CSVParser().parse(Upload("name\nAndr\u00e9\n".encode("cp1252"), "legacy.csv"))
    assert legacy.dataframe.iloc[0, 0] == "Andr\u00e9"
    assert legacy.warnings


@pytest.mark.parametrize(("delimiter", "selected"), [(";", "semicolon (;)"), ("\t", "tab (\\t)")])
def test_csv_selectable_delimiters(delimiter, selected):
    result = CSVParser().parse(Upload(f"name{delimiter}amount\nA{delimiter}1\n".encode(), "data.csv"), delimiter=selected)
    assert list(result.dataframe.columns) == ["name", "amount"]


def test_tsv_ingestion():
    result = TSVParser().parse(Upload(b"name\tamount\nA\t2\n", "data.tsv"))
    assert result.dataframe.iloc[0].to_dict() == {"name": "A", "amount": 2}


def test_csv_malformed_rows_raise_instead_of_skipping():
    with pytest.raises(ParsingError, match="expected 2 fields"):
        CSVParser().parse(Upload(b"a,b\n1,2,3\n", "bad.csv"))


def test_csv_duplicate_headers_are_reported():
    with pytest.raises(ParsingError, match="Duplicate column names"):
        CSVParser().parse(Upload(b"a,a\n1,2\n", "dupes.csv"))


def test_csv_blank_header_is_reported():
    with pytest.raises(ParsingError, match="Blank column names"):
        CSVParser().parse(Upload(b"a,\n1,2\n", "blank-header.csv"))


def test_excel_multiple_sheets_select_one():
    upload = Upload(workbook_bytes(), "book.xlsx")
    parser = ExcelParser()
    assert parser.sheet_names(upload) == ["Sales", "Regions"]
    result = parser.parse(upload, sheet_name="Regions")
    assert result.dataframe.to_dict("records") == [{"region": "North"}]


def test_excel_missing_sheet_is_clear():
    with pytest.raises(ParsingError, match="does not contain a sheet"):
        ExcelParser().parse(Upload(workbook_bytes(), "book.xlsx"), sheet_name="Missing")


def test_excel_duplicate_headers_are_reported():
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame([["A", "B"], [1, 2]], columns=["x", "x"]).to_excel(writer, index=False)
    with pytest.raises(ParsingError, match="Duplicate column names"):
        ExcelParser().parse(Upload(buffer.getvalue(), "dupes.xlsx"))


def test_corrupted_excel_is_clear():
    with pytest.raises(ParsingError, match="could not be opened"):
        ExcelParser().sheet_names(Upload(b"bad", "corrupt.xlsx"))


def test_json_array_of_objects():
    result = JSONParser().parse(Upload(b'[{"name":"A","count":2},{"name":"B","count":3}]', "rows.json"))
    assert result.dataframe.to_dict("records") == [{"name": "A", "count": 2}, {"name": "B", "count": 3}]


def test_json_record_oriented_and_json_lines():
    oriented = JSONParser().parse(Upload(b'{"r1":{"name":"A"},"r2":{"name":"B"}}', "records.json"))
    assert oriented.dataframe["_record_id"].tolist() == ["r1", "r2"]
    lines = JSONParser().parse(Upload(b'{"a":1}\n{"a":2}\n', "lines.json"))
    assert lines.dataframe["a"].tolist() == [1, 2]
    assert "JSON Lines" in lines.warnings[0]


def test_invalid_and_nested_json_are_explained():
    with pytest.raises(ParsingError, match="Invalid JSON"):
        JSONParser().parse(Upload(b'{"a":}\n', "invalid.json"))
    with pytest.raises(ParsingError, match="nested data"):
        JSONParser().parse(Upload(b'[{"a":{"b":1}}]', "nested.json"))
    with pytest.raises(ParsingError, match="repeats field"):
        JSONParser().parse(Upload(b'{"a":1,"a":2}', "duplicate-key.json"))
    with pytest.raises(ParsingError, match="nested data"):
        JSONParser().parse(Upload(b'{"a":{"b":1}}\n{"c":2}\n', "nested-lines.json"))


def test_parquet_ingestion_and_corrupt_data():
    buffer = io.BytesIO()
    pd.DataFrame({"x": [1, 2]}).to_parquet(buffer, engine="pyarrow")
    result = ParquetParser().parse(Upload(buffer.getvalue(), "data.parquet"))
    assert result.dataframe["x"].tolist() == [1, 2]
    with pytest.raises(ParsingError, match="could not be read"):
        ParquetParser().parse(Upload(b"PAR1bad!", "broken.parquet"))
