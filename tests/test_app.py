from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app import create_app
from core.loader import NAME_COLUMN, REPLACEMENTS_COLUMN


class ConstantEncoder:
    def encode(self, sentences, *, normalize_embeddings, convert_to_numpy):
        return np.ones((len(sentences), 4)) / 2


CATALOG_ROWS = {
    NAME_COLUMN: [
        "Труба бесшовная группа 4 89х8 К52 ТТТ-01.02.04-01",
        "Труба электросварная группа 2 219х8 К48 ТТТ-01.02.04-01",
    ],
    REPLACEMENTS_COLUMN: [
        "Труба бесшовная И4-89х8-К52 ТУ 14-3Р-167-2019 или Труба бесшовная 89х8-К52-4",
        "Труба электросварная И2-219х8-К48 ТУ 24.20.21",
    ],
}


def workbook_bytes(sheets: dict[str, pd.DataFrame]) -> BytesIO:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer) as writer:
        for sheet_name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
    buffer.seek(0)
    return buffer


@pytest.fixture
def client(tmp_path: Path):
    source_dir = tmp_path / "data"
    source_dir.mkdir()
    (source_dir / "catalog.xlsx").write_bytes(workbook_bytes({"НГП": pd.DataFrame(CATALOG_ROWS)}).getvalue())
    app = create_app(
        {
            "TESTING": True,
            "CATALOG_PATH": tmp_path / "missing.parquet",
            "SOURCE_DIR": source_dir,
            "EMBEDDINGS_PATH": tmp_path / "embeddings.npz",
        },
        encoder=ConstantEncoder(),
    )
    return app.test_client()


def page_text(response) -> str:
    return response.get_data(as_text=True)


def test_index_shows_search_and_upload_forms(client):
    response = client.get("/")

    assert response.status_code == 200
    text = page_text(response)
    assert 'name="q"' in text
    assert 'name="catalog"' in text
    assert "В справочнике 2 наименований" in text


def test_search_shows_exact_match_and_replacements(client):
    response = client.get("/search", query_string={"q": "труба б/ш 89*8 К52 гр.4"})

    text = page_text(response)
    assert response.status_code == 200
    assert "Труба бесшовная группа 4 89х8 К52 ТТТ-01.02.04-01" in text
    assert "Совпадают все параметры запроса" in text
    assert "Труба бесшовная И4-89х8-К52 ТУ 14-3Р-167-2019" in text
    assert "Точного совпадения в справочнике не найдено" not in text


def test_search_marks_generated_configuration_and_missing_exact_match(client):
    text = page_text(client.get("/search", query_string={"q": "Труба бесшовная группа 2 219х10 К48"}))

    assert "Точного совпадения в справочнике не найдено" in text
    assert "Труба бесшовная группа 2 219х10 К48 ТТТ-01.02.04-01" in text
    assert "Предполагаемая конфигурация — требует проверки" in text


def test_search_highlights_mismatched_parameters(client):
    text = page_text(client.get("/search", query_string={"q": "Труба бесшовная 219х8 К48"}))

    assert 'class="param mismatch"' in text


def test_empty_query_is_rejected(client):
    response = client.get("/search", query_string={"q": "  "})

    assert response.status_code == 400
    assert "Введите наименование продукции" in page_text(response)


def test_upload_replaces_catalog(client):
    new_catalog = pd.DataFrame({NAME_COLUMN: ["Труба обсадная ОТТМ-114х7,4-Е ГОСТ 632-80 А"], REPLACEMENTS_COLUMN: ["Труба обсадная ОТТМ-114х7,4-Е"]})

    response = client.post(
        "/upload",
        data={"catalog": (workbook_bytes({"ОБС, НКТ": new_catalog}), "new.xlsx")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert "Справочник загружен: 1 файл(ов), 1 уникальных пар" in page_text(response)
    search_text = page_text(client.get("/search", query_string={"q": "обсадная 114х7,4 Е"}))
    assert "Труба обсадная ОТТМ-114х7,4-Е ГОСТ 632-80 А" in search_text


def test_upload_rejects_non_xlsx(client):
    response = client.post("/upload", data={"catalog": (BytesIO(b"text"), "notes.txt")}, content_type="multipart/form-data")

    assert response.status_code == 400
    assert "Поддерживаются только файлы .xlsx" in page_text(response)


def test_upload_rejects_workbook_without_catalog_columns(client):
    workbook = workbook_bytes({"Лист1": pd.DataFrame({"Цена": [1]})})

    response = client.post("/upload", data={"catalog": (workbook, "prices.xlsx")}, content_type="multipart/form-data")

    assert response.status_code == 400
    assert "Не удалось прочитать справочник" in page_text(response)
    assert "В справочнике 2 наименований" in page_text(response)


def test_upload_rejects_corrupted_file(client):
    response = client.post("/upload", data={"catalog": (BytesIO(b"not a zip"), "broken.xlsx")}, content_type="multipart/form-data")

    assert response.status_code == 400
    assert "Не удалось прочитать файл" in page_text(response)
