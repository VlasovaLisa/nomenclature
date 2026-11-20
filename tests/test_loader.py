from pathlib import Path

import pandas as pd
import pytest

from core.loader import (
    NAME_COLUMN,
    REPLACEMENTS_COLUMN,
    CatalogFormatError,
    build_catalog,
    clean_records,
    load_catalog,
    load_records,
    read_catalog_records,
    split_replacements,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SOURCE_FILES = [DATA_DIR / "2022год_НГП.xlsx", DATA_DIR / "2023год_НГП.xlsx"]


def write_workbook(path: Path, sheets: dict[str, pd.DataFrame]) -> Path:
    with pd.ExcelWriter(path) as writer:
        for sheet_name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
    return path


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Труба бесшовная И3-168х12-К48-ТУ 14-3Р-167-2019 или Труба бесшовная 168х12-К48-3-ТУ 24.20.13.110-128-00186654-2019",
            ["Труба бесшовная И3-168х12-К48-ТУ 14-3Р-167-2019", "Труба бесшовная 168х12-К48-3-ТУ 24.20.13.110-128-00186654-2019"],
        ),
        (
            "Труба электросварная И2-720х8-К52 ТУ 24.20.21-016-53570464-2020Труба электросварная 720х8-К52-2 ТУ 24.20.21.000-132-00186654-2019, ",
            ["Труба электросварная И2-720х8-К52 ТУ 24.20.21-016-53570464-2020", "Труба электросварная 720х8-К52-2 ТУ 24.20.21.000-132-00186654-2019"],
        ),
        (
            "Труба электросварная 1020х10 09Г2С-В ГОСТ 10706-76, ГОСТ 10704-91, , ",
            ["Труба электросварная 1020х10 09Г2С-В ГОСТ 10706-76, ГОСТ 10704-91"],
        ),
        (
            "Труба обсадная ОТТМ-114х7,4-Е ГОСТ 632-80 А или эквивалент",
            ["Труба обсадная ОТТМ-114х7,4-Е ГОСТ 632-80 А или эквивалент"],
        ),
        (
            "Трубы бесшовные 114х11-К52-4 с покрытием ТУ 1390-008-74747996-2012 или ТУ 1390-004-30098597-2016",
            ["Трубы бесшовные 114х11-К52-4 с покрытием ТУ 1390-008-74747996-2012 или ТУ 1390-004-30098597-2016"],
        ),
        ("'Труба НКТ 89х6,5-Л ГОСТ 633-80 А", ["Труба НКТ 89х6,5-Л ГОСТ 633-80 А"]),
        ("", []),
    ],
)
def test_split_replacements(text, expected):
    assert split_replacements(text) == expected


def test_clean_records_drops_empty_and_non_pipe_names():
    records = pd.DataFrame(
        {
            "name": ["  Труба бесшовная 89х8 К52  ", None, "2026-09-14 00:00:00"],
            "replacements_raw": [None, "Труба", None],
        }
    )

    cleaned = clean_records(records)

    assert cleaned.to_dict("records") == [{"name": "Труба бесшовная 89х8 К52", "replacements_raw": ""}]


def test_build_catalog_keeps_distinct_replacements_for_same_name():
    records = pd.DataFrame(
        {
            "name": ["Труба А", "Труба А", "Труба А", "Труба Б"],
            "replacements_raw": ["Труба X", "Труба X", "Труба Y", "Труба X"],
        }
    )

    catalog = build_catalog(records)

    assert catalog[["name", "replacements_raw"]].values.tolist() == [
        ["Труба А", "Труба X"],
        ["Труба А", "Труба Y"],
        ["Труба Б", "Труба X"],
    ]
    assert catalog["item_id"].tolist() == [0, 1, 2]
    assert catalog["item_id"].is_unique


def test_read_catalog_records_skips_sheets_without_catalog_columns(tmp_path):
    catalog_sheet = pd.DataFrame({NAME_COLUMN: ["Труба А"], REPLACEMENTS_COLUMN: ["Труба X"]})
    price_sheet = pd.DataFrame({"Цена": [100]})
    workbook = write_workbook(tmp_path / "book.xlsx", {"НГП": catalog_sheet, "Расчет НМЦ": price_sheet})

    records = read_catalog_records(workbook)

    assert records.to_dict("records") == [{"name": "Труба А", "replacements_raw": "Труба X"}]


def test_read_catalog_records_rejects_workbook_without_catalog_columns(tmp_path):
    workbook = write_workbook(tmp_path / "book.xlsx", {"Лист1": pd.DataFrame({"Цена": [100]})})

    with pytest.raises(CatalogFormatError):
        read_catalog_records(workbook)


def test_read_catalog_records_accepts_file_object(tmp_path):
    catalog_sheet = pd.DataFrame({NAME_COLUMN: ["Труба А"], REPLACEMENTS_COLUMN: ["Труба X"]})
    workbook = write_workbook(tmp_path / "book.xlsx", {"НГП": catalog_sheet})

    with workbook.open("rb") as file:
        records = read_catalog_records(file)

    assert len(records) == 1


@pytest.mark.skipif(not all(path.exists() for path in SOURCE_FILES), reason="исходные справочники не найдены")
class TestSourceCatalogs:
    def test_record_count_matches_spec(self):
        assert len(load_records(SOURCE_FILES)) == 1794

    def test_unique_pair_count_matches_spec(self):
        assert len(load_catalog(SOURCE_FILES)) == 701

    def test_every_pair_has_replacement_variants(self):
        catalog = load_catalog(SOURCE_FILES)

        assert catalog["replacements"].map(len).gt(0).all()

    def test_glued_replacements_are_split(self):
        catalog = load_catalog(SOURCE_FILES)

        assert not catalog["replacements"].explode().str.contains(r"\dТруб").any()

    def test_no_variant_is_a_fragment_of_a_note(self):
        variants = load_catalog(SOURCE_FILES)["replacements"].explode()

        assert variants.str.match(r"Труб|Насосно-компрессорн").all()
