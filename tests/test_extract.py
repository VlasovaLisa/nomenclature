import pytest

from core.extract import PipeSpec, extract_pipe_group, extract_spec


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Труба бесшовная группа 4 89х8 К52 ТТТ-01.02.04-01", PipeSpec("бесшовная", 89, 8, "К52")),
        ("Труба электросварная группа 4 1020х24 К52 ТТТ-01.02.04-01", PipeSpec("электросварная", 1020, 24, "К52")),
        ("Труба электросварная 114х5 ГОСТ 10704-91 В-09Г2С ГОСТ 10705-80", PipeSpec("электросварная", 114, 5, "09Г2С")),
        ("Труба электросварная 820х10 ГОСТ 10704-91 В-17Г1С-У ГОСТ 10706-76", PipeSpec("электросварная", 820, 10, "17Г1С-У")),
        ("Труба электросварная 114х6 ГОСТ 10704-91 В-20 ГОСТ 10705-80", PipeSpec("электросварная", 114, 6, "20")),
        ("Труба бесшовная 325х10-К52-13ХФА ТУ 14-3Р-124-2017", PipeSpec("бесшовная", 325, 10, "К52")),
        ("Труба бесшовная 57х6 ГОСТ 8732-78 В-09Г2С", PipeSpec("бесшовная", 57, 6, "09Г2С")),
        ("Труба обсадная ОТТМ-114х7,4-Е ГОСТ 632-80 А", PipeSpec("обсадная", 114, 7.4, "Е")),
        ("Труба обсадная Б-Д-168х7,3-Б ТУ 14-3Р-29-2007", PipeSpec("обсадная", 168, 7.3, "Д")),
        ("Труба обсадная Баттресс-Е-168х10,6-Б ТУ 14-3Р-29-2007", PipeSpec("обсадная", 168, 10.6, "Е")),
        ("Труба обсадная электросварная с резьбовым соединением Батресс Дс1-168х8,9", PipeSpec("обсадная", 168, 8.9, "Д")),
        ("Труба обсадная ВС-114х7,37-Мс ТУ 14-3Р-30-2015", PipeSpec("обсадная", 114, 7.37, "М")),
        ("Труба обсадная BC-177,8х9,19-R3-N80-EW-RC-PSL-1 ГОСТ 31446-2017", PipeSpec("обсадная", 177.8, 9.19, "N80")),
        ("Труба НКТ 73х5,5-Л ГОСТ 633-80 А", PipeSpec("НКТ", 73, 5.5, "Л")),
        ("Насосно-компрессорные трубы НКТВ 60 х 5.0 -30ХГМА-CS-Л", PipeSpec("НКТ", 60, 5, None)),
        ("Труба НКМ бесшовная высокогерметичная 114х7-К исполнение А", PipeSpec("НКМ", 114, 7, "К")),
    ],
)
def test_extract_from_catalog_names(text, expected):
    assert extract_spec(text) == expected


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("тр. бесш. гр.2 159х10 К60 KCU -60", PipeSpec("бесшовная", 159, 10, "К60")),
        ("труба б/ш 89*8 к52 гр 4", PipeSpec("бесшовная", 89, 8, "К52")),
        ("Труба бесшовная 114x8 K50 гр.2", PipeSpec("бесшовная", 114, 8, "К50")),
        ("Труба бесшовнпя группа 2 114х6 К52", PipeSpec("бесшовная", 114, 6, "К52")),
        ("Труба электросвраная гр. 4 1020х24 К52", PipeSpec("электросварная", 1020, 24, "К52")),
        ("Труба э/с 114х5 ст.09Г2С", PipeSpec("электросварная", 114, 5, "09Г2С")),
        ("обсадная труба ОТТМ 178х9.2 Л ГОСТ 632", PipeSpec("обсадная", 178, 9.2, "Л")),
        ("Труба обсадная 114х7,4 ОТТМ гр.пр. Е", PipeSpec("обсадная", 114, 7.4, "Е")),
        ("НКТ 73х5.5-Л ГОСТ 633-80", PipeSpec("НКТ", 73, 5.5, "Л")),
        ("Труба НКТ 48х4 Е с муфтой", PipeSpec("НКТ", 48, 4, "Е")),
    ],
)
def test_extract_from_client_queries(query, expected):
    assert extract_spec(query) == expected


def test_exclusion_clause_is_not_taken_as_steel_grade():
    spec = extract_spec("Труба бесшовная группа 2 219х8 (исключая марку стали 09Г2С, 09Г2) с покрытием")

    assert spec.steel_grade is None


def test_size_is_not_taken_as_steel_grade():
    assert extract_spec("Труба бесшовная 89х6 ГОСТ 8732").steel_grade is None


def test_missing_parameters_are_none():
    assert extract_spec("труба") == PipeSpec()


@pytest.mark.parametrize(("text", "expected"), [("Труба бесшовная группа 4 89х8 К52", 4), ("тр. гр.2 89х8", 2), ("Труба 89х8", None)])
def test_extract_pipe_group(text, expected):
    assert extract_pipe_group(text) == expected


def test_compare_ignores_unknown_fields():
    query = PipeSpec("бесшовная", 89, 8, None)
    candidate = PipeSpec("бесшовная", 89, 6, "К52")

    assert query.compare(candidate) == {"pipe_type": True, "diameter": True, "wall_thickness": False, "steel_grade": None}
    assert query.mismatches(candidate) == ["wall_thickness"]


def test_compare_returns_plain_booleans_for_numpy_values():
    import numpy as np

    comparison = PipeSpec("бесшовная", 89.0, 8.0).compare(PipeSpec("бесшовная", np.float64(89), np.float64(6)))

    assert comparison["diameter"] is True
    assert comparison["wall_thickness"] is False


def test_from_dict_normalizes_grade_and_numbers():
    spec = PipeSpec.from_dict({"pipe_type": "обсадная", "diameter": 114, "wall_thickness": 7.4, "steel_grade": "e"})

    assert spec == PipeSpec("обсадная", 114.0, 7.4, "Е")
