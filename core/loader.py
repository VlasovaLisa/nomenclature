import re
from collections.abc import Iterable
from pathlib import Path
from typing import BinaryIO

import pandas as pd

NAME_COLUMN = "Полное наименование МТР"
REPLACEMENTS_COLUMN = "Замены"
CATALOG_COLUMNS = ["item_id", "name", "replacements_raw", "replacements"]

PIPE_NAME_PREFIX = "Труб"
REPLACEMENT_SEPARATOR = re.compile(r"\s+или\s+(?=Труб[аы]\s)|(?<=\S)(?=Труб[аы]\s)")
EDGE_NOISE = re.compile(r"^['\"\s]+|[\s,;]+$")

ExcelSource = str | Path | BinaryIO


class CatalogFormatError(ValueError):
    pass


def read_catalog_records(source: ExcelSource) -> pd.DataFrame:
    sheets = pd.read_excel(source, sheet_name=None, dtype=str)
    catalog_sheets = [
        sheet[[NAME_COLUMN, REPLACEMENTS_COLUMN]]
        for sheet in sheets.values()
        if {NAME_COLUMN, REPLACEMENTS_COLUMN} <= set(sheet.columns)
    ]
    if not catalog_sheets:
        raise CatalogFormatError(
            f"Не найден лист со столбцами «{NAME_COLUMN}» и «{REPLACEMENTS_COLUMN}»"
        )
    records = pd.concat(catalog_sheets, ignore_index=True)
    records.columns = ["name", "replacements_raw"]
    return records


def clean_records(records: pd.DataFrame) -> pd.DataFrame:
    cleaned = records.dropna(subset=["name"]).copy()
    cleaned["name"] = cleaned["name"].str.strip()
    cleaned["replacements_raw"] = cleaned["replacements_raw"].fillna("").str.strip()
    is_pipe = cleaned["name"].str.startswith(PIPE_NAME_PREFIX)
    return cleaned[is_pipe].reset_index(drop=True)


def split_replacements(text: str) -> list[str]:
    variants = (EDGE_NOISE.sub("", part) for part in REPLACEMENT_SEPARATOR.split(text))
    return [variant for variant in variants if variant]


def build_catalog(records: pd.DataFrame) -> pd.DataFrame:
    catalog = records.drop_duplicates(subset=["name", "replacements_raw"]).reset_index(drop=True)
    catalog.insert(0, "item_id", range(len(catalog)))
    catalog["replacements"] = catalog["replacements_raw"].map(split_replacements)
    return catalog[CATALOG_COLUMNS]


def load_records(sources: Iterable[ExcelSource]) -> pd.DataFrame:
    return clean_records(pd.concat([read_catalog_records(source) for source in sources], ignore_index=True))


def load_catalog(sources: Iterable[ExcelSource]) -> pd.DataFrame:
    return build_catalog(load_records(sources))
