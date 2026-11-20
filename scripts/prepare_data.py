import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.catalog import SPEC_COLUMNS, add_specs  # noqa: E402
from core.loader import build_catalog, load_records  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUTS = sorted((PROJECT_ROOT / "data").glob("*.xlsx"))
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "catalog.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description="Объединяет справочники МТР и сохраняет уникальные пары «Наименование – Замены»")
    parser.add_argument("inputs", nargs="*", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.inputs:
        parser.error("не найдено ни одного XLSX-файла")

    records = load_records(args.inputs)
    catalog = add_specs(build_catalog(records))
    unique_names = catalog.drop_duplicates("name")
    fully_extracted = unique_names[SPEC_COLUMNS].notna().all(axis=1).sum()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_parquet(args.output, index=False)

    print(f"Файлов: {len(args.inputs)}")
    print(f"Записей после очистки: {len(records)}")
    print(f"Уникальных пар «Наименование – Замены»: {len(catalog)}")
    print(f"Уникальных наименований: {len(unique_names)}, из них все 4 параметра извлечены у {fully_extracted}")
    print(f"Сохранено: {args.output}")


if __name__ == "__main__":
    main()
