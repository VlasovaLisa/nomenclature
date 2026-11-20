import time
from io import BytesIO

from flask import Blueprint, current_app, render_template, request

from core.loader import CatalogFormatError
from core.catalog import prepare_catalog
from core.service import NomenclatureService

blueprint = Blueprint("main", __name__)

ALLOWED_EXTENSION = ".xlsx"
SPEC_LABELS = {
    "pipe_type": "Тип трубы",
    "diameter": "Диаметр, мм",
    "wall_thickness": "Стенка, мм",
    "steel_grade": "Марка / класс",
}


def service() -> NomenclatureService:
    return current_app.extensions["nomenclature"]


def render_page(status: int = 200, **context):
    return render_template("index.html", service=service(), spec_labels=SPEC_LABELS, **context), status


@blueprint.get("/")
def index():
    return render_page()


@blueprint.get("/search")
def search():
    query = request.args.get("q", "").strip()
    if not query:
        return render_page(400, error="Введите наименование продукции.")
    started = time.perf_counter()
    response = service().search(query)
    seconds = time.perf_counter() - started
    return render_page(query=query, response=response, seconds=seconds)


@blueprint.post("/upload")
def upload():
    files = [file for file in request.files.getlist("catalog") if file.filename]
    if not files:
        return render_page(400, error="Выберите хотя бы один XLSX-файл.")
    wrong_files = [file.filename for file in files if not file.filename.lower().endswith(ALLOWED_EXTENSION)]
    if wrong_files:
        return render_page(400, error=f"Поддерживаются только файлы {ALLOWED_EXTENSION}: {', '.join(wrong_files)}")
    try:
        # SpooledTemporaryFile has no seekable() before Python 3.11, which pandas requires.
        catalog = prepare_catalog(BytesIO(file.read()) for file in files)
    except CatalogFormatError as error:
        return render_page(400, error=f"Не удалось прочитать справочник: {error}")
    except (ValueError, OSError):
        return render_page(400, error="Не удалось прочитать файл. Проверьте, что это корректный XLSX.")
    if catalog.empty:
        return render_page(400, error="В файлах не найдено ни одной трубы.")
    service().replace_catalog(catalog)
    return render_page(message=f"Справочник загружен: {len(files)} файл(ов), {service().pair_count} уникальных пар.")
