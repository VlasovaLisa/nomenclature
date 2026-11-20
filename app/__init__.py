from pathlib import Path

import pandas as pd
from flask import Flask

from core.catalog import prepare_catalog
from core.config_builder import format_size
from core.semantic_search import DEFAULT_MODEL, MODELS, EmbeddingCache, Encoder, SemanticSearcher, load_encoder
from core.service import NomenclatureService

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = {
    "CATALOG_PATH": PROJECT_ROOT / "data" / "processed" / "catalog.parquet",
    "SOURCE_DIR": PROJECT_ROOT / "data",
    "EMBEDDINGS_PATH": PROJECT_ROOT / "data" / "processed" / f"embeddings_{DEFAULT_MODEL}_norm.npz",
    "MODEL": DEFAULT_MODEL,
    "MAX_CONTENT_LENGTH": 20 * 1024 * 1024,
}


def load_startup_catalog(catalog_path: Path, source_dir: Path) -> pd.DataFrame:
    if catalog_path.exists():
        return pd.read_parquet(catalog_path)
    sources = sorted(source_dir.glob("*.xlsx"))
    if not sources:
        raise FileNotFoundError(f"Нет ни {catalog_path}, ни XLSX-справочников в {source_dir}")
    return prepare_catalog(sources)


def create_app(config: dict | None = None, encoder: Encoder | None = None) -> Flask:
    app = Flask(__name__)
    app.config.update(DEFAULT_CONFIG)
    app.config.update(config or {})

    model_config = MODELS[app.config["MODEL"]]
    encoder = encoder or load_encoder(model_config)
    cache = EmbeddingCache(Path(app.config["EMBEDDINGS_PATH"]))

    def build_searcher(catalog: pd.DataFrame) -> SemanticSearcher:
        return SemanticSearcher(catalog["name"], encoder, model_config, cache=cache)

    catalog = load_startup_catalog(Path(app.config["CATALOG_PATH"]), Path(app.config["SOURCE_DIR"]))
    app.extensions["nomenclature"] = NomenclatureService(catalog, build_searcher)
    app.add_template_filter(format_size, "size")

    from app.routes import blueprint

    app.register_blueprint(blueprint)
    return app
