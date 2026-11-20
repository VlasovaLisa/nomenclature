import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from core.semantic_search import DEFAULT_MODEL, MODELS, EmbeddingCache, SemanticSearcher, load_encoder  # noqa: E402

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def embeddings_path(model_key: str) -> Path:
    return PROCESSED_DIR / f"embeddings_{model_key}_norm.npz"


def main() -> None:
    parser = argparse.ArgumentParser(description="Считает и сохраняет эмбеддинги наименований справочника")
    parser.add_argument("--model", choices=MODELS, default=DEFAULT_MODEL)
    args = parser.parse_args()

    names = pd.read_parquet(PROCESSED_DIR / "catalog.parquet")["name"]
    config = MODELS[args.model]

    started = time.perf_counter()
    encoder = load_encoder(config)
    loaded = time.perf_counter()
    SemanticSearcher(names, encoder, config, cache=EmbeddingCache(embeddings_path(args.model)))
    finished = time.perf_counter()

    print(f"Модель: {config.model_id}")
    print(f"Загрузка модели: {loaded - started:.1f} с, эмбеддинги {names.nunique()} наименований: {finished - loaded:.1f} с")
    print(f"Сохранено: {embeddings_path(args.model)}")


if __name__ == "__main__":
    main()
