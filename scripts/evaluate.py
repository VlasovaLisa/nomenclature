import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import time  # noqa: E402

import pandas as pd  # noqa: E402

from core.evaluation import EXISTING, NEW_COMBINATION, SPEC_FIELDS, EvaluationReport, evaluate, load_queries  # noqa: E402
from core.ranker import HybridRanker, build_catalog_index  # noqa: E402
from core.search import Searcher  # noqa: E402
from core.semantic_search import DEFAULT_MODEL, MODELS, EmbeddingCache, SemanticSearcher, load_encoder  # noqa: E402
from core.tfidf_search import TfidfSearcher  # noqa: E402

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CATALOG_PATH = PROCESSED_DIR / "catalog.parquet"
QUERY_SETS = {
    "test": PROJECT_ROOT / "data" / "test" / "test_queries.json",
    "dev": PROJECT_ROOT / "data" / "test" / "dev_queries.json",
}
ALGORITHMS = ["tfidf", "semantic", "hybrid"]


def build_semantic_searcher(names: pd.Series, model_key: str, normalize_text: bool) -> SemanticSearcher:
    config = MODELS[model_key]
    text_mode = "norm" if normalize_text else "raw"
    cache = EmbeddingCache(PROCESSED_DIR / f"embeddings_{model_key}_{text_mode}.npz")
    return SemanticSearcher(names, load_encoder(config), config, normalize_text=normalize_text, cache=cache)


def build_searcher(args: argparse.Namespace, catalog: pd.DataFrame) -> Searcher:
    normalize_text = not args.raw_text
    if args.algo == "tfidf":
        return TfidfSearcher(catalog["name"])
    if args.algo == "semantic":
        return build_semantic_searcher(catalog["name"], args.model, normalize_text)
    candidate_searchers = {
        "semantic": lambda: build_semantic_searcher(catalog["name"], args.model, normalize_text),
        "tfidf": lambda: TfidfSearcher(catalog["name"]),
    }
    return HybridRanker(candidate_searchers[args.candidates](), build_catalog_index(catalog))


def describe(args: argparse.Namespace) -> str:
    text_mode = "raw" if args.raw_text else "norm"
    if args.algo == "tfidf":
        return "tfidf"
    if args.algo == "semantic":
        return f"semantic ({args.model}, {text_mode})"
    source = {"tfidf": "tfidf", "semantic": f"{args.model}, {text_mode}"}[args.candidates]
    return f"hybrid (кандидаты: {source})"


def print_report(report: EvaluationReport, verbose: bool) -> None:
    if verbose:
        for result in report.results:
            mark = "·" if not result.is_scored else (result.rank or "-")
            print(f"[{mark}] {result.query_id} {result.query}")
            wrong_fields = sorted(set(SPEC_FIELDS) - set(result.correctly_extracted_fields))
            if wrong_fields:
                print(f"      извлечение: ошибка в {wrong_fields}: {result.extracted_spec}")
            for position, hit in enumerate(result.hits, start=1):
                label = "сгенерирована" if hit.is_generated else f"{hit.score:.3f}"
                print(f"      {position}. {label}  {hit.name[:110]}")
            if result.top_hit_mismatches:
                print(f"      top-1 не совпадает по: {result.top_hit_mismatches}")
        print()

    print(f"Запросов с эталонной позицией: {len(report.scored)}")
    print(f"Hit Rate@3: {report.hit_rate(3):.0%}")
    print(f"Hit Rate@1: {report.hit_rate(1):.0%}")
    print(f"MRR@3:      {report.mean_reciprocal_rank:.3f}")
    print(f"Точность извлечения 4 параметров: {report.extraction_accuracy:.0%}")
    print(f"Top-1 с несовпадающими параметрами: существующие {report.top_hit_mismatch_rate(EXISTING):.0%}, "
          f"новые комбинации {report.top_hit_mismatch_rate(NEW_COMBINATION):.0%}")
    print(f"Сгенерированы конфигурации: существующие {report.generation_rate(EXISTING):.0%}, "
          f"новые комбинации {report.generation_rate(NEW_COMBINATION):.0%}; "
          f"конфигурация сохранила все 4 параметра: {report.preserved_config_rate:.0%} новых комбинаций")
    print(f"Время поиска: среднее {report.mean_seconds * 1000:.1f} мс, максимум {report.max_seconds * 1000:.1f} мс")


def main() -> None:
    parser = argparse.ArgumentParser(description="Оценка качества поиска на фиксированной выборке")
    parser.add_argument("--algo", choices=ALGORITHMS, default="tfidf")
    parser.add_argument("--model", choices=MODELS, default=DEFAULT_MODEL, help="модель для semantic")
    parser.add_argument("--raw-text", action="store_true", help="semantic: кодировать текст без нормализации")
    parser.add_argument("--candidates", choices=["semantic", "tfidf"], default="semantic", help="hybrid: источник кандидатов")
    parser.add_argument("--queries", choices=QUERY_SETS, default="test")
    parser.add_argument("-v", "--verbose", action="store_true", help="показать top-3 для каждого запроса")
    args = parser.parse_args()

    catalog = pd.read_parquet(CATALOG_PATH)
    started = time.perf_counter()
    searcher = build_searcher(args, catalog)
    setup_seconds = time.perf_counter() - started
    queries = load_queries(QUERY_SETS[args.queries])

    print(f"Алгоритм: {describe(args)}, выборка: {args.queries} ({len(queries)} запросов)")
    print(f"Подготовка (загрузка модели и эмбеддингов): {setup_seconds:.1f} с")
    print_report(evaluate(searcher, queries), args.verbose)


if __name__ == "__main__":
    main()
