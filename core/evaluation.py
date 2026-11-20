import json
import time
from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path

from core.extract import PipeSpec, extract_spec
from core.search import SearchHit, Searcher

EXISTING = "existing"
NEW_COMBINATION = "new_combination"
SPEC_FIELDS = [field.name for field in fields(PipeSpec)]


@dataclass(frozen=True)
class QueryResult:
    query_id: str
    query: str
    kind: str
    hits: list[SearchHit]
    rank: int | None
    seconds: float
    expected_spec: PipeSpec
    extracted_spec: PipeSpec
    top_hit_mismatches: list[str]

    @property
    def is_scored(self) -> bool:
        return self.kind == EXISTING

    @property
    def generated_hit(self) -> SearchHit | None:
        return next((hit for hit in self.hits if hit.is_generated), None)

    def generated_config_preserves_spec(self, extract: Callable[[str], PipeSpec] = extract_spec) -> bool:
        return self.generated_hit is not None and extract(self.generated_hit.name) == self.expected_spec

    @property
    def correctly_extracted_fields(self) -> list[str]:
        return [name for name in SPEC_FIELDS if getattr(self.extracted_spec, name) == getattr(self.expected_spec, name)]


@dataclass(frozen=True)
class EvaluationReport:
    results: list[QueryResult]

    @property
    def scored(self) -> list[QueryResult]:
        return [result for result in self.results if result.is_scored]

    def hit_rate(self, k: int) -> float:
        return sum(result.rank is not None and result.rank <= k for result in self.scored) / len(self.scored)

    @property
    def mean_reciprocal_rank(self) -> float:
        return sum(1 / result.rank for result in self.scored if result.rank) / len(self.scored)

    @property
    def extraction_accuracy(self) -> float:
        correct = sum(len(result.correctly_extracted_fields) for result in self.results)
        return correct / (len(SPEC_FIELDS) * len(self.results))

    def _of_kind(self, kind: str) -> list[QueryResult]:
        return [result for result in self.results if result.kind == kind]

    def top_hit_mismatch_rate(self, kind: str) -> float:
        selected = self._of_kind(kind)
        return sum(bool(result.top_hit_mismatches) for result in selected) / len(selected)

    def generation_rate(self, kind: str) -> float:
        selected = self._of_kind(kind)
        return sum(result.generated_hit is not None for result in selected) / len(selected)

    @property
    def preserved_config_rate(self) -> float:
        selected = self._of_kind(NEW_COMBINATION)
        return sum(result.generated_config_preserves_spec() for result in selected) / len(selected)

    @property
    def mean_seconds(self) -> float:
        return sum(result.seconds for result in self.results) / len(self.results)

    @property
    def max_seconds(self) -> float:
        return max(result.seconds for result in self.results)


def load_queries(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def first_expected_rank(hits: list[SearchHit], expected_names: list[str]) -> int | None:
    expected = set(expected_names)
    return next((position for position, hit in enumerate(hits, start=1) if hit.name in expected), None)


def evaluate(
    searcher: Searcher,
    queries: list[dict],
    k: int = 3,
    extract: Callable[[str], PipeSpec] = extract_spec,
) -> EvaluationReport:
    results = []
    for query in queries:
        started = time.perf_counter()
        hits = searcher.search(query["query"], k)
        seconds = time.perf_counter() - started
        expected_spec = PipeSpec.from_dict(query["expected_spec"])
        results.append(
            QueryResult(
                query_id=query["id"],
                query=query["query"],
                kind=query["kind"],
                hits=hits,
                rank=first_expected_rank(hits, query["expected_names"]),
                seconds=seconds,
                expected_spec=expected_spec,
                extracted_spec=extract(query["query"]),
                top_hit_mismatches=expected_spec.mismatches(extract(hits[0].name)) if hits else [],
            )
        )
    return EvaluationReport(results)
