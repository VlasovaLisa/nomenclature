from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import pandas as pd

from core.catalog import SPEC_COLUMNS
from core.config_builder import build_name, supports_template
from core.extract import PipeSpec, extract_pipe_group, extract_spec
from core.search import SearchHit, Searcher

CANDIDATE_COUNT = 30


@dataclass(frozen=True)
class CatalogEntry:
    name: str
    spec: PipeSpec
    group: int | None
    replacements: tuple[str, ...]


@dataclass(frozen=True)
class RankedResult:
    name: str
    spec: PipeSpec
    comparison: dict[str, bool | None]
    replacements: tuple[str, ...] = ()
    semantic_score: float | None = None
    is_exact: bool = False
    is_generated: bool = False
    assumed_group: bool = False

    @property
    def mismatches(self) -> list[str]:
        return [name for name, matched in self.comparison.items() if matched is False]


@dataclass(frozen=True)
class SearchResponse:
    query_spec: PipeSpec
    results: list[RankedResult] = field(default_factory=list)

    @property
    def has_exact_match(self) -> bool:
        return any(result.is_exact for result in self.results)


def _optional(value):
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def build_catalog_index(catalog: pd.DataFrame) -> dict[str, CatalogEntry]:
    index = {}
    for name, rows in catalog.groupby("name", sort=False):
        first = rows.iloc[0]
        spec = PipeSpec(**{column: _optional(first[column]) for column in SPEC_COLUMNS})
        replacements = tuple(dict.fromkeys(variant for variants in rows["replacements"] for variant in variants))
        group = _optional(first["pipe_group"])
        index[name] = CatalogEntry(name, spec, None if group is None else int(group), replacements)
    return index


class HybridRanker:
    def __init__(
        self,
        searcher: Searcher,
        catalog_index: Mapping[str, CatalogEntry],
        extract: Callable[[str], PipeSpec] = extract_spec,
        candidate_count: int = CANDIDATE_COUNT,
    ):
        self._searcher = searcher
        self._index = catalog_index
        self._extract = extract
        self._candidate_count = candidate_count

    def search_detailed(self, query: str, k: int = 3) -> SearchResponse:
        query_spec = self._extract(query)
        query_group = extract_pipe_group(query)
        candidates = [
            self._rank_candidate(query_spec, self._index[hit.name], hit.score)
            for hit in self._searcher.search(query, self._candidate_count)
            if hit.name in self._index
        ]
        ranked = sorted(candidates, key=lambda result: self._sort_key(result, query_group))

        if (ranked and ranked[0].is_exact) or not supports_template(query_spec):
            return SearchResponse(query_spec, ranked[:k])
        generated = self._generate(query_spec, query_group, ranked)
        return SearchResponse(query_spec, [generated, *ranked[: k - 1]])

    def search(self, query: str, k: int) -> list[SearchHit]:
        return [
            SearchHit(result.name, result.semantic_score or 0.0, is_generated=result.is_generated)
            for result in self.search_detailed(query, k).results
        ]

    def _rank_candidate(self, query_spec: PipeSpec, entry: CatalogEntry, score: float) -> RankedResult:
        comparison = query_spec.compare(entry.spec)
        matched = sum(value is True for value in comparison.values())
        is_exact = bool(query_spec.known_fields()) and matched == len(query_spec.known_fields())
        return RankedResult(entry.name, entry.spec, comparison, entry.replacements, score, is_exact=is_exact)

    def _sort_key(self, result: RankedResult, query_group: int | None) -> tuple:
        matched = sum(value is True for value in result.comparison.values())
        candidate_group = self._index[result.name].group
        group_differs = query_group is not None and candidate_group is not None and candidate_group != query_group
        return len(result.mismatches), -matched, group_differs, -result.semantic_score

    def _generate(self, query_spec: PipeSpec, query_group: int | None, ranked: list[RankedResult]) -> RankedResult:
        group = query_group
        if group is None:
            group = next(
                (self._index[result.name].group for result in ranked
                 if result.spec.pipe_type == query_spec.pipe_type and self._index[result.name].group is not None),
                None,
            )
        return RankedResult(
            name=build_name(query_spec, group),
            spec=query_spec,
            comparison=query_spec.compare(query_spec),
            is_generated=True,
            assumed_group=query_group is None and group is not None,
        )
