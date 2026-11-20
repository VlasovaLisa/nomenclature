import json
from pathlib import Path

import pytest

from core.evaluation import EXISTING, NEW_COMBINATION, evaluate, load_queries
from core.search import SearchHit

QUERY_DIR = Path(__file__).resolve().parent.parent / "data" / "test"


class FixedSearcher:
    def __init__(self, names: list[str]):
        self._names = names

    def search(self, query: str, k: int) -> list[SearchHit]:
        return [SearchHit(name, 1.0) for name in self._names[:k]]


SPEC_89X8_K52 = {"pipe_type": "бесшовная", "diameter": 89, "wall_thickness": 8, "steel_grade": "К52"}


def make_query(query_id: str, kind: str, expected_names: list[str], query: str = "труба", expected_spec: dict | None = None) -> dict:
    return {
        "id": query_id,
        "query": query,
        "kind": kind,
        "expected_names": expected_names,
        "expected_spec": expected_spec or SPEC_89X8_K52,
    }


def test_hit_rate_counts_only_existing_queries():
    searcher = FixedSearcher(["A", "B", "C", "D"])
    queries = [
        make_query("1", EXISTING, ["C"]),
        make_query("2", EXISTING, ["D"]),
        make_query("3", NEW_COMBINATION, []),
    ]

    report = evaluate(searcher, queries, k=3)

    assert [result.rank for result in report.results] == [3, None, None]
    assert report.hit_rate(3) == 0.5
    assert report.hit_rate(1) == 0.0


def test_mean_reciprocal_rank_uses_first_expected_position():
    searcher = FixedSearcher(["A", "B", "C"])
    queries = [make_query("1", EXISTING, ["A"]), make_query("2", EXISTING, ["B", "C"])]

    report = evaluate(searcher, queries)

    assert report.mean_reciprocal_rank == pytest.approx((1 + 1 / 2) / 2)


def test_any_expected_name_counts_as_hit():
    report = evaluate(FixedSearcher(["X", "B"]), [make_query("1", EXISTING, ["A", "B"])])

    assert report.hit_rate(3) == 1.0


def test_extraction_accuracy_counts_each_of_four_fields():
    queries = [
        make_query("1", EXISTING, ["A"], query="Труба бесшовная 89х8 К52"),
        make_query("2", EXISTING, ["A"], query="Труба бесшовная 89х6"),
    ]

    report = evaluate(FixedSearcher(["A"]), queries)

    assert report.extraction_accuracy == (4 + 2) / 8


def test_top_hit_mismatch_rate_is_reported_per_kind():
    searcher = FixedSearcher(["Труба бесшовная 89х6 К52"])
    queries = [make_query("1", EXISTING, ["A"]), make_query("2", NEW_COMBINATION, [])]

    report = evaluate(searcher, queries)

    assert report.results[0].top_hit_mismatches == ["wall_thickness"]
    assert report.top_hit_mismatch_rate(EXISTING) == 1.0
    assert report.top_hit_mismatch_rate(NEW_COMBINATION) == 1.0


class GeneratingSearcher:
    def __init__(self, generated_name: str):
        self._generated_name = generated_name

    def search(self, query: str, k: int) -> list[SearchHit]:
        return [SearchHit(self._generated_name, 0.0, is_generated=True)]


@pytest.mark.parametrize(
    ("generated_name", "preserved"),
    [("Труба бесшовная группа 2 89х8 К52 ТТТ-01.02.04-01", True), ("Труба бесшовная группа 2 89х6 К52 ТТТ-01.02.04-01", False)],
)
def test_generated_configuration_must_preserve_requested_spec(generated_name, preserved):
    report = evaluate(GeneratingSearcher(generated_name), [make_query("1", NEW_COMBINATION, [])])

    assert report.generation_rate(NEW_COMBINATION) == 1.0
    assert report.preserved_config_rate == float(preserved)


@pytest.mark.parametrize("filename", ["test_queries.json", "dev_queries.json"])
def test_query_sets_are_well_formed(filename):
    queries = load_queries(QUERY_DIR / filename)

    assert len({query["id"] for query in queries}) == len(queries)
    for query in queries:
        assert set(query["expected_spec"]) == {"pipe_type", "diameter", "wall_thickness", "steel_grade"}
        assert bool(query["expected_names"]) == (query["kind"] == EXISTING)


def test_test_set_matches_spec_requirements():
    queries = load_queries(QUERY_DIR / "test_queries.json")

    assert len(queries) == 20
    assert sum(query["kind"] == NEW_COMBINATION for query in queries) >= 5


def test_test_and_dev_sets_do_not_share_catalog_items():
    test_names = {name for query in load_queries(QUERY_DIR / "test_queries.json") for name in query["expected_names"]}
    dev_names = {name for query in load_queries(QUERY_DIR / "dev_queries.json") for name in query["expected_names"]}

    assert not test_names & dev_names
