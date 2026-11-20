import pandas as pd

from core.extract import PipeSpec
from core.ranker import CatalogEntry, HybridRanker, build_catalog_index
from core.search import SearchHit


class FixedSearcher:
    def __init__(self, scored_names: list[tuple[str, float]]):
        self._hits = [SearchHit(name, score) for name, score in scored_names]

    def search(self, query: str, k: int) -> list[SearchHit]:
        return self._hits[:k]


def entry(name: str, spec: PipeSpec, group: int | None = None, replacements: tuple[str, ...] = ()) -> CatalogEntry:
    return CatalogEntry(name, spec, group, replacements)


SEAMLESS_219X8_K48 = entry("Труба бесшовная группа 2 219х8 К48", PipeSpec("бесшовная", 219, 8, "К48"), 2)
WELDED_219X8_K48 = entry("Труба электросварная группа 2 219х8 К48", PipeSpec("электросварная", 219, 8, "К48"), 2, ("Замена А",))
WELDED_219X8_K48_GROUP4 = entry("Труба электросварная группа 4 219х8 К48", PipeSpec("электросварная", 219, 8, "К48"), 4)
WELDED_219X8_K50 = entry("Труба электросварная группа 2 219х8 К50", PipeSpec("электросварная", 219, 8, "К50"), 2)


def make_ranker(*scored_entries: tuple[CatalogEntry, float]) -> HybridRanker:
    index = {item.name: item for item, _ in scored_entries}
    return HybridRanker(FixedSearcher([(item.name, score) for item, score in scored_entries]), index)


def test_matching_parameters_outrank_higher_text_similarity():
    ranker = make_ranker((SEAMLESS_219X8_K48, 0.99), (WELDED_219X8_K50, 0.95), (WELDED_219X8_K48, 0.90))

    response = ranker.search_detailed("Труба эл/св 219х8 К48 группа 2")

    assert [result.name for result in response.results] == [
        WELDED_219X8_K48.name,
        SEAMLESS_219X8_K48.name,
        WELDED_219X8_K50.name,
    ]
    assert response.results[0].is_exact
    assert response.results[0].replacements == ("Замена А",)
    assert response.results[1].mismatches == ["pipe_type"]
    assert response.has_exact_match


def test_requested_group_breaks_ties_between_exact_matches():
    ranker = make_ranker((WELDED_219X8_K48_GROUP4, 0.99), (WELDED_219X8_K48, 0.90))

    response = ranker.search_detailed("Труба электросварная 219х8 К48 группа 2")

    assert response.results[0].name == WELDED_219X8_K48.name


def test_mismatched_parameters_are_never_exact():
    ranker = make_ranker((WELDED_219X8_K50, 0.99))

    response = ranker.search_detailed("Труба электросварная 219х8 К48 группа 2")

    assert not any(result.is_exact for result in response.results if not result.is_generated)


def test_new_combination_gets_generated_configuration_first():
    ranker = make_ranker((WELDED_219X8_K50, 0.99), (WELDED_219X8_K48, 0.95))

    response = ranker.search_detailed("Труба электросварная 219х10 К48 группа 4", k=3)

    generated = response.results[0]
    assert generated.is_generated
    assert generated.name == "Труба электросварная группа 4 219х10 К48 ТТТ-01.02.04-01"
    assert not generated.assumed_group
    assert len(response.results) == 3
    assert not response.has_exact_match


def test_missing_group_is_taken_from_nearest_candidate_and_marked_as_assumed():
    ranker = make_ranker((WELDED_219X8_K48_GROUP4, 0.99))

    generated = ranker.search_detailed("Труба электросварная 219х10 К48").results[0]

    assert generated.name == "Труба электросварная группа 4 219х10 К48 ТТТ-01.02.04-01"
    assert generated.assumed_group


def test_no_generation_when_exact_match_exists():
    ranker = make_ranker((WELDED_219X8_K48, 0.9))

    response = ranker.search_detailed("Труба электросварная 219х8 К48 группа 2")

    assert not any(result.is_generated for result in response.results)


def test_no_generation_for_unsupported_template():
    casing = entry("Труба обсадная ОТТМ-114х7,4-Д", PipeSpec("обсадная", 114, 7.4, "Д"))
    ranker = make_ranker((casing, 0.9))

    response = ranker.search_detailed("Труба обсадная ОТТМ-114х7,4-Е")

    assert [result.name for result in response.results] == [casing.name]
    assert not response.has_exact_match


def test_search_returns_hits_with_generated_flag():
    ranker = make_ranker((WELDED_219X8_K48, 0.9))

    hits = ranker.search("Труба электросварная 219х10 К48 группа 2", k=2)

    assert [hit.is_generated for hit in hits] == [True, False]


def test_build_catalog_index_merges_replacements_and_converts_types():
    catalog = pd.DataFrame(
        {
            "name": ["Труба А", "Труба А"],
            "replacements": [["Замена 1", "Замена 2"], ["Замена 2", "Замена 3"]],
            "pipe_type": ["бесшовная", "бесшовная"],
            "diameter": [89.0, 89.0],
            "wall_thickness": [8.0, 8.0],
            "steel_grade": [None, None],
            "pipe_group": pd.array([4, 4], dtype="Int64"),
        }
    )

    index = build_catalog_index(catalog)

    assert index["Труба А"] == CatalogEntry("Труба А", PipeSpec("бесшовная", 89.0, 8.0, None), 4, ("Замена 1", "Замена 2", "Замена 3"))
    assert type(index["Труба А"].spec.diameter) is float
