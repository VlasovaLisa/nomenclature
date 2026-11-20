from core.tfidf_search import TfidfSearcher

NAMES = [
    "Труба бесшовная группа 2 114х6 К52 ТТТ-01.02.04-01",
    "Труба электросварная группа 4 1020х24 К52 ТТТ-01.02.04-01",
    "Труба обсадная ОТТМ-245х7,9-Д ГОСТ 632-80 А",
]


def test_exact_name_ranks_first():
    searcher = TfidfSearcher(NAMES)

    hits = searcher.search(NAMES[1], k=3)

    assert hits[0].name == NAMES[1]
    assert hits[0].score > hits[1].score


def test_returns_at_most_k_hits():
    assert len(TfidfSearcher(NAMES).search("труба", k=2)) == 2


def test_duplicate_names_are_indexed_once():
    searcher = TfidfSearcher([NAMES[0], NAMES[0], NAMES[1]])

    hits = searcher.search(NAMES[0], k=3)

    assert [hit.name for hit in hits] == [NAMES[0], NAMES[1]]
