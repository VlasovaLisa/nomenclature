from collections.abc import Callable

import pandas as pd

from core.ranker import HybridRanker, SearchResponse, build_catalog_index
from core.search import Searcher

SearcherFactory = Callable[[pd.DataFrame], Searcher]


class NomenclatureService:
    def __init__(self, catalog: pd.DataFrame, searcher_factory: SearcherFactory):
        self._searcher_factory = searcher_factory
        self.replace_catalog(catalog)

    def replace_catalog(self, catalog: pd.DataFrame) -> None:
        ranker = HybridRanker(self._searcher_factory(catalog), build_catalog_index(catalog))
        self._catalog, self._ranker = catalog, ranker

    @property
    def pair_count(self) -> int:
        return len(self._catalog)

    @property
    def name_count(self) -> int:
        return self._catalog["name"].nunique()

    def search(self, query: str, k: int = 3) -> SearchResponse:
        return self._ranker.search_detailed(query, k)
