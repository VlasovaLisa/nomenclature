from collections.abc import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from core.search import SearchHit


class TfidfSearcher:
    def __init__(self, names: Iterable[str]):
        self._names = list(dict.fromkeys(names))
        self._vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), lowercase=True)
        self._matrix = self._vectorizer.fit_transform(self._names)

    def search(self, query: str, k: int) -> list[SearchHit]:
        query_vector = self._vectorizer.transform([query])
        scores = (self._matrix @ query_vector.T).toarray().ravel()
        top_indices = np.argsort(-scores, kind="stable")[:k]
        return [SearchHit(self._names[i], float(scores[i])) for i in top_indices]
