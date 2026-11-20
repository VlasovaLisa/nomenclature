import numpy as np

from core.semantic_search import EmbeddingCache, ModelConfig, SemanticSearcher

VOCABULARY = ["бесшовная", "электросварная", "обсадная", "89х8", "114х6"]


class BagOfWordsEncoder:
    def __init__(self):
        self.encoded: list[str] = []

    def encode(self, sentences, *, normalize_embeddings, convert_to_numpy):
        self.encoded.extend(sentences)
        vectors = np.array([[float(word in sentence) for word in VOCABULARY] + [1e-3] for sentence in sentences])
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


CONFIG = ModelConfig("fake-model", query_prefix="query: ", passage_prefix="passage: ")
NAMES = ["Труба бесшовная 89х8", "Труба электросварная 114х6", "Труба обсадная 89х8"]


def test_most_similar_name_ranks_first():
    searcher = SemanticSearcher(NAMES, BagOfWordsEncoder(), CONFIG)

    hits = searcher.search("труба б/ш 89*8", k=2)

    assert [hit.name for hit in hits] == ["Труба бесшовная 89х8", "Труба обсадная 89х8"]


def test_prefixes_and_normalization_are_applied():
    encoder = BagOfWordsEncoder()
    searcher = SemanticSearcher(NAMES[:1], encoder, CONFIG)

    searcher.search("Тр. Б/Ш", k=1)

    assert encoder.encoded == ["passage: труба бесшовная 89х8", "query: труба бесшовная"]


def test_raw_text_mode_skips_normalization():
    encoder = BagOfWordsEncoder()

    SemanticSearcher(NAMES[:1], encoder, ModelConfig("fake-model"), normalize_text=False)

    assert encoder.encoded == ["Труба бесшовная 89х8"]


def test_cached_embeddings_are_reused(tmp_path):
    cache = EmbeddingCache(tmp_path / "embeddings.npz")
    SemanticSearcher(NAMES, BagOfWordsEncoder(), CONFIG, cache=cache)
    encoder = BagOfWordsEncoder()

    SemanticSearcher(NAMES, encoder, CONFIG, cache=cache)

    assert encoder.encoded == []


def test_cache_is_rebuilt_when_catalog_changes(tmp_path):
    cache = EmbeddingCache(tmp_path / "embeddings.npz")
    SemanticSearcher(NAMES, BagOfWordsEncoder(), CONFIG, cache=cache)
    encoder = BagOfWordsEncoder()

    SemanticSearcher(NAMES[:2], encoder, CONFIG, cache=cache)

    assert len(encoder.encoded) == 2


def test_duplicate_names_are_indexed_once():
    searcher = SemanticSearcher([NAMES[0], NAMES[0]], BagOfWordsEncoder(), CONFIG)

    assert len(searcher.search("труба", k=5)) == 1
