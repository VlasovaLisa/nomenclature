import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from core.normalize import normalize
from core.search import SearchHit


class Encoder(Protocol):
    def encode(self, sentences: list[str], *, normalize_embeddings: bool, convert_to_numpy: bool) -> np.ndarray: ...


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    query_prefix: str = ""
    passage_prefix: str = ""
    revision: str | None = None


MODELS = {
    "minilm": ModelConfig("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
    "e5-small": ModelConfig(
        "intfloat/multilingual-e5-small",
        query_prefix="query: ",
        passage_prefix="passage: ",
        revision="614241f622f53c4eeff9890bdc4f31cfecc418b3",
    ),
    "rubert-tiny2": ModelConfig("cointegrated/rubert-tiny2"),
}
DEFAULT_MODEL = "e5-small"


def load_encoder(config: ModelConfig) -> Encoder:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.model_id, revision=config.revision, device="cpu")


def encode(encoder: Encoder, texts: Sequence[str]) -> np.ndarray:
    return encoder.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)


class EmbeddingCache:
    def __init__(self, path: Path):
        self._path = path

    @staticmethod
    def fingerprint(config: ModelConfig, normalize_text: bool, names: Sequence[str]) -> str:
        payload = json.dumps([config.model_id, config.revision, config.passage_prefix, normalize_text, list(names)], ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def load(self, fingerprint: str) -> np.ndarray | None:
        if not self._path.exists():
            return None
        with np.load(self._path) as stored:
            if str(stored["fingerprint"]) != fingerprint:
                return None
            return stored["embeddings"]

    def save(self, fingerprint: str, embeddings: np.ndarray) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self._path, fingerprint=fingerprint, embeddings=embeddings)


class SemanticSearcher:
    def __init__(
        self,
        names: Iterable[str],
        encoder: Encoder,
        config: ModelConfig,
        normalize_text: bool = True,
        cache: EmbeddingCache | None = None,
    ):
        self._names = list(dict.fromkeys(names))
        self._encoder = encoder
        self._config = config
        self._normalize_text = normalize_text
        self._embeddings = self._load_or_build_embeddings(cache)

    def _prepare(self, text: str, prefix: str) -> str:
        return prefix + (normalize(text) if self._normalize_text else text)

    def _load_or_build_embeddings(self, cache: EmbeddingCache | None) -> np.ndarray:
        fingerprint = EmbeddingCache.fingerprint(self._config, self._normalize_text, self._names)
        cached = cache.load(fingerprint) if cache else None
        if cached is not None:
            return cached
        embeddings = encode(self._encoder, [self._prepare(name, self._config.passage_prefix) for name in self._names])
        if cache:
            cache.save(fingerprint, embeddings)
        return embeddings

    def search(self, query: str, k: int) -> list[SearchHit]:
        query_vector = encode(self._encoder, [self._prepare(query, self._config.query_prefix)])[0]
        scores = self._embeddings @ query_vector
        top_indices = np.argsort(-scores, kind="stable")[:k]
        return [SearchHit(self._names[i], float(scores[i])) for i in top_indices]
