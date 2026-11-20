from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SearchHit:
    name: str
    score: float
    is_generated: bool = False


class Searcher(Protocol):
    def search(self, query: str, k: int) -> list[SearchHit]: ...
