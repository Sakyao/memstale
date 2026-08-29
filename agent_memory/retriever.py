"""Hybrid retrieval: BM25 (sparse) + dense vectors, fused with RRF.

Pipeline (mirrors production hybrid-search designs):
    BM25 over FTS5        → sparse ranking
    embedding cosine      → dense ranking
    Reciprocal Rank Fusion (RRF) merges both ranked lists so that a result
    which ranks well in *either* channel surfaces — robust to query phrasing
    differences between the user and the stored text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .models import ScoredMemory
from .timeline import is_effective


@dataclass
class QueryFilters:
    """Optional retrieval filters."""

    entity_ids: list[str] = field(default_factory=list)
    at: str | None = None  # only memories effective at this time
    min_score: float = 0.0
    limit: int = 10


class Retriever:
    """Hybrid (BM25 + dense) retriever with optional temporal filters."""

    def __init__(self, store, embedder, rrf_k: int = 60):
        self.store = store
        self.embedder = embedder
        self.rrf_k = rrf_k

    def _dense_candidates(self, query: str, limit: int) -> list[ScoredMemory]:
        qv = self.embedder.embed(query)
        scored = []
        for m in self.store.list_memories():
            mv = self.embedder.embed(m.content)
            scored.append(ScoredMemory(m, float(qv @ mv)))
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[: max(limit * 3, 10)]

    def _sparse_candidates(self, query: str, limit: int) -> list[ScoredMemory]:
        memories = self.store.search_fts(query, limit=limit * 3)
        # rank by FTS order; assign a descending pseudo-score
        n = max(len(memories), 1)
        return [ScoredMemory(m, (n - i) / n) for i, m in enumerate(memories)]

    @staticmethod
    def _rrf(ranked: list[ScoredMemory], k: int) -> dict[str, float]:
        fused: dict[str, float] = {}
        for rank, item in enumerate(ranked):
            fused[item.memory.id] = fused.get(item.memory.id, 0.0) + 1.0 / (k + rank + 1)
        return fused

    def search(self, query: str, filters: QueryFilters | None = None) -> list[ScoredMemory]:
        filters = filters or QueryFilters()
        sparse = self._sparse_candidates(query, filters.limit)
        dense = self._dense_candidates(query, filters.limit)

        fused: dict[str, float] = {}
        for mid, s in self._rrf(sparse, self.rrf_k).items():
            fused[mid] = fused.get(mid, 0.0) + s
        for mid, s in self._rrf(dense, self.rrf_k).items():
            fused[mid] = fused.get(mid, 0.0) + s

        memories = {m.id: m for m in self.store.list_memories()}
        results: list[ScoredMemory] = []
        for mid, score in sorted(fused.items(), key=lambda kv: kv[1], reverse=True):
            m = memories.get(mid)
            if m is None:
                continue
            if not is_effective(m, filters.at):
                continue
            if filters.entity_ids and not (set(m.entity_ids) & set(filters.entity_ids)):
                continue
            if score < filters.min_score:
                continue
            results.append(ScoredMemory(m, score))
            if len(results) >= filters.limit:
                break
        return results

    def rerank(self, results: list[ScoredMemory], query: str) -> list[ScoredMemory]:
        """Optional semantic rerank on top of retrieved candidates."""
        qv = self.embedder.embed(query)
        for item in results:
            mv = self.embedder.embed(item.memory.content)
            item.score += 0.5 * float(qv @ mv)
        results.sort(key=lambda s: s.score, reverse=True)
        return results
