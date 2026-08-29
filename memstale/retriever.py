"""Hybrid retrieval: BM25 (sparse) + dense vectors, fused with RRF.

Pipeline (mirrors production hybrid-search designs):
    BM25 over FTS5        → sparse ranking
    embedding cosine      → dense ranking
    Reciprocal Rank Fusion (RRF) merges both ranked lists so that a result
    which ranks well in *either* channel surfaces — robust to query phrasing
    differences between the user and the stored text.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .models import ScoredMemory
from .timeline import is_effective_at, parse_time


@dataclass
class QueryFilters:
    """Optional retrieval filters."""

    entity_ids: list[str] = field(default_factory=list)
    at: str | None = None  # only memories effective at this time
    min_score: float = 0.0
    limit: int = 10


class Retriever:
    """Hybrid (BM25 + dense) retriever with optional temporal filters.

    `freshness_weight` (default 0) enables temporal re-ranking for queries
    WITHOUT an explicit `at` timestamp ("what's true *now*?"): candidates are
    boosted by how recently they took effect (exponential decay from
    `effective_at`). Queries that DO pass an `at` time are untouched — they
    keep pure bitemporal semantics.
    """

    def __init__(
        self,
        store,
        embedder,
        rrf_k: int = 60,
        freshness_weight: float = 0.0,
        freshness_halflife_days: float = 30.0,
        current_state_demote: bool = False,
        stale_penalty: float = 0.02,
    ):
        self.store = store
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.freshness_weight = freshness_weight
        self.freshness_halflife_days = freshness_halflife_days
        # "demote, don't exclude": for current-state queries, superseded /
        # future facts are pushed to the bottom of the ranking instead of
        # being dropped, so recall never suffers from an imperfect conflict
        # resolver while staleness@1 stays low.
        self.current_state_demote = current_state_demote
        self.stale_penalty = stale_penalty

    def _freshness(self, memory, now: datetime) -> float:
        """Exponential time-decay of a memory's recency in [0, 1]."""
        try:
            days = max(0.0, (now - parse_time(memory.effective_at)).total_seconds() / 86400.0)
        except ValueError:
            return 0.0
        return math.exp(-days / self.freshness_halflife_days)

    def _dense_candidates(self, query: str, limit: int) -> list[ScoredMemory]:
        qv = self.embedder.embed(query)
        scored = []
        # Include deprecated memories: bitemporal filtering decides validity
        # *at query time*; a superseded fact may still be the correct answer
        # for a past timestamp.
        for m in self.store.list_memories(include_deprecated=True):
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

        memories = {m.id: m for m in self.store.list_memories(include_deprecated=True)}
        now = datetime.now(timezone.utc)
        use_freshness = self.freshness_weight > 0 and filters.at is None
        ranked: list[ScoredMemory] = []
        for mid, score in fused.items():
            m = memories.get(mid)
            if m is None:
                continue
            if filters.entity_ids and not (set(m.entity_ids) & set(filters.entity_ids)):
                continue
            if score < filters.min_score:
                continue
            if filters.at is not None:
                # explicit timestamp: strict bitemporal semantics
                if not is_effective_at(m, filters.at, self.store):
                    continue
                final = score
            elif self.current_state_demote:
                # current-state query: demote stale/future, never exclude
                valid = is_effective_at(m, None, self.store)
                final = score if valid else score * self.stale_penalty
                if valid and use_freshness:
                    final = score * (1.0 + self.freshness_weight * self._freshness(m, now))
            else:
                if not is_effective_at(m, None, self.store):
                    continue
                final = score
                if use_freshness:
                    final = score * (1.0 + self.freshness_weight * self._freshness(m, now))
            ranked.append(ScoredMemory(m, final))
        ranked.sort(key=lambda s: s.score, reverse=True)
        return ranked[: filters.limit]

    def rerank(self, results: list[ScoredMemory], query: str) -> list[ScoredMemory]:
        """Optional semantic rerank on top of retrieved candidates."""
        qv = self.embedder.embed(query)
        for item in results:
            mv = self.embedder.embed(item.memory.content)
            item.score += 0.5 * float(qv @ mv)
        results.sort(key=lambda s: s.score, reverse=True)
        return results
