"""Conflict resolution with soft-deprecation (软失效).

When new knowledge contradicts existing facts, we do NOT hard-delete the old
fact. Instead the old memory is marked ``deprecated`` and linked to its
replacement via ``replaced_by``. This keeps full auditability: at any point in
time you can ask "what did we believe before/after the update?".

The default judge is a cheap heuristic (entity overlap + embedding similarity
+ lexical distance). Pass your own ``judge`` callable to plug in an
LLM-as-a-judge layer — the same pattern used in production orchestration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from .models import Memory
from .timeline import is_effective


@dataclass
class Conflict:
    """A detected conflict between new memory and an existing one."""

    new_memory: Memory
    existing_memory: Memory
    score: float
    reason: str = ""


JudgeFn = Callable[[Memory, Memory], float]


_STOPWORDS = {
    "the", "is", "are", "was", "were", "has", "have", "had", "a", "an",
    "and", "or", "of", "to", "in", "on", "at", "for", "it", "its", "that",
    "this", "with", "from", "as", "by", "not", "be", "been", "being",
    # structural/template words that pollute topic Jaccard between unrelated
    # facts ("checkout-api runs on port 8443" vs "payout-scheduler runs on
    # port 5523" share "runs on port in production" without sharing a topic)
    "run", "runs", "running", "port", "ports", "production", "staging",
    "service", "services", "all", "every", "now", "still", "currently",
    "switched", "start", "started", "cost", "costs", "uses", "use", "used",
    "is", "are", "was", "were", "my", "our", "we", "i", "the", "a", "an",
}


def _content_tokens(text: str) -> set[str]:
    """Content-bearing tokens: alphanumeric, lowercased, stopwords removed."""
    return {t for t in re.findall(r"[A-Za-z0-9]+", text.lower()) if len(t) > 1 and t not in _STOPWORDS}


def default_judge(embedder) -> JudgeFn:
    """Default conflict judge: vector similarity gated by topic overlap.

    Returns similarity in [0, 1]. Higher means "more likely to be a
    contradictory restatement of the same fact".

    Naive similarity alone over-triggers: two "X is the CEO of Y" facts about
    *different* companies share wording and would wrongly deprecate each other.
    We gate on the Jaccard overlap of content-bearing tokens (topic cohesion):
    only memories about the same subject can be in conflict.
    """

    def judge(new: Memory, existing: Memory) -> float:
        v1 = embedder.embed(new.content)
        v2 = embedder.embed(existing.content)
        sim = float(v1 @ v2)  # normalized vectors -> cosine

        nw = _content_tokens(new.content)
        ew = _content_tokens(existing.content)
        inter = nw & ew
        union = nw | ew
        jac = len(inter) / len(union) if union else 0.0

        if jac < 0.2:
            # different topic -> not a conflict, whatever the phrasing
            return sim * 0.1
        return sim * (0.4 + 0.6 * min(jac * 2, 1.0))

    return judge


class ConflictResolver:
    """Find and apply soft-deprecation for conflicting memories."""

    def __init__(
        self,
        store,
        embedder,
        threshold: float = 0.55,
        judge: JudgeFn | None = None,
    ):
        self.store = store
        self.embedder = embedder
        self.threshold = threshold
        self.judge = judge or default_judge(embedder)

    def find_conflicts(self, new_memory: Memory) -> list[Conflict]:
        """Return existing active memories that conflict with ``new_memory``."""
        conflicts: list[Conflict] = []
        candidates = [m for m in self.store.list_memories() if m.id != new_memory.id]
        for existing in candidates:
            if not is_effective(existing):
                continue
            # Candidate gate: shared entity OR topic-token overlap. The topic
            # fallback lets the resolver work even when writes carry no entity
            # labels (e.g. raw session streams from a benchmark).
            if not self._related(new_memory, existing):
                continue
            score = self.judge(new_memory, existing)
            if score >= self.threshold:
                conflicts.append(Conflict(new_memory, existing, score))
        conflicts.sort(key=lambda c: c.score, reverse=True)
        return conflicts

    @staticmethod
    def _related(a: Memory, b: Memory) -> bool:
        """Cheap topic gate: shared entities, or content-token Jaccard >= 0.25."""
        if set(a.entity_ids) & set(b.entity_ids):
            return True
        ta, tb = _content_tokens(a.content), _content_tokens(b.content)
        union = ta | tb
        if not union:
            return False
        return len(ta & tb) / len(union) >= 0.25

    def resolve(self, new_memory: Memory) -> list[Conflict]:
        """Deprecate conflicting memories, linking them to ``new_memory``."""
        conflicts = self.find_conflicts(new_memory)
        for conflict in conflicts:
            self.store.deprecate_memory(conflict.existing_memory.id, replaced_by=new_memory.id)
        return conflicts
