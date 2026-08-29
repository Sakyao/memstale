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


def default_judge(embedder) -> JudgeFn:
    """Default conflict judge: vector similarity * shared-entity bonus.

    Returns similarity in [0, 1]. Higher means "more likely to be a
    contradictory restatement of the same fact".
    """

    def judge(new: Memory, existing: Memory) -> float:
        v1 = embedder.embed(new.content)
        v2 = embedder.embed(existing.content)
        sim = float(v1 @ v2)  # normalized vectors -> cosine
        shared = len(set(new.entity_ids) & set(existing.entity_ids))
        overlap = len(set(new.content) & set(existing.content)) / max(
            len(set(new.content) | set(existing.content)), 1
        )
        # Boost when entities overlap; penalize when they are essentially the
        # same text (duplicate, not a contradiction).
        score = sim * (1.0 + 0.5 * min(shared, 3))
        if overlap > 0.85:
            score *= 0.3
        return score

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
            # Only compare memories touching at least one shared entity to keep
            # the default judge cheap and precise.
            if not (set(existing.entity_ids) & set(new_memory.entity_ids)):
                continue
            score = self.judge(new_memory, existing)
            if score >= self.threshold:
                conflicts.append(Conflict(new_memory, existing, score))
        conflicts.sort(key=lambda c: c.score, reverse=True)
        return conflicts

    def resolve(self, new_memory: Memory) -> list[Conflict]:
        """Deprecate conflicting memories, linking them to ``new_memory``."""
        conflicts = self.find_conflicts(new_memory)
        for conflict in conflicts:
            self.store.deprecate_memory(conflict.existing_memory.id, replaced_by=new_memory.id)
        return conflicts
