"""Bitemporal timeline: knowledge validity over time.

Two clocks per memory:
    effective_at  — when the fact becomes true in the world  (生效时间)
    created_at    — when the fact was discovered / written  (发现时间)

Separating them lets the system answer both
    "what is true right now?"  and  "what did we know at time T?"
Unlike naive vector stores, facts that are no longer effective (or that
were superseded) are naturally excluded — no stale answers.
"""

from __future__ import annotations

from datetime import datetime, timezone

from .models import Memory, now_iso

# ISO 8601 strings sort lexicographically when the same format is used.
MIN_TIME = "0000-01-01T00:00:00+00:00"
MAX_TIME = "9999-12-31T23:59:59+00:00"


def parse_time(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))


def is_effective(memory: Memory, at: str | None = None) -> bool:
    """True if a memory is valid at time `at` (defaults to *now*).

    Simple validity: not deprecated and effective_at <= at.
    See :func:`is_effective_at` for the full bitemporal semantics.
    """
    if memory.deprecated:
        return False
    query = at or now_iso()
    return parse_time(memory.effective_at) <= parse_time(query)


def is_effective_at(memory: Memory, at: str | None, store) -> bool:
    """Bitemporal validity: is `memory` the *truth at time* `at`?

    A deprecated memory is still the truth for query times before it was
    superseded. E.g. "capital = Astana" (deprecated in 2022 by a rename)
    is the correct answer to a 2020 query, but not to a 2023 query.
    """
    at_t = parse_time(at or now_iso())
    if parse_time(memory.effective_at) > at_t:
        return False
    if memory.deprecated:
        if not memory.replaced_by:
            return False  # manually deprecated, no successor
        replaced = store.get_memory(memory.replaced_by)
        if replaced and parse_time(replaced.created_at) <= at_t:
            return False  # already superseded at time `at`
    return True


class Timeline:
    """Query memories through a temporal lens."""

    def __init__(self, store):
        self.store = store

    def active_at(self, at: str | None = None) -> list[Memory]:
        """Memories that are the truth at time `at` (default: now).

        Uses full bitemporal semantics: a superseded memory still counts for
        query times before its replacement was discovered.
        """
        return [m for m in self.store.list_memories(include_deprecated=True) if is_effective_at(m, at, self.store)]

    def effective_between(self, start: str, end: str) -> list[Memory]:
        """Memories effective within [start, end], soft-deprecation aware.

        A deprecated memory still appears here when the memory that replaced
        it was discovered *after* ``start`` (it was still the truth during
        part of the window).
        """
        s, e = parse_time(start), parse_time(end)
        filtered = []
        for m in self.store.list_memories(include_deprecated=True):
            eff = parse_time(m.effective_at)
            if eff > e or eff < s:
                continue
            if not m.deprecated:
                filtered.append(m)
                continue
            replaced = self.store.get_memory(m.replaced_by) if m.replaced_by else None
            replaced_at = parse_time(replaced.created_at) if replaced else parse_time(MAX_TIME)
            if replaced_at > s:
                filtered.append(m)
        return sorted(filtered, key=lambda m: parse_time(m.effective_at))

    def replay(self, entity_id: str | None = None, limit: int = 50) -> list[Memory]:
        """Replay knowledge evolution in discovery order (created_at ascending)."""
        memories = self.store.list_memories(include_deprecated=True)
        if entity_id:
            memories = [m for m in memories if entity_id in m.entity_ids]
        memories.sort(key=lambda m: parse_time(m.created_at))
        return memories[-limit:]

    def knowledge_at(self, entity_id: str | None, at: str) -> list[Memory]:
        """What we knew at discovery time `at` (even if now superseded)."""
        memories = self.store.list_memories(include_deprecated=True)
        if entity_id:
            memories = [m for m in memories if entity_id in m.entity_ids]
        out = []
        for m in memories:
            if parse_time(m.created_at) <= parse_time(at):
                out.append(m)
        out.sort(key=lambda m: parse_time(m.created_at))
        return out
