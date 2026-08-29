"""High-level facade: AgentMemory ties everything together.

Typical usage::

    from memstale import AgentMemory

    mem = AgentMemory("memory.db")
    mem.remember("Sakya builds memstale", entities=["Sakya", "memstale"])
    mem.remember("memstale is a temporal memory library", entities=["memstale"])
    results = mem.query("what does Sakya work on?")
"""

from __future__ import annotations

from pathlib import Path

from .conflict import ConflictResolver
from .embedder import Embedder, HashingEmbedder
from .graph import MemoryGraph
from .models import Entity, Memory, Relation, ScoredMemory
from .retriever import QueryFilters, Retriever
from .store import MemoryStore
from .timeline import Timeline


class AgentMemory:
    """All-in-one facade over store / timeline / conflict / retrieval / graph."""

    def __init__(
        self,
        path: str | Path = ":memory:",
        embedder: Embedder | None = None,
        conflict_threshold: float = 0.55,
        auto_resolve: bool = True,
    ):
        self.store = MemoryStore(path)
        self.embedder = embedder or HashingEmbedder()
        self.timeline = Timeline(self.store)
        self.conflicts = ConflictResolver(self.store, self.embedder, threshold=conflict_threshold)
        self.retriever = Retriever(self.store, self.embedder)
        self.graph = MemoryGraph(self.store)
        self.auto_resolve = auto_resolve

    # ------------------------------------------------------------------ #
    # write path
    # ------------------------------------------------------------------ #
    def remember(
        self,
        content: str,
        entities: list[str] | None = None,
        relations: list[tuple[str, str, str]] | None = None,
        source: str = "",
        effective_at: str | None = None,
        created_at: str | None = None,
        metadata: dict | None = None,
        auto_resolve: bool | None = None,
    ) -> Memory:
        """Store a memory; optionally resolve conflicts against existing facts."""
        entity_ids: list[str] = []
        for name in entities or []:
            entity = self._ensure_entity(name)
            entity_ids.append(entity.id)

        memory = Memory(
            content=content,
            entity_ids=entity_ids,
            source=source,
            metadata=metadata or {},
        )
        if effective_at:
            memory.effective_at = effective_at
        if created_at:
            memory.created_at = created_at

        self.store.add_memory(memory)

        for src, tgt, rel_type in relations or []:
            s = self._ensure_entity(src)
            t = self._ensure_entity(tgt)
            self.store.add_relation(Relation(source_id=s.id, target_id=t.id, type=rel_type))

        if self.auto_resolve if auto_resolve is None else auto_resolve:
            self.conflicts.resolve(memory)
        return memory

    def _ensure_entity(self, name: str) -> Entity:
        existing = self.store.find_entity_by_name(name)
        if existing:
            return existing
        entity = Entity(name=name)
        self.store.upsert_entity(entity)
        return entity

    # ------------------------------------------------------------------ #
    # read path
    # ------------------------------------------------------------------ #
    def query(
        self,
        question: str,
        k: int = 10,
        entity_ids: list[str] | None = None,
        at: str | None = None,
    ) -> list[ScoredMemory]:
        """Hybrid retrieve memories relevant to `question`."""
        filters = QueryFilters(entity_ids=entity_ids or [], at=at, limit=k)
        return self.retriever.search(question, filters)

    def recall(self, memory_id: str) -> Memory | None:
        return self.store.get_memory(memory_id)

    # ------------------------------------------------------------------ #
    # temporal / graph / conflict views
    # ------------------------------------------------------------------ #
    def active_memories(self, at: str | None = None) -> list[Memory]:
        return self.timeline.active_at(at)

    def timeline_of(self, entity_id: str | None = None) -> list[Memory]:
        return self.timeline.replay(entity_id)

    def add_relation(self, source: str, target: str, rel_type: str) -> Relation:
        s = self._ensure_entity(source)
        t = self._ensure_entity(target)
        rel = Relation(source_id=s.id, target_id=t.id, type=rel_type)
        self.store.add_relation(rel)
        return rel

    def neighbors(self, entity: str, depth: int = 1):
        ent = self.store.find_entity_by_name(entity)
        if ent is None:
            return []
        return self.graph.neighbors(ent.id, depth)

    def paths(self, source: str, target: str, max_depth: int = 5):
        s = self.store.find_entity_by_name(source)
        t = self.store.find_entity_by_name(target)
        if s is None or t is None:
            return []
        return self.graph.paths(s.id, t.id, max_depth)

    def deprecate(self, memory_id: str, replaced_by: str | None = None) -> bool:
        return self.store.deprecate_memory(memory_id, replaced_by=replaced_by)

    def conflicts_for(self, content: str, entities: list[str] | None = None) -> list:
        entity_ids = []
        for name in entities or []:
            ent = self.store.find_entity_by_name(name)
            if ent:
                entity_ids.append(ent.id)
        probe = Memory(content=content, entity_ids=entity_ids)
        return self.conflicts.find_conflicts(probe)

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> "AgentMemory":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
