"""Entity-relationship graph over memories (networkx-backed, persistable).

Answering "who depends on whom", "what changed", and neighborhood questions
needs a graph view, not just a vector bag. This module builds the graph from
stored entities + relations and offers traversal helpers.

Neo4j is *not* required: the default backend is an in-process networkx graph
hydrated from SQLite, keeping the library zero-config. Swap in a real graph DB
by re-implementing the handful of traversal methods.
"""

from __future__ import annotations

from typing import Iterable

try:
    import networkx as nx
except ImportError:  # pragma: no cover
    nx = None  # type: ignore[assignment]

from .models import Entity, Relation


class MemoryGraph:
    """Graph view over entities and relations."""

    def __init__(self, store):
        self.store = store
        if nx is None:  # pragma: no cover
            raise RuntimeError("networkx is required for graph features: pip install agent-memory[graph]")
        self._g: nx.DiGraph = nx.DiGraph()
        self._reload()

    def _reload(self) -> None:
        self._g.clear()
        for entity in self.store.list_entities():
            self._g.add_node(entity.id, name=entity.name, metadata=entity.metadata)
        for rel in self.store.list_relations():
            self._g.add_edge(
                rel.source_id,
                rel.target_id,
                type=rel.type,
                id=rel.id,
                metadata=rel.metadata,
            )

    # ------------------------------------------------------------------ #
    # mutation helpers (keep store and graph in sync)
    # ------------------------------------------------------------------ #
    def add_entity(self, entity: Entity) -> Entity:
        self.store.upsert_entity(entity)
        self._reload()
        return entity

    def add_relation(self, relation: Relation) -> Relation:
        self.store.add_relation(relation)
        self._reload()
        return relation

    def remove_relation(self, relation_id: str) -> bool:
        ok = self.store.delete_relation(relation_id)
        if ok:
            self._reload()
        return ok

    # ------------------------------------------------------------------ #
    # traversal
    # ------------------------------------------------------------------ #
    def neighbors(self, entity_id: str, depth: int = 1) -> list[Entity]:
        """Entities reachable within `depth` hops (undirected view)."""
        self._reload()
        if entity_id not in self._g:
            return []
        seen = {entity_id}
        frontier = {entity_id}
        for _ in range(depth):
            nxt: set[str] = set()
            for node in frontier:
                nxt.update(self._g.successors(node))
                nxt.update(self._g.predecessors(node))
            nxt -= seen
            seen |= nxt
            frontier = nxt
        seen.discard(entity_id)
        return [self.store.get_entity(e) for e in sorted(seen) if self.store.get_entity(e)]

    def paths(self, source_id: str, target_id: str, max_depth: int = 5) -> list[list[dict]]:
        """All simple directed paths from source to target as relation chains."""
        self._reload()
        if source_id not in self._g or target_id not in self._g:
            return []
        try:
            raw = nx.all_simple_paths(self._g, source_id, target_id, cutoff=max_depth)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
        out = []
        for path in raw:
            chain = []
            for a, b in zip(path, path[1:]):
                edge = self._g[a][b]
                chain.append(
                    {
                        "from": self._g.nodes[a]["name"],
                        "to": self._g.nodes[b]["name"],
                        "type": edge.get("type", ""),
                    }
                )
            out.append(chain)
        return out

    def entities_of_memory(self, memory_id: str) -> list[Entity]:
        """Resolve entity objects attached to a memory."""
        memory = self.store.get_memory(memory_id)
        if memory is None:
            return []
        return [e for e in (self.store.get_entity(eid) for eid in memory.entity_ids) if e]

    def all_relations(self) -> list[Relation]:
        return self.store.list_relations()

    def summary(self) -> dict:
        self._reload()
        return {
            "entities": self._g.number_of_nodes(),
            "relations": self._g.number_of_edges(),
        }
