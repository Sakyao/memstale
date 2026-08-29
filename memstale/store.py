"""SQLite-backed persistence layer for memories, entities and relations.

Storage layout:
    memories  (id, content, entity_ids, metadata, effective_at, created_at,
               source, deprecated, replaced_by)
    entities  (id, name, metadata)
    relations (id, source_id, target_id, type, metadata)
    memory_fts (FTS5 index over content for sparse retrieval)

The schema deliberately avoids hard requirements on external services
(no Neo4j / Milvus needed) while keeping a clean seam to swap in heavier
backends later.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .models import Entity, Memory, Relation

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    entity_ids TEXT NOT NULL DEFAULT '[]',
    metadata TEXT NOT NULL DEFAULT '{}',
    effective_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT '',
    deprecated INTEGER NOT NULL DEFAULT 0,
    replaced_by TEXT
);
CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS relations (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    type TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_memories_effective ON memories(effective_at);
CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    content, id UNINDEXED, tokenize = 'unicode61'
);
"""


def _to_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _from_json(text: str):
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


class MemoryStore:
    """Persistent store with memory / entity / relation CRUD and FTS search."""

    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------ #
    # memories
    # ------------------------------------------------------------------ #
    def add_memory(self, memory: Memory) -> Memory:
        self._conn.execute(
            """
            INSERT INTO memories
                (id, content, entity_ids, metadata, effective_at, created_at,
                 source, deprecated, replaced_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory.id,
                memory.content,
                _to_json(memory.entity_ids),
                _to_json(memory.metadata),
                memory.effective_at,
                memory.created_at,
                memory.source,
                int(memory.deprecated),
                memory.replaced_by,
            ),
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO memory_fts(rowid, content, id) VALUES (?, ?, ?)",
            (self._memory_rowid(memory.id), memory.content, memory.id),
        )
        self._conn.commit()
        return memory

    def _memory_rowid(self, memory_id: str) -> int:
        row = self._conn.execute(
            "SELECT rowid FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"memory not found: {memory_id}")
        return int(row["rowid"])

    def get_memory(self, memory_id: str) -> Memory | None:
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        return self._row_to_memory(row) if row else None

    def list_memories(self, include_deprecated: bool = False) -> list[Memory]:
        sql = "SELECT * FROM memories"
        if not include_deprecated:
            sql += " WHERE deprecated = 0"
        sql += " ORDER BY created_at DESC"
        rows = self._conn.execute(sql).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def update_memory(self, memory: Memory) -> Memory:
        self._conn.execute(
            """
            UPDATE memories SET content=?, entity_ids=?, metadata=?,
                effective_at=?, created_at=?, source=?, deprecated=?,
                replaced_by=?
            WHERE id=?
            """,
            (
                memory.content,
                _to_json(memory.entity_ids),
                _to_json(memory.metadata),
                memory.effective_at,
                memory.created_at,
                memory.source,
                int(memory.deprecated),
                memory.replaced_by,
                memory.id,
            ),
        )
        self._conn.execute(
            "UPDATE memory_fts SET content=? WHERE id=?",
            (memory.content, memory.id),
        )
        self._conn.commit()
        return memory

    def delete_memory(self, memory_id: str) -> bool:
        cur = self._conn.execute("DELETE FROM memories WHERE id=?", (memory_id,))
        self._conn.execute("DELETE FROM memory_fts WHERE id=?", (memory_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def deprecate_memory(self, memory_id: str, replaced_by: str | None = None) -> bool:
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id=?", (memory_id,)
        ).fetchone()
        if row is None:
            return False
        memory = self._row_to_memory(row)
        memory.deprecated = True
        memory.replaced_by = replaced_by
        self.update_memory(memory)
        return True

    def search_fts(self, query: str, limit: int = 20) -> list[Memory]:
        """Sparse full-text search over memory content (BM25 by SQLite FTS5)."""
        try:
            rows = self._conn.execute(
                """
                SELECT memories.* FROM memory_fts
                JOIN memories ON memories.id = memory_fts.id
                WHERE memory_fts MATCH ?
                ORDER BY bm25(memory_fts)
                LIMIT ?
                """,
                (query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            # Query syntax unsupported by FTS5 (e.g. special chars) -> fallback LIKE
            rows = self._conn.execute(
                "SELECT * FROM memories WHERE content LIKE ? ORDER BY created_at DESC LIMIT ?",
                (f"%{query}%", limit),
            ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    # ------------------------------------------------------------------ #
    # entities
    # ------------------------------------------------------------------ #
    def upsert_entity(self, entity: Entity) -> Entity:
        self._conn.execute(
            """
            INSERT INTO entities (id, name, metadata) VALUES (?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET name=excluded.name, metadata=excluded.metadata
            """,
            (entity.id, entity.name, _to_json(entity.metadata)),
        )
        self._conn.commit()
        return entity

    def get_entity(self, entity_id: str) -> Entity | None:
        row = self._conn.execute(
            "SELECT * FROM entities WHERE id=?", (entity_id,)
        ).fetchone()
        return self._row_to_entity(row) if row else None

    def find_entity_by_name(self, name: str) -> Entity | None:
        row = self._conn.execute(
            "SELECT * FROM entities WHERE name=? COLLATE NOCASE", (name,)
        ).fetchone()
        return self._row_to_entity(row) if row else None

    def list_entities(self) -> list[Entity]:
        rows = self._conn.execute("SELECT * FROM entities ORDER BY name").fetchall()
        return [self._row_to_entity(r) for r in rows]

    # ------------------------------------------------------------------ #
    # relations
    # ------------------------------------------------------------------ #
    def add_relation(self, relation: Relation) -> Relation:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO relations (id, source_id, target_id, type, metadata)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                relation.id,
                relation.source_id,
                relation.target_id,
                relation.type,
                _to_json(relation.metadata),
            ),
        )
        self._conn.commit()
        return relation

    def list_relations(self) -> list[Relation]:
        rows = self._conn.execute("SELECT * FROM relations").fetchall()
        return [self._row_to_relation(r) for r in rows]

    def delete_relation(self, relation_id: str) -> bool:
        cur = self._conn.execute("DELETE FROM relations WHERE id=?", (relation_id,))
        self._conn.commit()
        return cur.rowcount > 0

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _row_to_memory(row: sqlite3.Row) -> Memory:
        return Memory(
            id=row["id"],
            content=row["content"],
            entity_ids=_from_json(row["entity_ids"]) or [],
            metadata=_from_json(row["metadata"]) or {},
            effective_at=row["effective_at"],
            created_at=row["created_at"],
            source=row["source"],
            deprecated=bool(row["deprecated"]),
            replaced_by=row["replaced_by"],
        )

    @staticmethod
    def _row_to_entity(row: sqlite3.Row) -> Entity:
        return Entity(id=row["id"], name=row["name"], metadata=_from_json(row["metadata"]) or {})

    @staticmethod
    def _row_to_relation(row: sqlite3.Row) -> Relation:
        return Relation(
            id=row["id"],
            source_id=row["source_id"],
            target_id=row["target_id"],
            type=row["type"],
            metadata=_from_json(row["metadata"]) or {},
        )

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
