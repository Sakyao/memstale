"""Data models for memstale."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


def now_iso() -> str:
    """Current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Memory:
    """A single memory unit (fact / note / observation).

    Attributes:
        content: The textual content of the memory.
        entity_ids: Entities referenced by this memory.
        effective_at: Time from which this memory is valid (生效时间).
        created_at: Time this memory was written/discovered (发现时间).
        source: Where the memory came from (doc, chat, tool, ...).
        deprecated: Soft-deleted flag (冲突软失效).
        replaced_by: Memory id that superseded this one.
        metadata: Arbitrary extra fields.
    """

    content: str
    entity_ids: list[str] = field(default_factory=list)
    effective_at: str = field(default_factory=now_iso)
    created_at: str = field(default_factory=now_iso)
    source: str = ""
    deprecated: bool = False
    replaced_by: str | None = None
    metadata: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "entity_ids": list(self.entity_ids),
            "effective_at": self.effective_at,
            "created_at": self.created_at,
            "source": self.source,
            "deprecated": self.deprecated,
            "replaced_by": self.replaced_by,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Memory":
        return cls(**data)


@dataclass
class Entity:
    """A named entity in the memory graph."""

    name: str
    metadata: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "metadata": dict(self.metadata)}

    @classmethod
    def from_dict(cls, data: dict) -> "Entity":
        return cls(**data)


@dataclass
class Relation:
    """Directed relationship between two entities."""

    source_id: str
    target_id: str
    type: str
    metadata: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "type": self.type,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Relation":
        return cls(**data)


@dataclass
class ScoredMemory:
    """A memory returned by retrieval with its score."""

    memory: Memory
    score: float

    def to_dict(self) -> dict:
        return {"memory": self.memory.to_dict(), "score": self.score}
