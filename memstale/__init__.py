"""memstale: temporal-aware long-term memory for AI agents.

Hybrid vector + graph memory with:
- bitemporal facts (effective_at / created_at)
- conflict soft-deprecation
- BM25 + dense vector retrieval fused with RRF
"""

from .models import Memory, Entity, Relation, ScoredMemory
from .embedder import Embedder, HashingEmbedder
from .store import MemoryStore
from .timeline import Timeline
from .conflict import ConflictResolver
from .retriever import Retriever
from .graph import MemoryGraph
from .memory import AgentMemory

__version__ = "0.1.0"

__all__ = [
    "Memory",
    "Entity",
    "Relation",
    "ScoredMemory",
    "Embedder",
    "HashingEmbedder",
    "MemoryStore",
    "Timeline",
    "ConflictResolver",
    "Retriever",
    "MemoryGraph",
    "AgentMemory",
    "__version__",
]
