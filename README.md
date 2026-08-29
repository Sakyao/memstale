<div align="center">

# 🧠 agent-memory

**Temporal-aware long-term memory for AI agents** — hybrid vector + graph retrieval
with **bitemporal facts** and **conflict soft-deprecation**.

Zero-config · Pure Python · SQLite-backed · MCP-ready

[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-17_passing-brightgreen?style=for-the-badge)](tests)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-8A2BE2.svg?style=for-the-badge)](https://github.com/Sakyao/agent-memory/pulls)

</div>

---

## Why this exists

LLMs are **amnesiacs**: every conversation starts from scratch. Existing "memory"
solutions treat memory as a *vector bag* — they answer "what is semantically
similar" but ignore two things that actually matter:

| Problem | Consequence |
|---------|-------------|
| ❌ **No sense of time** | Agent trusts stale facts: "the capital of France is Lyon" (effective 2099) still gets retrieved *today* |
| ❌ **No conflict resolution** | New knowledge contradicts old facts; the agent doesn't know which to trust |
| ❌ **No relationships** | A pure vector store can't answer "who depends on whom" |

`agent-memory` solves these with a **bitemporal data model** and **hybrid retrieval**,
borrowing battle-tested patterns from production Agent infrastructure:

- **Bitemporal facts** — every memory carries `effective_at` (when it becomes true)
  and `created_at` (when it was discovered), so you can ask "what is true now?"
  *and* "what did we know at time T?"
- **Conflict soft-deprecation** — when new knowledge contradicts old facts, the old
  one is *softly deprecated* and linked to its replacement, keeping full auditability.
- **Hybrid retrieval (BM25 + dense + RRF)** — sparse and dense channels are fused
  with Reciprocal Rank Fusion, robust to wording gaps between question and memory.
- **Entity graph** — entities and typed relations give you neighborhood and path
  queries on top of the vector store.

## Features

- 🕑 **Bitemporal** facts: `effective_at` / `created_at`, timeline replay, "knowledge at T"
- ⚖️ **Soft-deprecation**: contradictory facts don't get deleted, they get superseded
- 🔎 **Hybrid retrieval**: BM25 (FTS5) + hashing-embedding dense + RRF fusion
- 🕸️ **Entity graph**: typed relations, neighborhood & path traversal (networkx)
- 🧩 **Pluggable embedder**: zero-dependency hashing embedder by default,
  `sentence-transformers` optional
- 🔌 **MCP server** included: let Claude/Cursor/OpenClaw remember across sessions
- 🪶 **Zero-config**: SQLite persistence, no external services required

## Quickstart

```bash
pip install agent-memory
```

```python
from agent_memory import AgentMemory

mem = AgentMemory("memory.db")

# remember facts with entities
mem.remember("The capital of France is Paris", entities=["France"], source="handbook")

# query with hybrid retrieval
for hit in mem.query("what is the capital of France?", k=3):
    print(f"[{hit.score:.3f}] {hit.memory.content}")

# a fact that only becomes true in the future is NOT retrieved today
mem.remember(
    "The capital of France is Lyon",
    entities=["France"],
    effective_at="2099-01-01T00:00:00+00:00",
)

# new knowledge soft-deprecates the old fact (nothing is hard-deleted)
mem.remember("The capital of France is Paris (updated)", entities=["France"])
```

### CLI

```bash
# store
agent-memory --db memory.db add "Sakya builds agent-memory" --entity Sakya --entity agent-memory

# hybrid search
agent-memory --db memory.db query "what does Sakya work on" -k 5

# timeline replay
agent-memory --db memory.db timeline --entity Sakya

# entity graph neighborhood
agent-memory --db memory.db graph --entity agent-memory

# preview conflicts
agent-memory --db memory.db conflicts "some new fact" --entity France

# soft-delete
agent-memory --db memory.db deprecate <memory-id>
```

### As an MCP server (optional)

```bash
pip install agent-memory[mcp]
python mcp_server.py --db memory.db
```

Exposes `remember`, `query`, `timeline` tools over stdio so any MCP client can
give your agent durable memory.

## How it works

```
  Agent (API / CLI / MCP)
          │
   ┌──────▼──────┐
   │ AgentMemory │  high-level facade
   └──────┬──────┘
   ┌──────▼───────────────────────────────────┐
   │  store      SQLite + FTS5                │
   │  timeline   bitemporal filtering/replay  │
   │  conflict   soft-deprecation resolver    │
   │  retriever  BM25 + dense + RRF           │
   │  graph      entity relations (networkx)  │
   │  embedder   hashing / sentence-transformers
   └──────────────────────────────────────────┘
```

### Bitemporal model

| field | meaning | answers |
|-------|---------|---------|
| `effective_at` | when the fact becomes true in the world | "what is true right now?" |
| `created_at` | when the fact was discovered | "what did we know at T?" |
| `deprecated` / `replaced_by` | soft-deletion link | "what superseded this?" |

### Conflict resolution

The default judge is a cheap heuristic (entity overlap × vector similarity ×
lexical distance). Plug in an **LLM-as-a-judge** by passing a custom `judge`:

```python
mem.conflicts.judge = lambda new, old: llm_judge(new.content, old.content)
```

Same "deterministic gate + LLM judgement" split used in production Agent
orchestration — the pattern is identical to a fail-closed evaluator layer.

## Project layout

```
agent-memory/
├── agent_memory/
│   ├── memory.py       # AgentMemory facade
│   ├── models.py       # Memory / Entity / Relation / ScoredMemory
│   ├── store.py        # SQLite + FTS5 persistence
│   ├── timeline.py     # bitemporal queries & replay
│   ├── conflict.py     # soft-deprecation resolver
│   ├── retriever.py    # BM25 + dense + RRF
│   ├── graph.py        # entity graph traversal
│   ├── embedder.py     # pluggable embeddings
│   └── cli.py          # command-line interface
├── mcp_server.py       # optional MCP server
├── examples/quickstart.py
└── tests/              # 17 unit tests (unittest, zero-dep)
```

## Roadmap

- [ ] Cross-backend abstraction for Neo4j / Milvus
- [ ] LLM-as-a-judge conflict evaluator out of the box
- [ ] Streaming timeline visualizer
- [ ] `agent-memory[st]` embedder docs

## License

MIT © [Sakya](https://github.com/Sakyao)
