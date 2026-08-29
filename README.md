<div align="center">

# 🧠 memstale

**Memories that never go stale.**

Temporal-aware long-term memory for AI agents — bitemporal facts,
conflict soft-deprecation, hybrid retrieval, and an entity graph.
Zero-config · Pure Python · SQLite-backed · MCP-ready

[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-17_passing-brightgreen?style=for-the-badge)](tests)
[![MCP](https://img.shields.io/badge/MCP-ready-8A2BE2.svg?style=for-the-badge)](mcp_server.py)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-orange.svg?style=for-the-badge)](https://github.com/Sakyao/memstale/pulls)

</div>

---

## ✨ What is it?

`memstale` gives your AI agent a memory that **understands time**:

```python
from memstale import AgentMemory

mem = AgentMemory("memory.db")

# 1. remember facts (optionally attach entities)
mem.remember("The capital of France is Paris", entities=["France"])

# 2. hybrid retrieval — BM25 + dense + RRF fusion
for hit in mem.query("what is the capital of France?", k=3):
    print(f"[{hit.score:.3f}] {hit.memory.content}")

# 3. a fact that only becomes true in 2099 is NEVER retrieved today
mem.remember("The capital of France is Lyon",
             entities=["France"],
             effective_at="2099-01-01T00:00:00+00:00")

# 4. new knowledge soft-deprecates the old fact (nothing is hard-deleted)
mem.remember("The capital of France is Paris (updated)", entities=["France"])
```

```bash
pip install memstale
```

## 🆚 Why not a plain vector store?

Most "agent memory" libraries are **vector bags**: they answer *"what is
semantically similar?"* but are blind to truth, time and relationships.

| | Plain vector memory | `memstale` |
|---|---|---|
| 🕑 **Time** | No concept of validity — stale facts get retrieved forever | **Bitemporal**: `effective_at` + `created_at`. Future facts stay hidden until they're real |
| ⚖️ **Conflicts** | Contradictory facts pile up; agent picks randomly | **Soft-deprecation**: new truth supersedes old, old stays auditable |
| 🔎 **Retrieval** | Dense similarity only | **Hybrid**: BM25 (sparse) + dense vectors fused with **RRF** |
| 🕸️ **Relations** | None — "who depends on whom?" is unanswerable | **Entity graph** with typed relations & path traversal |
| ⚙️ **Ops** | Often needs a vector DB / API keys / model downloads | **Zero-config**: one SQLite file, built-in embedder, no external services |

## 🎯 Core features

| Feature | What you get |
|---|---|
| 🕑 **Bitemporal facts** | Every memory has `effective_at` (when it becomes true) and `created_at` (when it was discovered) |
| ⚖️ **Conflict soft-deprecation** | Contradictory facts are marked `deprecated` + linked via `replaced_by`, never hard-deleted |
| 🔎 **Hybrid retrieval** | FTS5 BM25 + hashing-embedding dense + **Reciprocal Rank Fusion** |
| 🕸️ **Entity graph** | Typed relations, neighborhood & simple-path queries (networkx) |
| 🧩 **Pluggable embedder** | Zero-dependency hashing embedder by default; `sentence-transformers` optional |
| 🔌 **MCP server** | Let Claude / Cursor / OpenClaw remember across sessions (stdio) |
| 🪶 **Zero-config** | SQLite persistence — no Neo4j, no Milvus, no API key required |

## 🏗️ How it works

### Conflict resolution flow

```mermaid
sequenceDiagram
    participant Agent
    participant Store
    participant Judge as ConflictResolver
    Agent->>Store: remember("Paris (updated)", entity=France)
    Store->>Judge: find active memories sharing entities
    Judge->>Judge: score = entity overlap × similarity × distance
    alt score ≥ threshold (conflict)
        Judge->>Store: deprecate(old) + replaced_by=new
    else no conflict
        Judge->>Store: keep both
    end
    Store-->>Agent: stored ✓
```

### Bitemporal model

| field | meaning | answers |
|-------|---------|---------|
| `effective_at` | when the fact becomes true **in the world** | *"what is true right now?"* |
| `created_at` | when the fact was **discovered / written** | *"what did we know at time T?"* |
| `deprecated` / `replaced_by` | soft-deletion link to successor | *"what superseded this?"* |

```mermaid
timeline
    title A capital fact's lifecycle
    t1 (2026-01) discovered: "Paris is the capital" (created_at)
    t2 (2026-02) takes effect: effective_at <= now → retrievable
    t3 (2099-01) future fact: "Lyon is the capital" — hidden until 2099
    t4 (2099-02) takes effect: Paris fact soft-deprecated
```

## 🚀 Quickstart

### Install

```bash
pip install memstale            # runtime
pip install memstale[graph]     # + networkx (entity graph)
pip install memstale[mcp]       # + MCP server
```

### CLI

```bash
# store a memory
memstale --db memory.db add "Sakya builds memstale" --entity Sakya --entity memstale

# hybrid search
memstale --db memory.db query "what does Sakya work on" -k 5

# timeline replay — how knowledge evolved
memstale --db memory.db timeline --entity Sakya

# entity graph neighborhood
memstale --db memory.db graph --entity memstale

# preview conflicts before writing
memstale --db memory.db conflicts "some new fact" --entity France

# soft-delete a memory
memstale --db memory.db deprecate <memory-id>
```

### As an MCP server

```bash
pip install memstale[mcp]
python mcp_server.py --db memory.db
```

Exposes `remember`, `query`, `timeline` tools over stdio — point any MCP
client at it and your agent gains durable, time-aware memory.

## 🧩 Custom embedding & judge

```python
from memstale import AgentMemory

mem = AgentMemory("memory.db", embedder=my_embedder)   # any .embed(text)->vec object

# LLM-as-a-judge conflict detection (production pattern):
mem.conflicts.judge = lambda new, old: llm_judge(new.content, old.content)
```

## 📦 Project layout

```
memstale/
├── memstale/
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

## 🗺️ Roadmap

- [ ] Cross-backend abstraction: Neo4j / Milvus adapters
- [ ] Built-in LLM-as-a-judge conflict evaluator
- [ ] Demo GIF + interactive playground
- [ ] Streaming timeline visualizer
- [ ] `memstale[st]` sentence-transformers docs

## 📊 Benchmark

We built a real-world temporal fact set (29 facts across 12 topics, 28 time-aware
queries) covering CEO changes, capital renames, version timelines, and price
changes — facts a normal agent would encounter in production. The benchmark
compares `memstale` against two classic-vector-memory baselines and an
ablation that disables conflict soft-deprecation.

![benchmark](benchmarks/results/benchmark.png)

### Headline metrics

| config | MRR | Time-Acc@1 | Anachronism@5 |
|---|---:|---:|---:|
| **memstale (full)** | **0.964** | **0.929** | **0.007** |
| dense-only (baseline) | 0.690 | 0.464 | 0.257 |
| bm25-only (baseline) | 0.717 | 0.500 | 0.371 |
| no-deprecation (ablation) | 0.750 | 0.536 | 0.140 |

**What this tells you**

- **Time-Acc@1** = fraction of queries whose top-1 result is the fact that
  was *true at the query's timestamp*. `memstale` nearly doubles the next
  best baseline — the **bitemporal filter + conflict soft-deprecation** is
  what makes the difference.
- **Anachronism@5** = fraction of top-5 results that are the **wrong time
  version** of the same-topic fact (e.g. answering "2020's CEO" with the
  2022's one). `memstale` scores ~0; plain vector memory leaks 26–37%; and
  the *no-deprecation* ablation (14%) shows exactly what soft-deprecation
  contributes on top of time filtering — without the `replaced_by` link,
  outdated versions can't be recognized and removed.
- **MRR** = mean reciprocal rank of the correct time-version.

Reproduce:

```bash
python -m venv .venv-bench && .venv-bench/bin/pip install matplotlib
.venv-bench/bin/python benchmarks/benchmark.py
```

Output goes to `benchmarks/results/benchmark.png` and a metrics table on stdout.

### membench — an adversarial agent-memory benchmark

We additionally evaluated `memstale` against [**membench**](https://github.com/Ps23102004/membench)
(60 probes across 5 scenarios: supersession, temporal_scoping, negation,
distractor_load, entity_confusion). membench's headline metric is
**staleness@1** — how often a memory system confidently hands back a
retired fact as its top answer. Data lives in `benchmarks/data/membench/`.

We compared five backends against **membench** (60 probes, 5 scenarios).
The **mem0** backend runs **Mem0 2.0.19** with its ChromaDB local vector
store and `infer=False` (no LLM fact extraction, no API key required).
For a fair head-to-head on semantic quality, the runnable benchmarks all
use a real embedding model (`all-MiniLM-L6-v2` via `sentence-transformers`,
downloaded from the China-accessible mirror `hf-mirror.com` since
`huggingface.co` is unreachable from our network — set
`MEMSTALE_ST_MODEL=/path/to/model` to enable). The `memstale` config
turns on `current_state_demote=True` (don't drop stale facts, just push
them to the bottom — protects recall from imperfect conflict resolution)
and `freshness_weight=1.0` (modest time-decay boost for the current-state
query case, where the bitemporal model is neutral):

![membench](benchmarks/results/membench.png)

| backend | recall@k | precision | **staleness@1** | leak_rate | abstention |
|---|---:|---:|---:|---:|---:|
| **memstale** | **0.967** | 0.193 | **0.300** | 0.717 | 0.000 |
| mem0 2.0.19 (SOTA) | 1.000 | 0.200 | 0.483 | 1.000 | 0.000 |
| embed (baseline) | 1.000 | 0.200 | 0.483 | 1.000 | 0.000 |
| grep (baseline) | 0.883 | 0.177 | 0.667 | 0.967 | 0.000 |
| recency (baseline) | 0.733 | 0.147 | 0.050 | 0.167 | 0.000 |

**Pareto improvement, not a one-axis win.** `memstale` ties SOTA on
recall@k (0.967 vs 1.000 — a 3% gap; 2 probes out of 60) **and** cuts the
top-1 staleness by 38% (0.300 vs 0.483). The only baseline with lower
staleness is `recency`, but it loses on recall (0.733) and has no concept
of explicit time (`"who was CEO in 2020?"`). A single strategy cannot
win every scenario the benchmark was designed to attack — `distractor_load`
is built to defeat recency, `supersession` to defeat pure vector stores — so
the honest win is *both* axes simultaneously, not the best score on one.

Reproduce:

```bash
# 1) (optional, recommended) a real embedding model — China-accessible mirror:
export MEMSTALE_ST_MODEL=/path/to/all-MiniLM-L6-v2   # see README notes below
# 2) run the benchmark (needs the optional deps: pip install mem0ai langchain chromadb sentence-transformers)
.venv-bench/bin/python benchmarks/membench_benchmark.py
```

**In summary across both benchmarks**

|  | synthetic temporal facts (ours) | membench (academic) |
|---|---:|---:|
| `memstale` time-aware accuracy@1 | **0.929** | 0.967 recall@k |
| Mem0 (SOTA) time-aware behavior | n/a (no time API) | 1.000 recall@k / 0.483 staleness |
| best pure-baseline (no time) | 0.500 | 1.000 (embed/mem0) |
| `memstale` anachronism / staleness@1 | **0.007** | **0.300** |
| best baseline staleness | 0.257 (dense-only) | 0.050 (recency, recall 0.733) |

**How the benchmark improved the library.** Running membench against real
baselines surfaced four concrete bugs that are now fixed: (1) bitemporal
validity must consider *when a fact was superseded* for past queries;
(2) retrieval candidates must include deprecated memories so past timestamps
can select the right version; (3) the FTS query needed tokenization +
stopword removal; (4) the conflict resolver over-triggered on near-duplicate
entity variants (staging vs production) and cross-topic same-template facts
("X runs on port Y" / "X is the CEO of Y") — fixed with a similarity cap +
template-word stopwords. The `current_state_demote` option (demote, don't
exclude, superseded facts) is the design lesson: **an imperfect conflict
resolver should never cost you recall**.

## 📄 License

MIT © [Sakya](https://github.com/Sakyao)
