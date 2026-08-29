"""Run memstale against the membench agent-memory benchmark.

membench (github.com/Ps23102004/membench) is an adversarial benchmark for
agent memory backends. Its headline metric, **staleness@1**, measures how
often a memory system's *top-ranked* answer is a fact that used to be true
but no longer is — exactly the failure memstale is designed to eliminate.

Protocol (from membench, MIT):
  * scenarios are session streams: each session writes facts at a timestamp,
    then asks probes about the *current* state
  * a probe declares `expected` substrings and `must_not_contain` substrings
  * `staleness@1` = fraction of answered probes whose top-1 result contains
    a must_not substring (i.e. the retired/outdated fact)
  * `recall@k` = fraction of probes whose top-k contains an expected substring

Backends compared:
  * memstale        — bitemporal memory + conflict soft-deprecation + hybrid RRF
  * embed (baseline)- pure dense embedding top-k, no time awareness
  * grep (baseline) — keyword matching only
  * recency (baseline) — naive fix: semantic candidates re-ranked newest-first

The original membench `embed`/`recency` backends use Ollama's
nomic-embed-text; here we use memstale's built-in hashing embedder so the
whole benchmark runs offline with zero external services.

Run:
    python benchmarks/membench_benchmark.py
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memstale import AgentMemory, HashingEmbedder  # noqa: E402

# Use a real semantic embedder (sentence-transformers) when available via
# MEMSTALE_ST_MODEL=<path to a local model dir>. Strong embeddings unlock a
# semantic conflict judge that recognizes rewrite-style supersession
# ("I use tabs" -> "everything is spaces now"). With only the zero-dependency
# hashing embedder, the conservative topic-Jaccard judge stays in effect.
ST_MODEL = os.environ.get("MEMSTALE_ST_MODEL", "")


def make_embedder():
    if ST_MODEL:
        from memstale.embedder import SentenceTransformerEmbedder  # noqa: PLC0415

        return SentenceTransformerEmbedder(ST_MODEL)
    return HashingEmbedder()


def make_judge(embedder):
    """Semantic conflict judge: strong embeddings + topic-Jaccard fallback.

    Recognizes rewrite-style supersession (low lexical overlap, high semantic
    similarity) while keeping the cross-topic guard from the hashing-only mode.
    """
    from memstale.conflict import _content_tokens  # noqa: PLC0415

    def judge(new, existing):
        v1 = embedder.embed(new.content)
        v2 = embedder.embed(existing.content)
        sim = float(v1 @ v2)
        nw = _content_tokens(new.content)
        ew = _content_tokens(existing.content)
        union = nw | ew
        jac = len(nw & ew) / len(union) if union else 0.0
        if jac >= 0.85:
            # near-duplicate text -> same fact, different label -> NOT a contradiction
            return sim * 0.1
        if jac >= 0.7 and sim >= 0.80:
            # same template, only the VALUE changed ("editor is Vim" -> "... Neovim"):
            # a genuine update, even though semantic similarity is near-identical.
            # Distinguishable from env variants (staging/production) because those
            # add tokens, dropping Jaccard below ~0.7.
            return sim
        if jac >= 0.2:
            # same topic, updated value -> contradiction
            return sim * (0.4 + 0.6 * min(jac * 2, 1.0))
        # Low lexical overlap: ambiguous. Two near-identical entity names
        # (checkout-api vs checkout-api-staging) or rewrite-style supersession
        # (tabs -> spaces) both land here; soft-deprecating on pure similarity
        # wrongly kills production facts in entity-confusion scenarios, so we
        # stay conservative and let retrieval (with real embeddings) decide.
        return sim * 0.1

    return judge

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "membench")
OUT = os.path.join(HERE, "results")


# --------------------------------------------------------------------------- #
# grader + metrics (adapted from membench, MIT)
# --------------------------------------------------------------------------- #
def _contains_any(texts: list[str], needles: list[str]) -> bool:
    if not needles:
        return False
    haystack = "\n".join(texts).lower()
    return any(n.lower() in haystack for n in needles)


def grade_probe(probe: dict, returned: list[str]) -> dict:
    expected = probe.get("expected", [])
    must_not = probe.get("must_not_contain", [])
    hit = _contains_any(returned, expected)
    hit1 = _contains_any(returned[:1], expected)  # top-1 must be the answer
    answered = bool(returned)
    stale_at_1 = (_contains_any(returned[:1], must_not) if must_not else None) if answered else None
    leaked = (_contains_any(returned, must_not) if must_not else None) if answered else None
    tokens = sum(len(r) for r in returned) / 4
    relevant = sum(1 for r in returned if _contains_any([r], expected))
    precision = relevant / len(returned) if returned else 0.0
    return {
        "hit": hit,
        "hit1": hit1,
        "answered": answered,
        "stale_at_1": stale_at_1,
        "leaked": leaked,
        "precision": precision,
        "tokens": tokens,
    }


def aggregate(records: list[dict]) -> dict:
    n = len(records)
    recall = sum(1 for r in records if r["hit"]) / n
    recall1 = sum(1 for r in records if r["hit1"]) / n
    precision = sum(r["precision"] for r in records) / n
    abstention = sum(1 for r in records if not r["answered"]) / n
    stale_records = [r for r in records if r["stale_at_1"] is not None]
    staleness = sum(1 for r in stale_records if r["stale_at_1"]) / len(stale_records) if stale_records else None
    leak_records = [r for r in records if r["leaked"] is not None]
    leak = sum(1 for r in leak_records if r["leaked"]) / len(leak_records) if leak_records else None
    return {
        "n_probes": n,
        "recall@k": round(recall, 3),
        "recall@1": round(recall1, 3),
        "precision@k": round(precision, 3),
        "staleness@1": round(staleness, 3) if staleness is not None else None,
        "leak_rate@k": round(leak, 3) if leak is not None else None,
        "abstention_rate": round(abstention, 3),
    }


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #
class MemstaleBackend:
    name = "memstale"

    def __init__(self):
        self.mem: AgentMemory | None = None

    def reset(self):
        embedder = make_embedder()
        self.mem = AgentMemory(
            auto_resolve=True,
            embedder=embedder,
            conflict_threshold=0.45 if ST_MODEL else 0.55,
            freshness_weight=1.0 if ST_MODEL else 0.0,
            current_state_demote=True,   # demote stale, don't exclude
            stale_penalty=0.02,
        )
        if ST_MODEL:
            self.mem.conflicts.judge = make_judge(embedder)

    def write(self, text: str, meta: dict | None = None) -> None:
        ts = (meta or {}).get("timestamp", "")
        # the fact becomes true when it is discovered in the session stream
        self.mem.remember(text, effective_at=ts, created_at=ts, source="session")

    def query(self, q: str, k: int) -> list[str]:
        return [s.memory.content for s in self.mem.query(q, k=k)]


class EmbedBackend:
    """Pure dense top-k, no time awareness — the classic vector-memory baseline."""

    name = "embed"

    def __init__(self):
        self.embedder = make_embedder()
        self._store: list[dict] = []

    def reset(self):
        self._store = []

    def write(self, text: str, meta: dict | None = None) -> None:
        self._store.append({"text": text, "ts": (meta or {}).get("timestamp", ""), "vec": self.embedder.embed(text)})

    def query(self, q: str, k: int) -> list[str]:
        qv = self.embedder.embed(q)
        scored = sorted(self._store, key=lambda it: float(qv @ it["vec"]), reverse=True)
        return [it["text"] for it in scored[:k]]


class GrepBackend:
    """Keyword overlap matching only."""

    name = "grep"
    _STOP = {"the", "is", "are", "was", "were", "a", "an", "and", "or", "of", "to", "in", "on", "at", "for", "do", "i", "my", "we", "it", "what", "who", "am", "still", "currently", "now", "right", "does", "how"}

    def __init__(self):
        self._store: list[str] = []

    def reset(self):
        self._store = []

    def write(self, text: str, meta: dict | None = None) -> None:
        self._store.append(text)

    def query(self, q: str, k: int) -> list[str]:
        qtoks = {t for t in re.findall(r"[a-z0-9]+", q.lower()) if len(t) > 2 and t not in self._STOP}
        scored = []
        for text in self._store:
            ttoks = {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2 and t not in self._STOP}
            scored.append((len(qtoks & ttoks), text))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored[:k]]


class RecencyBackend:
    """Naive staleness fix: wide semantic pool, re-ranked newest-first."""

    name = "recency"

    def __init__(self):
        self.embedder = make_embedder()
        self._store: list[dict] = []

    def reset(self):
        self._store = []

    def write(self, text: str, meta: dict | None = None) -> None:
        self._store.append({"text": text, "ts": (meta or {}).get("timestamp", ""), "vec": self.embedder.embed(text)})

    def query(self, q: str, k: int) -> list[str]:
        qv = self.embedder.embed(q)
        pool = sorted(self._store, key=lambda it: float(qv @ it["vec"]), reverse=True)[: max(k * 3, 10)]
        pool.sort(key=lambda it: it["ts"], reverse=True)
        return [it["text"] for it in pool[:k]]


class Mem0Backend:
    """Mem0 (state-of-the-art agent memory library, v2.0.19) — SOTA comparison.

    Runs fully offline: Mem0's `langchain` embedder is pointed at a custom
    embedder backed by the same hashing embedder used by every other backend,
    ChromaDB is the local vector store, and `infer=False` disables the LLM
    fact-extraction step so no API key or model download is needed.
    """

    name = "mem0"

    def __init__(self):
        self.mem = None
        self._tmpdir: str | None = None

    def reset(self):
        os.environ["MEM0_TELEMETRY"] = "False"  # offline: disable posthog
        from langchain.embeddings.base import Embeddings  # noqa: PLC0415
        from mem0 import Memory  # noqa: PLC0415

        class _HashEmbeddings(Embeddings):
            def __init__(self):
                self.e = make_embedder()

            def embed_documents(self, texts):
                return [self.e.embed(t).tolist() for t in texts]

            def embed_query(self, text):
                return self.e.embed(text).tolist()

        if self._tmpdir:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
        self._tmpdir = tempfile.mkdtemp(prefix="mem0-bench-")
        config = {
            "llm": {
                "provider": "ollama",
                "config": {"model": "unused", "ollama_base_url": "http://localhost:11434"},
            },
            "embedder": {"provider": "langchain", "config": {"model": _HashEmbeddings()}},
            "vector_store": {
                "provider": "chroma",
                "config": {"collection_name": "mem0", "path": self._tmpdir},
            },
        }
        self.mem = Memory.from_config(config)

    def write(self, text: str, meta: dict | None = None) -> None:
        # OSS Mem0 SDK has no timestamp support -> it is time-unaware, which is
        # exactly the property we want to measure against memstale.
        self.mem.add(text, infer=False, user_id="bench")

    def query(self, q: str, k: int) -> list[str]:
        res = self.mem.search(q, top_k=k, filters={"user_id": "bench"})
        return [r["memory"] for r in res.get("results", [])]


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #
def run_backend(backend, scenario: dict, k: int = 5) -> list[dict]:
    backend.reset()
    records = []
    for session in scenario["sessions"]:
        ts = session.get("timestamp", "")
        for w in session.get("writes", []):
            backend.write(w["text"], {"timestamp": ts, "session_id": session.get("session_id")})
        for p in session.get("probes", []):
            returned = backend.query(p["question"], k)
            records.append(grade_probe(p, returned))
    return records


def main():
    scenarios = {}
    for f in sorted(Path(DATA).glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        scenarios[data["suite"]] = data

    backends = [MemstaleBackend(), Mem0Backend(), EmbedBackend(), GrepBackend(), RecencyBackend()]

    print("suite               ", end="")
    for b in backends:
        print(f"{b.name:>12s}", end="")
    print("   (recall | staleness@1)")

    rows = {}
    for suite, sc in scenarios.items():
        rows[suite] = {}
        for b in backends:
            rec = run_backend(b, sc)
            agg = aggregate(rec)
            rows[suite][b.name] = agg
            print(f"{suite:18s} {b.name:>12s}  r={agg['recall@k']:5.3f} s={agg['staleness@1']}")
    print()

    # overall (all probes across suites)
    print("=== OVERALL (all scenarios combined) ===")
    overall = {}
    for b in backends:
        all_rec = []
        for sc in scenarios.values():
            all_rec += run_backend(b, sc)
        overall[b.name] = aggregate(all_rec)
    print(f"{'config':12s} {'recall@k':>8s} {'recall@1':>9s} {'precision':>9s} {'staleness@1':>11s} {'leak_rate':>9s}")
    for name, a in overall.items():
        print(f"{name:12s} {a['recall@k']:8.3f} {a['recall@1']:9.3f} {a['precision@k']:9.3f} {a['staleness@1']:11.3f} {a['leak_rate@k']:9.3f}")

    _plot(overall, rows)
    return overall


def _plot(overall: dict, rows: dict) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[skip] matplotlib not installed")
        return

    os.makedirs(OUT, exist_ok=True)
    colors = {"memstale": "#2E86AB", "mem0": "#7B2D8B", "embed": "#E4572E", "grep": "#F2A541", "recency": "#A23B72"}
    names = list(overall.keys())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # 1) staleness@1 (headline) — lower is better
    ax = axes[0]
    s = [overall[n]["staleness@1"] for n in names]
    bars = ax.bar(range(len(names)), s, color=[colors[n] for n in names])
    ax.set_title("staleness@1 (lower = better)\nretired fact served as the top answer")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylim(0, max(s) * 1.15 + 0.02)
    ax.bar_label(bars, fmt="%.2f", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # 2) recall@1 and recall@k — higher is better
    ax = axes[1]
    width = 0.38
    r1 = [overall[n]["recall@1"] for n in names]
    rk = [overall[n]["recall@k"] for n in names]
    bars1 = ax.bar([x - width / 2 for x in range(len(names))], r1, width, label="recall@1", color=[colors[n] for n in names], alpha=0.9)
    bars2 = ax.bar([x + width / 2 for x in range(len(names))], rk, width, label="recall@k", color=[colors[n] for n in names], alpha=0.45)
    ax.set_title("recall@1 vs recall@k (higher = better)")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # 3) staleness by suite
    ax = axes[2]
    suites = list(rows.keys())
    width = 0.2
    for i, n in enumerate(names):
        vals = [rows[su][n]["staleness@1"] for su in suites]
        ax.bar([x + i * width for x in range(len(suites))], vals, width, label=n, color=colors[n])
    ax.set_title("staleness@1 by suite (lower = better)")
    ax.set_xticks([x + 1.5 * width for x in range(len(suites))])
    ax.set_xticklabels([s.replace("_", "\n") for s in suites], fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("membench (agent memory benchmark) — memstale vs baselines", fontsize=13, y=1.03)
    fig.tight_layout()
    out = os.path.join(OUT, "membench.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nsaved chart -> {out}")


if __name__ == "__main__":
    main()
