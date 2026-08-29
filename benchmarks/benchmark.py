"""Benchmark memstale against baselines and ablations on real-world temporal facts.

Baselines / ablations compared:
  * memstale (full)      — hybrid BM25+dense+RRF retrieval, bitemporal filtering,
                           conflict soft-deprecation
  * dense-only           — plain dense vector retrieval over ALL memories
                           (no time filter, no deprecation) = classic vector memory
  * bm25-only            — plain FTS5 BM25 retrieval, no time filter
  * no-deprecation       — memstale retrieval but conflicts are NOT soft-deprecated

Metrics:
  * Recall@k (k=1..10)   — is the time-correct fact in the top-k?
  * MRR                   — mean reciprocal rank of the time-correct fact
  * Time-Acc@1           — fraction of queries where top-1 is the time-correct fact
  * Stale@k              — fraction of top-k results that are NOT the time-correct
                           version (outdated or future facts)

Run:
    python benchmarks/benchmark.py
"""

from __future__ import annotations

import json
import os
import sys
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memstale import AgentMemory  # noqa: E402
from memstale.timeline import is_effective_at  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "facts.json")
OUT = os.path.join(HERE, "results")
K_LIST = [1, 2, 3, 5, 8, 10]


def load_facts():
    with open(DATA, encoding="utf-8") as f:
        data = json.load(f)
    facts, queries = [], []
    for topic in data["topics"]:
        facts.extend(topic["facts"])
        for q in topic["queries"]:
            q = dict(q)
            q["topic"] = topic["topic"]
            queries.append(q)
    return facts, queries


def build_memory(facts, auto_resolve: bool) -> AgentMemory:
    """Write facts in discovery order (created_at ascending)."""
    mem = AgentMemory(auto_resolve=auto_resolve)
    for fact in sorted(facts, key=lambda x: x["created_at"]):
        mem.remember(
            fact["content"],
            entities=fact["entities"],
            effective_at=fact["effective_at"],
            created_at=fact["created_at"],
            source="real-world",
        )
    return mem


# --------------------------------------------------------------------------- #
# retrieval strategies
# --------------------------------------------------------------------------- #
def full_retrieve(mem, question, at, k):
    return [s.memory for s in mem.query(question, at=at, k=k)]


def dense_only_retrieve(mem, question, at, k):
    """Plain dense retrieval over ALL memories — the classic vector-memory baseline."""
    qv = mem.embedder.embed(question)
    scored = []
    for m in mem.store.list_memories(include_deprecated=True):
        mv = mem.embedder.embed(m.content)
        scored.append((float(qv @ mv), m))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored[:k]]


def bm25_only_retrieve(mem, question, at, k):
    return mem.store.search_fts(question, limit=k)


def evaluate(retrieve_fn, queries, store) -> dict:
    recall = {k: [] for k in K_LIST}
    stale = {k: [] for k in K_LIST}
    mrr, acc1 = [], []
    misses = []
    for q in queries:
        results = retrieve_fn(q["question"], q["at"], max(K_LIST))
        contents = [m.content for m in results]
        gold = q["gold"]
        for k in K_LIST:
            topk = contents[:k]
            recall[k].append(1.0 if topk and gold in topk else 0.0)
            # stale = fraction of top-k that were NOT the truth at query time
            invalid = [m for m in results[:k] if not is_effective_at(m, q["at"], store)]
            stale[k].append(len(invalid) / k if results else 1.0)
        for i, c in enumerate(contents):
            if c == gold:
                mrr.append(1.0 / (i + 1))
                break
        else:
            mrr.append(0.0)
        acc1.append(1.0 if contents and contents[0] == gold else 0.0)
        if gold not in contents[:max(K_LIST)]:
            misses.append(f"  {q['topic']:26s} at={q['at'][:10]} -> {gold[:40]}")
    return {
        "recall": {k: mean(recall[k]) for k in K_LIST},
        "stale": {k: mean(stale[k]) for k in K_LIST},
        "mrr": mean(mrr),
        "acc1": mean(acc1),
        "misses": misses,
    }


def main():
    facts, queries = load_facts()
    print(f"loaded {len(facts)} facts, {len(queries)} time-aware queries")

    mem_full = build_memory(facts, auto_resolve=True)
    mem_nodep = build_memory(facts, auto_resolve=False)

    results = {
        "memstale (full)": evaluate(lambda q, at, k: full_retrieve(mem_full, q, at, k), queries, mem_full.store),
        "dense-only (baseline)": evaluate(lambda q, at, k: dense_only_retrieve(mem_full, q, at, k), queries, mem_full.store),
        "bm25-only (baseline)": evaluate(lambda q, at, k: bm25_only_retrieve(mem_full, q, at, k), queries, mem_full.store),
        "no-deprecation (ablation)": evaluate(lambda q, at, k: full_retrieve(mem_nodep, q, at, k), queries, mem_nodep.store),
    }

    for name, r in results.items():
        if r["misses"]:
            print(f"\n=== misses: {name} ({len(r['misses'])}) ===")
            for m in r["misses"]:
                print(m)

    print("\n=== Headline metrics ===")
    print(f"{'config':28s} {'MRR':>6s} {'Time-Acc@1':>10s} {'Stale@5':>8s}")
    for name, r in results.items():
        print(f"{name:28s} {r['mrr']:6.3f} {r['acc1']:10.3f} {r['stale'][5]:8.3f}")

    print("\n=== Recall@k ===")
    header = f"{'config':28s}" + "".join(f"{k:>8d}" for k in K_LIST)
    print(header)
    for name, r in results.items():
        row = f"{name:28s}" + "".join(f"{r['recall'][k]:8.3f}" for k in K_LIST)
        print(row)

    _plot(results)
    return results


def _plot(results: dict) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        print("\n[skip] matplotlib not installed — run: pip install matplotlib")
        return

    os.makedirs(OUT, exist_ok=True)
    colors = {
        "memstale (full)": "#2E86AB",
        "dense-only (baseline)": "#E4572E",
        "bm25-only (baseline)": "#F2A541",
        "no-deprecation (ablation)": "#A23B72",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # 1) Recall@k curves
    ax = axes[0]
    for name, r in results.items():
        ax.plot(K_LIST, [r["recall"][k] for k in K_LIST], marker="o", label=name, color=colors[name])
    ax.set_title("Retrieval quality (Recall@k)")
    ax.set_xlabel("k")
    ax.set_ylabel("Recall@k")
    ax.set_xticks(K_LIST)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 2) Time-aware accuracy
    ax = axes[1]
    names = list(results.keys())
    acc = [results[n]["acc1"] for n in names]
    bars = ax.bar(range(len(names)), acc, color=[colors[n] for n in names])
    ax.set_title("Time-Aware Accuracy@1")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.split(" ")[0] for n in names], rotation=20, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # 3) Stale rate (outdated / future facts leaking into results)
    ax = axes[2]
    stale = [results[n]["stale"][5] for n in names]
    bars = ax.bar(range(len(names)), stale, color=[colors[n] for n in names])
    ax.set_title("Stale facts in top-5 (lower = better)")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.split(" ")[0] for n in names], rotation=20, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("memstale benchmark — real-world temporal facts", fontsize=13, y=1.03)
    fig.tight_layout()
    out = os.path.join(OUT, "benchmark.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nsaved chart -> {out}")


if __name__ == "__main__":
    main()
