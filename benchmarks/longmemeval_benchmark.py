"""Evaluate memstale vs baselines on **LongMemEval** — the industry-standard
long-term conversation memory benchmark (Mem0 et al. report SOTA on it).

Dataset (HF: xiaowu0162/longmemeval-cleaned, S split): 500 question samples.
Each sample:
  * `haystack_sessions` — the candidate conversation sessions (memory space)
  * `answer_session_ids` — the session(s) that ground the answer (gold)
  * `question` / `answer` — the memory-retrieval query and its answer

Protocol (session-level memory retrieval, no LLM judge):
  1. write each haystack session as one memory unit (turns joined with
     role markers)
  2. retrieve top-k sessions for the question
  3. recall@k: is the gold answer session in the top-k?  MRR too.

Run (requires the downloaded file):
    MEMSTALE_ST_MODEL=/path/to/model .venv-bench/bin/python benchmarks/longmemeval_benchmark.py /path/to/longmemeval_s_cleaned.json [n_samples]
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.membench_benchmark import (  # noqa: E402
    EmbedBackend,
    Mem0Backend,
    MemstaleBackend,
    RecencyBackend,
    make_embedder,
)

K_LIST = [1, 2, 3, 5, 10]


def load_samples(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = f.read()
    import json as j

    dec = j.JSONDecoder()
    obj, _ = dec.raw_decode(data.strip())
    return obj  # top-level is a list of 500 samples


def session_to_text(turns: list[dict]) -> str:
    parts = []
    for t in turns:
        role = t.get("role", "assistant")
        content = t.get("content", "").strip()
        if content:
            parts.append(f"{role}: {content}")
    return "\n".join(parts)


def evaluate(backend, samples, max_samples: int) -> dict:
    recall = {k: 0 for k in K_LIST}
    mrr_sum = 0.0
    n_questions = 0
    n_sessions = 0
    for sample in samples[:max_samples]:
        sessions = sample.get("haystack_sessions", [])
        gold_ids = set(sample.get("answer_session_ids", []))
        n_sessions += len(sessions)
        # write each session as a memory unit
        mem_to_session: dict[str, str] = {}
        for sid, turns in zip(sample.get("haystack_session_ids", []), sessions):
            text = session_to_text(turns)
            if not text.strip():
                continue
            m = backend.write(text, {})
            mem_to_session[getattr(m, "id", str(id(text)))] = sid
        question = sample.get("question", "")
        if not question or not gold_ids:
            continue
        n_questions += 1
        if n_questions % 10 == 0:
            print(f"  ... {n_questions} questions processed (recall@1={recall[1]/n_questions:.2f})", flush=True)
        results = backend.query(question, max(K_LIST))
        # map retrieved text back to session ids
        retrieved_sessions = []
        for text in results:
            sid = _find_session(text, sessions, sample.get("haystack_session_ids", []))
            retrieved_sessions.append(sid)
        hit = None
        for rank, sid in enumerate(retrieved_sessions, 1):
            if sid in gold_ids:
                hit = rank
                break
        if hit:
            mrr_sum += 1.0 / hit
            for k in K_LIST:
                if hit <= k:
                    recall[k] += 1
    total = max(n_questions, 1)
    return {
        "n_questions": n_questions,
        "n_sessions": n_sessions,
        "recall": {k: round(recall[k] / total, 3) for k in K_LIST},
        "mrr": round(mrr_sum / total, 3),
    }


def _find_session(text: str, sessions: list, session_ids: list) -> str | None:
    """Best-effort: which session produced this retrieved text?"""
    for sid, turns in zip(session_ids, sessions):
        if text == session_to_text(turns):
            return sid
    # fallback: prefix match on the first user turn
    for sid, turns in zip(session_ids, sessions):
        for t in turns:
            if t.get("content") and t["content"][:80] in text:
                return sid
    return None


def main(argv: list[str] | None = None) -> None:
    if not argv:
        argv = sys.argv[1:]
    if len(argv) < 1:
        print("usage: longmemeval_benchmark.py <longmemeval_s_cleaned.json> [n_samples]")
        sys.exit(1)
    path = argv[0]
    max_samples = int(argv[1]) if len(argv) > 1 else 30

    print(f"loading LongMemEval from {path} ...")
    samples = load_samples(path)
    n = min(max_samples, len(samples))
    print(f"samples: {len(samples)} (evaluating first {n})")

    backends = {
        "memstale": MemstaleBackend(),
        "mem0": Mem0Backend(),
        "embed": EmbedBackend(),
        "recency": RecencyBackend(),
    }

    print(f"\n{'backend':10s} {'questions':>9s} {'sessions':>8s} {'recall@1':>9s} {'recall@3':>9s} {'recall@5':>9s} {'recall@10':>10s} {'MRR':>6s}")
    results = {}
    for name, b in backends.items():
        try:
            b.reset()
            r = evaluate(b, samples, n)
            results[name] = r
            print(
                f"{name:10s} {r['n_questions']:9d} {r['n_sessions']:8d} "
                f"{r['recall'][1]:9.3f} {r['recall'][3]:9.3f} {r['recall'][5]:9.3f} "
                f"{r['recall'][10]:10.3f} {r['mrr']:6.3f}"
            )
        except Exception as e:  # pragma: no cover
            print(f"{name:10s} FAILED: {e}")
    _plot(results)


def _plot(results: dict) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    colors = {"memstale": "#2E86AB", "mem0": "#7B2D8B", "embed": "#E4572E", "recency": "#A23B72"}
    names = list(results.keys())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    # recall@k curves
    ax = axes[0]
    for name in names:
        xs = K_LIST
        ys = [results[name]["recall"][k] for k in xs]
        ax.plot(xs, ys, marker="o", label=name, color=colors[name])
    ax.set_title("LongMemEval: recall@k (higher = better)")
    ax.set_xlabel("k")
    ax.set_ylabel("recall@k")
    ax.set_xticks(xs)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    # MRR bars
    ax = axes[1]
    mrr = [results[n]["mrr"] for n in names]
    bars = ax.bar(range(len(names)), mrr, color=[colors[n] for n in names])
    ax.set_title("LongMemEval: MRR (higher = better)")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylim(0, max(mrr) * 1.2 + 0.02)
    ax.bar_label(bars, fmt="%.2f", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.suptitle("LongMemEval (industry-standard long-conversation memory benchmark)", fontsize=12, y=1.03)
    fig.tight_layout()
    out = os.path.join(out_dir, "longmemeval.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved chart -> {out}")


if __name__ == "__main__":
    main()
