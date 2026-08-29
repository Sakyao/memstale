"""Quickstart: hybrid retrieval + bitemporal facts + conflict soft-deprecation.

Run from the project root:  python examples/quickstart.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memstale import AgentMemory

mem = AgentMemory("quickstart.db")

# 1. Remember facts (optionally attach entities)
mem.remember("The capital of France is Paris", entities=["France"], source="handbook")
mem.remember("Philo supports tool calling and memory management", entities=["Philo"])

# 2. Query with hybrid retrieval (BM25 + dense, RRF fusion)
for hit in mem.query("what can Philo do?", k=3):
    print(f"[{hit.score:.3f}] {hit.memory.content}")

# 3. Bitemporal facts: schedule a fact that only takes effect in the future
mem.remember(
    "The capital of France is Lyon",
    entities=["France"],
    effective_at="2099-01-01T00:00:00+00:00",
)

print("\n-- active now --")
for m in mem.active_memories():
    print("-", m.content)

# 4. Conflict soft-deprecation: a newer fact supersedes the old one
new = mem.remember("The capital of France is Paris (updated)", entities=["France"])
deprecated = [m for m in mem.store.list_memories(include_deprecated=True) if m.deprecated]
print("\n-- soft-deprecated --")
for m in deprecated:
    print(f"- {m.content}  (replaced by {m.replaced_by})")

mem.close()
