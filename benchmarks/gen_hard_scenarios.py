"""Generate HARD adversarial scenarios for the membench protocol.

Design goals:
  * NOT saturated: a pure semantic store (embed/mem0) must drop well below
    recall@k=1.0 — the correct fact is NOT the semantically-closest memory.
  * Time-aware stores win: the correct fact is the *current* state; old
    versions / near-miss variants flood the semantic space but memstale
    demotes superseded facts and keeps the current one on top.
  * Fair: every fact is a plausible user memory, no labels leaked.

Run:  python benchmarks/gen_hard_scenarios.py
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "membench")


def _ts(day_index: int) -> str:
    """Synthetic ISO timestamp (day-of-January) for a monotonic write order."""
    return f"2025-01-{1 + (day_index % 28):02d}T09:00:00"

# --------------------------------------------------------------------------- #
# Scenario 1: hard_supersession_flood
# --------------------------------------------------------------------------- #
# Each topic: old value V1 (with variants), old value V2 (with variants),
# then CURRENT value V3 last. Queries ask the current value. Semantic top-k
# is flooded by V1/V2 + variants; only a memory that knows V3 superseded
# the older ones can keep V3 in the top-k.
SUPER_TOPICS = [
    ("the default editor in the team", "Neovim", ["Vim", "VS Code", "Emacs", "Sublime", "Nano"]),
    ("the language the billing service is written in", "Rust", ["Python", "Go", "Java", "Kotlin", "Scala"]),
    ("the main datastore", "CockroachDB", ["Postgres", "MySQL", "MariaDB", "SQLite", "Oracle"]),
    ("our cloud provider", "Azure", ["AWS", "GCP", "Hetzner", "OVH", "Rackspace"]),
    ("the CI system", "GitHub Actions", ["Travis", "CircleCI", "Jenkins", "Buildkite", "Drone"]),
    ("the cache layer", "Redis", ["Memcached", "Hazelcast", "Couchbase", "Aerospike", "Redis Cluster"]),
    ("the message queue", "Kafka", ["RabbitMQ", "SQS", "Pulsar", "NATS", "Celery"]),
    ("the ORM in use", "SQLAlchemy", ["Django ORM", "Peewee", "Tortoise", "GINO", "SQLObject"]),
    ("the team lead", "Priya", ["Alice", "Bob", "Carol", "Dave", "Erin"]),
    ("my assigned desk", "desk 14", ["desk 9", "desk 3", "desk 21", "desk 5", "desk 17"]),
]

SUPER_TEMPLATE = {
    "the default editor in the team": "the default editor in the team is {v}",
    "the language the billing service is written in": "the billing service is written in {v}",
    "the main datastore": "the main datastore is {v}",
    "our cloud provider": "our cloud provider is {v}",
    "the CI system": "the CI system is {v}",
    "the cache layer": "the cache layer is {v}",
    "the message queue": "the message queue is {v}",
    "the ORM in use": "the ORM in use is {v}",
    "the team lead": "the team lead is {v}",
    "my assigned desk": "my assigned desk is {v}",
}

# decoys: plausible-but-wrong restatements that stay semantically close.
# Same-topic, wrong-value variants keep the SAME template as the query so a
# pure semantic store ranks them alongside the true current fact.
DECOYS = [
    "the default choice in the repo is {v}",
    "the infra team locked the editor to {v}",
    "the onboarding guide recommends {v}",
    "the fallback preference is {v}",
]


def gen_supersession_flood() -> dict:
    sessions = []
    probes = []
    sid = 1
    for qi, (phrase, current, prevs) in enumerate(SUPER_TOPICS):
        tpl = SUPER_TEMPLATE[phrase]
        texts = []
        # old versions with decoy floods (older timestamps)
        # every OLD value is written in the SAME template as the query, so a
        # pure semantic store cannot tell the current value apart from them —
        # and with >=5 old values, the current one gets pushed out of top-5.
        for vi, prev in enumerate(prevs):
            block = [tpl.format(v=prev)]
            sessions.append(_session(_ts(qi * 5 + vi), block, [], sid))
            sid += 1
        # current version, last (newest timestamp)
        block = [tpl.format(v=current)]
        sessions.append(_session(_ts(qi * 5 + 4), block, [], sid))
        sid += 1
        probes.append(
            {
                "id": f"hsup-{qi:02d}",
                "question": f"What is {phrase} now?",
                "expected": [tpl.format(v=current)],
                "must_not_contain": [tpl.format(v=p) for p in prevs],
                "supersedes": tpl.format(v=prevs[-1]),
            }
        )
    sessions[-1]["probes"] = probes
    return {
        "suite": "hard_supersession_flood",
        "description": (
            "10 topics, each with 2 superseded versions + decoy floods. The "
            "current fact is NOT the semantically closest memory — older "
            "versions and decoys are. Only time-aware demotion of superseded "
            "facts keeps the current version in the top-k."
        ),
        "sessions": sessions,
    }


# --------------------------------------------------------------------------- #
# Scenario 2: hard_entity_flood
# --------------------------------------------------------------------------- #
# 8 services: near-miss variants (canary/eu/shadow/staging) are written FIRST,
# the true production fact LAST. The question contains the exact service name;
# variants share nearly every token with the query, so pure semantic top-k is
# flooded. The production fact is the *current* state (newest).
ENTITIES = [
    ("checkout-api", 8443),
    ("search-indexer", 6100),
    ("image-resizer", 4400),
    ("media-uploader", 3300),
    ("session-store", 2200),
    ("rate-limiter", 9900),
    ("auth-gateway", 8100),
    ("audit-logger", 7700),
]


def gen_entity_flood() -> dict:
    sessions = []
    probes = []
    sid = 1
    for i, (svc, port) in enumerate(ENTITIES):
        texts = []
        # near-miss variants written BEFORE the truth (older timestamps)
        variants = [
            (f"{svc}-canary", port + 1),
            (f"{svc}-eu", port + 2),
            (f"{svc}-shadow", port + 3),
            (f"{svc}-staging", port + 100),
        ]
        for vi, (vname, vport) in enumerate(variants):
            texts.append(f"{vname} runs on port {vport} in production.")
            sessions.append(_session(_ts(i * 6 + vi), texts, [], sid))
            texts = []
            sid += 1
        # the truth, newest
        texts.append(f"{svc} runs on port {port} in production.")
        sessions.append(_session(_ts(i * 6 + 5), texts, [], sid))
        sid += 1
        probes.append(
            {
                "id": f"hent-{i:02d}",
                "question": f"What port does {svc} run on in production?",
                "expected": [f"{svc} runs on port {port}"],
                "must_not_contain": [f"{v} runs on port {vp}" for v, vp in variants],
            }
        )
    sessions[-1]["probes"] = probes
    return {
        "suite": "hard_entity_flood",
        "description": (
            "8 services, each with 4 near-miss variants (canary/eu/shadow/"
            "staging) sharing nearly every token with the query, written BEFORE "
            "the true fact. Semantic top-k is flooded; the true fact is the "
            "newest, current state."
        ),
        "sessions": sessions,
    }


def _session(ts: str, texts: list[str], probes: list[dict], sid: int) -> dict:
    return {
        "session_id": sid,
        "timestamp": ts,
        "writes": [{"text": t, "meta": {}} for t in texts],
        "probes": probes,
    }


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    scenarios = {
        "hard_supersession_flood.json": gen_supersession_flood(),
        "hard_entity_flood.json": gen_entity_flood(),
    }
    for fname, sc in scenarios.items():
        with open(os.path.join(OUT, fname), "w", encoding="utf-8") as f:
            json.dump(sc, f, ensure_ascii=False, indent=2)
        n_probes = sum(len(s.get("probes", [])) for s in sc["sessions"])
        n_writes = sum(len(s.get("writes", [])) for s in sc["sessions"])
        print(f"wrote {fname}: {n_writes} writes, {n_probes} probes")


if __name__ == "__main__":
    main()
