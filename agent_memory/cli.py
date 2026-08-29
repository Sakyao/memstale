"""Command-line interface for agent-memory.

Examples::

    agent-memory add "Sakya is building agent-memory" --entity Sakya --entity agent-memory
    agent-memory query "what is Sakya working on" -k 5
    agent-memory timeline --entity Sakya
    agent-memory graph --entity Sakya
    agent-memory deprecate <memory-id>
"""

from __future__ import annotations

import argparse
import sys

from .memory import AgentMemory


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-memory",
        description="Temporal-aware long-term memory for AI agents.",
    )
    parser.add_argument("--db", default=":memory:", help="SQLite path (default: :memory:)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="store a memory")
    p_add.add_argument("content")
    p_add.add_argument("--entity", action="append", default=[], dest="entities")
    p_add.add_argument("--source", default="")
    p_add.add_argument("--no-resolve", action="store_true", help="skip conflict resolution")
    p_add.add_argument("--show-id", action="store_true", help="print the new memory id")

    p_query = sub.add_parser("query", help="hybrid search")
    p_query.add_argument("question")
    p_query.add_argument("-k", type=int, default=5)

    p_tl = sub.add_parser("timeline", help="replay knowledge evolution")
    p_tl.add_argument("--entity", default=None)

    p_graph = sub.add_parser("graph", help="entity graph neighborhood")
    p_graph.add_argument("--entity", required=True)
    p_graph.add_argument("--depth", type=int, default=1)

    p_conf = sub.add_parser("conflicts", help="preview conflicts for a memory")
    p_conf.add_argument("content")
    p_conf.add_argument("--entity", action="append", default=[], dest="entities")

    p_dep = sub.add_parser("deprecate", help="soft-delete a memory")
    p_dep.add_argument("memory_id")

    p_list = sub.add_parser("list", help="list active memories")
    return parser


def _print_memory(idx: int, m) -> None:
    flag = "[DEPRECATED]" if m.deprecated else ""
    print(f"{idx}. {m.content}  {flag}")
    print(f"   id={m.id[:8]}  effective={m.effective_at[:10]}  created={m.created_at[:10]}")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    with AgentMemory(args.db) as mem:
        if args.command == "add":
            m = mem.remember(
                args.content,
                entities=args.entities,
                source=args.source,
                auto_resolve=not args.no_resolve,
            )
            print(f"stored {m.id}")
            if args.show_id:
                print(m.id)
        elif args.command == "query":
            results = mem.query(args.question, k=args.k)
            if not results:
                print("no memories found")
                return 0
            for i, item in enumerate(results, 1):
                _print_memory(i, item.memory)
                print(f"   score={item.score:.4f}")
        elif args.command == "timeline":
            entity = None
            if args.entity:
                ent = mem.store.find_entity_by_name(args.entity)
                entity = ent.id if ent else None
            for i, m in enumerate(mem.timeline_of(entity), 1):
                _print_memory(i, m)
        elif args.command == "graph":
            ent = mem.store.find_entity_by_name(args.entity)
            if ent is None:
                print(f"entity not found: {args.entity}")
                return 1
            neighbors = mem.neighbors(args.entity, depth=args.depth)
            names = [getattr(n, "name", str(n)) for n in neighbors]
            print(f"neighbors of '{args.entity}': {', '.join(names) or '(none)'}")
        elif args.command == "conflicts":
            conflicts = mem.conflicts_for(args.content, entities=args.entities)
            if not conflicts:
                print("no conflicts found")
                return 0
            for c in conflicts:
                print(f"- conflicts with: {c.existing_memory.content} (score={c.score:.3f})")
        elif args.command == "deprecate":
            ok = mem.deprecate(args.memory_id)
            print("deprecated" if ok else "memory not found")
        elif args.command == "list":
            for i, m in enumerate(mem.active_memories(), 1):
                _print_memory(i, m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
