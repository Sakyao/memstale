"""Optional MCP server exposing agent-memory to AI agents.

Run (requires the `mcp` extra)::

    pip install agent-memory[mcp]
    python mcp_server.py --db memory.db

Then point your MCP client (Claude, Cursor, OpenClaw, ...) at the stdio
entrypoint. This lets an agent *remember* and *retrieve* across sessions.

Tools exposed:
    remember   — store a memory (with conflict auto-resolution)
    query      — hybrid retrieve memories
    timeline   — replay knowledge evolution for an entity
"""

from __future__ import annotations

import argparse

from agent_memory import AgentMemory


def run(db_path: str = ":memory:") -> None:
    try:
        from mcp.server.fastmcp import FastMCP  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "mcp package not installed. Run: pip install agent-memory[mcp]"
        ) from exc

    mcp = FastMCP("agent-memory")
    mem = AgentMemory(db_path)

    @mcp.tool()
    def remember(content: str, entities: list[str] | None = None, source: str = "") -> str:
        """Store a memory. Existing conflicting memories are soft-deprecated."""
        m = mem.remember(content, entities=entities, source=source)
        return f"stored {m.id}"

    @mcp.tool()
    def query(question: str, k: int = 5) -> list[dict]:
        """Hybrid (BM25 + dense) retrieve relevant memories."""
        return [
            {"content": s.memory.content, "score": round(s.score, 4), "id": s.memory.id}
            for s in mem.query(question, k=k)
        ]

    @mcp.tool()
    def timeline(entity: str) -> list[dict]:
        """Replay how knowledge about an entity evolved over time."""
        ent = mem.store.find_entity_by_name(entity)
        items = mem.timeline_of(ent.id if ent else None)
        return [
            {
                "content": m.content,
                "effective_at": m.effective_at,
                "created_at": m.created_at,
                "deprecated": m.deprecated,
            }
            for m in items
        ]

    mcp.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="agent-memory MCP server")
    parser.add_argument("--db", default=":memory:", help="SQLite database path")
    args = parser.parse_args()
    run(args.db)
