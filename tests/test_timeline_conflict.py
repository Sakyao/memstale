"""Bitemporal timeline + conflict soft-deprecation tests."""

import os
import tempfile
import unittest

from memstale import AgentMemory, Memory


def _mem(content, effective_at, created_at):
    return Memory(
        content=content,
        effective_at=effective_at,
        created_at=created_at,
    )


class TestTimeline(unittest.TestCase):
    def setUp(self):
        self.mem = AgentMemory()

    def test_effective_filtering_by_time(self):
        self.mem.store.add_memory(
            _mem("version 1", effective_at="2026-01-01T00:00:00+00:00", created_at="2026-01-02T00:00:00+00:00")
        )
        self.mem.store.add_memory(
            _mem("future version", effective_at="2099-06-01T00:00:00+00:00", created_at="2099-01-01T00:00:00+00:00")
        )
        # A memory that takes effect in the future must not appear now.
        active_now = self.mem.timeline.active_at("2026-03-01T00:00:00+00:00")
        self.assertEqual([m.content for m in active_now], ["version 1"])
        # At the future time point, both are effective.
        active_future = self.mem.timeline.active_at("2099-06-01T00:00:00+00:00")
        self.assertEqual({m.content for m in active_future}, {"version 1", "future version"})

    def test_knowledge_at_uses_discovery_time(self):
        self.mem.store.add_memory(
            _mem("fact A", effective_at="2026-01-01T00:00:00+00:00", created_at="2026-01-10T00:00:00+00:00")
        )
        self.mem.store.add_memory(
            _mem("fact B", effective_at="2026-01-01T00:00:00+00:00", created_at="2026-02-10T00:00:00+00:00")
        )
        known = self.mem.timeline.knowledge_at(None, "2026-01-15T00:00:00+00:00")
        self.assertEqual([m.content for m in known], ["fact A"])

    def test_effective_between_with_deprecation(self):
        old = self.mem.remember("old pricing", entities=["price"], effective_at="2026-01-01T00:00:00+00:00")
        new = self.mem.remember("new pricing", entities=["price"], effective_at="2026-07-01T00:00:00+00:00")
        self.mem.deprecate(old.id, replaced_by=new.id)
        window = self.mem.timeline.effective_between("2026-01-01T00:00:00+00:00", "2026-12-31T00:00:00+00:00")
        contents = {m.content for m in window}
        self.assertIn("old pricing", contents)  # deprecated but was true during part of the window
        self.assertIn("new pricing", contents)


class TestConflict(unittest.TestCase):
    def setUp(self):
        self.mem = AgentMemory(conflict_threshold=0.5)

    def test_conflicting_memory_soft_deprecates_old(self):
        self.mem.remember(
            "The capital of France is Paris",
            entities=["France"],
        )
        new = self.mem.remember(
            "The capital of France is Lyon",
            entities=["France"],
        )
        old = self.mem.active_memories()
        self.assertEqual(len(old), 1)
        self.assertEqual(old[0].id, new.id)
        # old memory is soft-deprecated and links to the new one
        all_mems = self.mem.store.list_memories(include_deprecated=True)
        deprecated = [m for m in all_mems if m.deprecated]
        self.assertEqual(len(deprecated), 1)
        self.assertEqual(deprecated[0].replaced_by, new.id)

    def test_non_conflicting_memory_survives(self):
        self.mem.remember("Sakya works on agents", entities=["Sakya"])
        self.mem.remember("The weather is sunny today", entities=["weather"])
        self.assertEqual(len(self.mem.active_memories()), 2)

    def test_conflicts_for_preview(self):
        self.mem.remember("The capital of France is Paris", entities=["France"])
        conflicts = self.mem.conflicts_for("The capital of France is Lyon", entities=["France"])
        self.assertTrue(conflicts)


if __name__ == "__main__":
    unittest.main()
