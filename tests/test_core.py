"""Core storage + facade tests."""

import os
import tempfile
import unittest

from memstale import AgentMemory, Memory


class TestStoreCRUD(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.mem = AgentMemory(self.path)

    def tearDown(self):
        self.mem.close()
        os.unlink(self.path)

    def test_remember_and_recall(self):
        m = self.mem.remember("Sakya builds memstale", entities=["Sakya"])
        got = self.mem.recall(m.id)
        self.assertIsNotNone(got)
        self.assertEqual(got.content, "Sakya builds memstale")
        self.assertEqual(got.entity_ids, [m.entity_ids[0]])

    def test_remember_dedupes_entities(self):
        m1 = self.mem.remember("a", entities=["Sakya"])
        m2 = self.mem.remember("b", entities=["Sakya"])
        self.assertEqual(m1.entity_ids[0], m2.entity_ids[0])

    def test_active_memories_exclude_deprecated(self):
        m = self.mem.remember("old fact", entities=["x"])
        self.mem.deprecate(m.id)
        active = self.mem.active_memories()
        self.assertNotIn(m.id, [x.id for x in active])

    def test_persistence_across_reopen(self):
        self.mem.remember("persisted fact", entities=["y"])
        self.mem.close()
        mem2 = AgentMemory(self.path)
        try:
            self.assertEqual(len(mem2.active_memories()), 1)
        finally:
            mem2.close()


class TestFacade(unittest.TestCase):
    def test_memory_facade_roundtrip(self):
        with AgentMemory() as mem:
            mem.remember("the sky is blue", entities=["sky"])
            results = mem.query("what color is the sky")
            self.assertTrue(results)
            self.assertIn("sky", results[0].memory.content.lower())


if __name__ == "__main__":
    unittest.main()
