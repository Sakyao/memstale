"""Hybrid retrieval + graph tests."""

import unittest

from agent_memory import AgentMemory, Relation


class TestRetrieval(unittest.TestCase):
    def setUp(self):
        self.mem = AgentMemory()
        self.mem.remember("Sakya builds an AI agent framework called Philo", entities=["Sakya", "Philo"])
        self.mem.remember("Philo supports tool calling and memory management", entities=["Philo"])
        self.mem.remember("The weather in Beijing is sunny", entities=["weather"])

    def test_semantic_retrieval(self):
        results = self.mem.query("what does Philo do")
        self.assertTrue(results)
        top = results[0].memory.content.lower()
        self.assertIn("philo", top)

    def test_entity_filter(self):
        entity = self.mem.store.find_entity_by_name("weather")
        results = self.mem.query("anything", entity_ids=[entity.id])
        self.assertEqual(len(results), 1)
        self.assertIn("weather", results[0].memory.content)

    def test_temporal_filter_excludes_inactive(self):
        self.mem.remember(
            "temporary notice",
            entities=["notice"],
            effective_at="2099-01-01T00:00:00+00:00",
        )
        results = self.mem.query("notice")
        self.assertNotIn("temporary notice", [r.memory.content for r in results])


class TestGraph(unittest.TestCase):
    def setUp(self):
        self.mem = AgentMemory()
        self.mem.remember("Sakya develops agent-memory", entities=["Sakya", "agent-memory"])
        self.mem.remember("agent-memory uses hybrid retrieval", entities=["agent-memory", "retrieval"])
        self.mem.add_relation("Sakya", "agent-memory", "develops")
        self.mem.add_relation("agent-memory", "retrieval", "uses")

    def test_neighbors(self):
        neighbors = self.mem.neighbors("agent-memory")
        names = {n.name for n in neighbors}
        self.assertIn("Sakya", names)
        self.assertIn("retrieval", names)

    def test_paths(self):
        paths = self.mem.paths("Sakya", "retrieval")
        self.assertTrue(paths)
        self.assertEqual(paths[0][0]["type"], "develops")

    def test_graph_summary(self):
        self.assertEqual(self.mem.graph.summary()["entities"], 3)


if __name__ == "__main__":
    unittest.main()
