"""End-to-end tests over the synthetic story, using the offline heuristic backend."""

import tempfile
import unittest
from datetime import datetime

from engram.config import EngramConfig
from engram.decisions import HeuristicBackend
from engram.retrieval import RetrievalLoop
from engram.store import MemoryStore
from engram.write import WritePipeline

STORY = [
    ("2026-03-02 09:10", "Noor switched to a vegan diet after watching a documentary about dairy farming.", "noor-vegan"),
    ("2026-03-02 09:12", "Sam has been afraid of heights since a childhood fall from a ladder.", "sam-fear"),
    ("2026-03-05 18:40", "Sam's doctor told him he needs regular cardio exercise for his heart health.", "sam-doctor"),
    ("2026-03-08 11:00", "Sam started climbing at Riverside gym because his doctor said he needed cardio.", "sam-climbing"),
    ("2026-03-12 20:05", "Noor's favorite cuisine is Ethiopian; she always orders the vegetable platter.", "noor-food"),
    ("2026-03-15 08:30", "Dev borrowed Noor's tripod and promised to return it before the Lisbon trip.", "dev-tripod"),
    ("2026-03-18 14:00", "Noor and Sam are planning a Lisbon trip in June for a friend's wedding.", "lisbon"),
    ("2026-03-21 19:45", "Sam no longer climbs at Riverside; he switched to Harbor gym because it got too crowded.", "sam-harbor"),
    ("2026-03-25 12:00", "hi", "greeting"),
    ("2026-03-26 10:15", "Noor is allergic to peanuts, so she always checks restaurant menus carefully.", "noor-peanuts"),
]


def build_store() -> tuple[MemoryStore, dict[str, int]]:
    store = MemoryStore()
    pipeline = WritePipeline(store, HeuristicBackend(), EngramConfig())
    ids = {}
    for iso_time, text, label in STORY:
        timestamp = datetime.fromisoformat(iso_time).timestamp()
        result = pipeline.ingest(text, timestamp)
        ids[label] = result.get("node_id")
    return store, ids


class WritePathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store, cls.ids = build_store()

    def test_greeting_filtered(self):
        self.assertFalse(self.ids["greeting"])

    def test_all_real_observations_stored(self):
        stored = [label for label, nid in self.ids.items() if nid]
        self.assertEqual(len(stored), len(STORY) - 1)

    def test_entity_edges_connect_noor_memories(self):
        neighbors = [other for other, _, _ in
                     self.store.neighbors(self.ids["noor-vegan"], "entity", limit=32)]
        self.assertIn(self.ids["noor-peanuts"], neighbors)

    def test_cause_edge_from_doctor_to_climbing(self):
        neighbors = [other for other, _, _ in
                     self.store.neighbors(self.ids["sam-climbing"], "cause", limit=32)]
        self.assertIn(self.ids["sam-doctor"], neighbors)

    def test_time_edge_chain(self):
        neighbors = [other for other, kind, _ in
                     self.store.neighbors(self.ids["sam-harbor"], "time", limit=32)]
        self.assertIn(self.ids["noor-peanuts"], neighbors)

    def test_superseded_flag_on_riverside(self):
        node = self.store.nodes[self.ids["sam-climbing"]]
        self.assertEqual(node.flags.get("superseded_by"), str(self.ids["sam-harbor"]))

    def test_preference_typed(self):
        scores = self.store.nodes[self.ids["noor-vegan"]].type_scores
        self.assertGreaterEqual(scores.get("type_like", 0.0), 0.5)


class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store, cls.ids = build_store()
        cls.loop = RetrievalLoop(cls.store, HeuristicBackend(), EngramConfig())

    def test_causal_query_reaches_doctor(self):
        evidence, trace = self.loop.query("Why did Sam start climbing?")
        found = {c.node.node_id for c in evidence}
        self.assertIn(self.ids["sam-doctor"], found)
        self.assertGreaterEqual(trace.budgets.get("cause", 0), 1)

    def test_multi_hop_query_finds_both_facts(self):
        evidence, _ = self.loop.query("Who borrowed Noor's tripod and when is the Lisbon trip?")
        found = {c.node.node_id for c in evidence}
        self.assertIn(self.ids["dev-tripod"], found)
        self.assertIn(self.ids["lisbon"], found)

    def test_recency_query_prefers_harbor(self):
        evidence, _ = self.loop.query("Which gym does Sam climb at now?")
        found = [c.node.node_id for c in evidence]
        self.assertIn(self.ids["sam-harbor"], found)
        harbor = next(c for c in evidence if c.node.node_id == self.ids["sam-harbor"])
        riverside = next((c for c in evidence if c.node.node_id == self.ids["sam-climbing"]), None)
        if riverside is not None:
            self.assertGreaterEqual(harbor.score, riverside.score)
        self.assertTrue(
            self.store.nodes[self.ids["sam-climbing"]].flags.get("superseded_by"))

    def test_trace_is_complete(self):
        _, trace = self.loop.query("What does Noor like to eat?")
        self.assertTrue(trace.routing)
        self.assertTrue(trace.evidence_ids)
        self.assertIn(trace.stop_reason,
                      {"anchors_only", "budget_exhausted", "enough_evidence",
                       "low_expected_value", "round_limit"})


class DeterminismAndPersistenceTests(unittest.TestCase):
    def test_build_is_deterministic(self):
        first_store, first_ids = build_store()
        second_store, second_ids = build_store()
        self.assertEqual(first_ids, second_ids)
        self.assertEqual(
            [n.text for n in sorted(first_store.nodes.values(), key=lambda n: n.node_id)],
            [n.text for n in sorted(second_store.nodes.values(), key=lambda n: n.node_id)])

    def test_persistence_roundtrip(self):
        store, _ = build_store()
        with tempfile.TemporaryDirectory() as directory:
            store.save(directory)
            restored = MemoryStore.load(directory)
        self.assertEqual(
            {n.node_id: n.text for n in store.nodes.values()},
            {n.node_id: n.text for n in restored.nodes.values()})
        for view in store.edges:
            self.assertEqual(len(store.edges[view]), len(restored.edges[view]))


if __name__ == "__main__":
    unittest.main()
