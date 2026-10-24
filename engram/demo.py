"""Offline demo: a small synthetic history, then four queries with full traces.

Runs with no API keys and no network — deterministic hashed embeddings and
heuristic decisions. Use ``--backend llm`` on the CLI for live quality.
"""

from __future__ import annotations

from datetime import datetime

from .answer import AnswerSynthesizer
from .config import EngramConfig
from .decisions import HeuristicBackend
from .retrieval import RetrievalLoop
from .store import MemoryStore
from .write import WritePipeline

STORY = [
    ("2026-03-02 09:10", "Noor switched to a vegan diet after watching a documentary about dairy farming."),
    ("2026-03-02 09:12", "Sam has been afraid of heights since a childhood fall from a ladder."),
    ("2026-03-05 18:40", "Sam's doctor told him he needs regular cardio exercise for his heart health."),
    ("2026-03-08 11:00", "Sam started climbing at Riverside gym because his doctor said he needed cardio."),
    ("2026-03-12 20:05", "Noor's favorite cuisine is Ethiopian; she always orders the vegetable platter."),
    ("2026-03-15 08:30", "Dev borrowed Noor's tripod and promised to return it before the Lisbon trip."),
    ("2026-03-18 14:00", "Noor and Sam are planning a Lisbon trip in June for a friend's wedding."),
    ("2026-03-21 19:45", "Sam no longer climbs at Riverside; he switched to Harbor gym because it got too crowded."),
    ("2026-03-25 12:00", "hi"),
    ("2026-03-25 12:01", "ok cool"),
    ("2026-03-26 10:15", "Noor is allergic to peanuts, so she always checks restaurant menus carefully."),
]

QUERIES = [
    "What dietary restrictions does Noor have?",
    "Why did Sam start climbing?",
    "Which gym does Sam climb at now?",
    "Who borrowed Noor's tripod and when is the Lisbon trip?",
]


def build_story(pipeline: WritePipeline) -> None:
    print("=" * 72)
    print("1. INGEST  (admission -> typing -> candidate search -> linking)")
    print("=" * 72)
    for iso_time, text in STORY:
        timestamp = datetime.fromisoformat(iso_time).timestamp()
        result = pipeline.ingest(text, timestamp)
        if result["stored"]:
            kinds = ", ".join(link.get("kind", link.get("flag", "?"))
                              for link in result["links"]) or "no links"
            print(f"  + node {result['node_id']:>2}  [{kinds}]")
            print(f"      {text}")
        else:
            print(f"  - {result['reason']:<9} {text}")


def show_store(store: MemoryStore) -> None:
    print("\n" + "=" * 72)
    print("2. MEMORY PLANE")
    print("=" * 72)
    print(f"  nodes: {len(store.nodes)}")
    for view, edges in store.edges.items():
        print(f"  {view:>7} edges: {len(edges)}")
    for node in store.nodes.values():
        if node.flags:
            print(f"  flags on #{node.node_id}: {node.flags}")


def run_queries(store: MemoryStore, config: EngramConfig) -> None:
    loop = RetrievalLoop(store, HeuristicBackend(), config)
    synthesizer = AnswerSynthesizer(config)
    print("\n" + "=" * 72)
    print("3. QUERY  (routing -> budgets -> anchors -> expand -> stop)")
    print("=" * 72)
    for question in QUERIES:
        evidence, trace = loop.query(question)
        print(f"\nQ: {question}")
        active = [key for key, value in trace.routing.items()
                  if key.startswith("need_") and value >= 0.35]
        print(f"  routing     : {', '.join(key[5:] for key in active)}")
        print(f"  budgets     : { {k: v for k, v in trace.budgets.items() if v} }")
        print(f"  stop reason : {trace.stop_reason}")
        for candidate in evidence:
            via = (f"via {candidate.via_view}/{candidate.via_kind}"
                   if candidate.via_view else "anchor")
            note = f"  flags={candidate.node.flags}" if candidate.node.flags else ""
            print(f"  * [{candidate.score:.2f}] {via}{note} #{candidate.node.node_id}")
            print(f"      {candidate.node.text[:70]}")
        answer = synthesizer.answer(question, [c.node for c in evidence])
        print(f"  A ({answer['mode']}): {answer['answer'][:200]}")


def run() -> None:
    print("Engram — offline demo (no API keys, no network, deterministic)\n")
    config = EngramConfig(offline=True)
    store = MemoryStore()
    pipeline = WritePipeline(store, HeuristicBackend(), config)
    build_story(pipeline)
    show_store(store)
    run_queries(store, config)
    print("\nDone. Try live decisions with: python -m engram build --backend llm ...")


if __name__ == "__main__":
    run()
