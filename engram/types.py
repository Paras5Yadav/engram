"""Core data structures shared across the memory pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

VIEWS = ("topic", "time", "cause", "entity")

TYPE_KEYS = ("type_event", "type_fact", "type_how", "type_like")


@dataclass
class MemoryNode:
    """One canonical observation. Immutable in spirit: we keep the original
    text, timestamp, and provenance until explicit deletion; updates add links
    or flags. Store reads return snapshots."""

    node_id: int
    text: str
    timestamp: Optional[float]
    source: str = "conversation"
    entities: list[str] = field(default_factory=list)
    type_scores: dict[str, float] = field(default_factory=lambda: dict.fromkeys(TYPE_KEYS, 0.0))
    embedding: list[float] = field(default_factory=list)
    flags: dict[str, str] = field(default_factory=dict)  # e.g. {"superseded_by": "12"}
    created_at: float = 0.0
    scope: str = "default"
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryNode":
        data = dict(data)
        data["type_scores"] = {
            key if key.startswith("type_") else "type_" + key: value
            for key, value in data.get("type_scores", {}).items()
        }
        return cls(**data)


@dataclass
class Edge:
    """A directed link in one of the four graph views."""

    view: str          # topic | time | cause | entity
    src: int
    dst: int
    kind: str          # e.g. shares_topic, before, explains, same_person
    weight: float = 1.0
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Candidate:
    node: MemoryNode
    via_view: Optional[str] = None
    via_kind: str = ""
    score: float = 0.0
    path: tuple[int, ...] = ()

    def to_dict(self) -> dict:
        data = asdict(self)
        data["node"].pop("embedding", None)
        return data


@dataclass
class Trace:
    """Every controller decision, kept inspectable."""

    routing: dict[str, float] = field(default_factory=dict)
    budgets: dict[str, int] = field(default_factory=dict)
    rounds: list[dict] = field(default_factory=list)
    stop_reason: str = ""
    evidence_ids: list[int] = field(default_factory=list)
    budget_used: dict[str, int] = field(default_factory=dict)
    scope: str = "default"
    backend: str = ""
    calls: int = 0
    cache_hits: int = 0
    fallbacks: int = 0
    elapsed_ms: float = 0.0
    edges_examined: int = 0
    nodes_examined: int = 0
    context_bytes: int = 0
    vector_nodes_scanned: int = 0
    vector_search_complete: bool = True

    def to_dict(self) -> dict:
        return asdict(self)
