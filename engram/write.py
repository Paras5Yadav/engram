"""Atomic admission, typing, pairwise linking, and conservative fact updates."""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict

from .claims import fact_key, newer, same_claim
from .config import EngramConfig
from .decisions import extract_entities
from .questions import (
    admission_questions,
    consolidation_questions,
    relation_questions,
    typing_questions,
)
from .runtime import DecisionSession, make_backend
from .store import MemoryStore, validate_observation
from .types import TYPE_KEYS


def fuse_rankings(rankings: list[list[int]]) -> dict[int, float]:
    scores = {}
    for ranking in rankings:
        for rank, node_id in enumerate(ranking):
            scores[node_id] = scores.get(node_id, 0.0) + 1.0 / (61 + rank)
    return scores


class WritePipeline:
    def __init__(self, store: MemoryStore, backend=None, config: EngramConfig | None = None):
        self.store, self.config = store, config or EngramConfig()
        self.backend = backend if backend is not None else make_backend(self.config)
        self._cache = OrderedDict()

    def ingest(self, text: str, timestamp: float | None = None, source="conversation",
               entities=None, *, scope="default", metadata=None, idempotency_key=None) -> dict:
        validate_observation(text, timestamp, source, entities, scope, metadata)
        if len(text) > self.config.max_observation_chars:
            raise ValueError("observation exceeds max_observation_chars")
        if idempotency_key is not None and (not isinstance(idempotency_key, str) or not idempotency_key.strip() or len(idempotency_key) > 256):
            raise ValueError("idempotency_key must be a nonempty string of at most 256 characters")
        metadata = dict(metadata or {})
        fingerprint = hashlib.sha256(json.dumps([text, timestamp, source, entities, scope, metadata],
            sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()
        session = DecisionSession(self.backend, self.config, self._cache)
        # This deliberately serializes local writers while deciding links, so a
        # concurrent update cannot race the candidate set or supersession flags.
        with self.store.transaction():
            if idempotency_key is not None:
                previous = self.store.replay(scope, idempotency_key)
                if previous:
                    result = json.loads(previous["result"])
                    if result.get("request_hash") != fingerprint:
                        raise ValueError("idempotency_key was already used for different input")
                    return dict(result, replayed=True)
            observation_id = self.store.record_observation(text, timestamp, source, scope, metadata, idempotency_key)
            result = self._ingest(text.strip(), timestamp, source, entities, scope, metadata, session)
            result.update(observation_id=observation_id, request_hash=fingerprint, runtime=session.stats())
            self.store.finish_observation(observation_id, result)
            return result

    def _ingest(self, text, timestamp, source, entities, scope, metadata, session):
        if not text:
            return {"stored": False, "reason": "empty"}
        duplicate = self.store.exact_match(text, scope)
        if duplicate:
            return {"stored": False, "reason": "duplicate", "node_id": duplicate.node_id, "decider": "code"}
        entities = entities if entities is not None else extract_entities(text)
        state = {"observation": text, "observation_entities": entities, "observation_ts": timestamp,
                 "observation_fact_key": fact_key(text, entities, metadata),
                 "recent": [n.text for n in self.store.recent(5, scope=scope)]}
        answers = session.ask(state, {**admission_questions(), **typing_questions()})
        if self.config.admission_enabled and answers["keep"] < self.config.keep_threshold:
            return {"stored": False, "reason": "filtered", "keep": answers["keep"], "decider": session.last_backend}
        # Fuzzy duplicate scores are advisory: a one-character/number correction
        # must never disappear. The raw observation is retained for every input.
        scores = {key: round(answers[key], 3) for key in TYPE_KEYS}
        node = self.store.add(text, timestamp, source, entities, scores, scope=scope, metadata=metadata)
        candidates = self._candidates(node, session)
        jobs = []
        states = []
        for candidate in candidates:
            cstate = dict(state, candidate=candidate.text, candidate_ts=candidate.timestamp,
                          candidate_entities=candidate.entities,
                          candidate_fact_key=fact_key(candidate.text, candidate.entities, candidate.metadata))
            states.append(cstate)
            jobs.append((cstate, {**relation_questions(), **consolidation_questions()}))
        # All pairs and all relation types are judged together by batch-capable providers.
        decisions = session.ask_many(jobs)
        links = []
        for candidate, cstate, answers in zip(candidates, states, decisions):
            for view, field, src, dst, kind in (
                ("topic", "topic_link", node.node_id, candidate.node_id, "shares_topic"),
                ("cause", "explains", candidate.node_id, node.node_id, "explains"),
                ("cause", "explained_by", node.node_id, candidate.node_id, "explains"),
                ("entity", "same_entity", node.node_id, candidate.node_id, "same_entity"),
            ):
                if answers[field] >= self.config.link_threshold:
                    self.store.link(view, src, dst, kind, answers[field])
                    links.append({"view": view, "to": candidate.node_id, "kind": kind, "p": answers[field]})
            if answers["representation"] in ("combine", "generalize"):
                kind = "same_fact" if answers["representation"] == "combine" else "pattern"
                self.store.link("topic", node.node_id, candidate.node_id, kind, answers["repeats"])
                links.append({"view": "topic", "to": candidate.node_id, "kind": kind})
            if answers["conflicts"] >= self.config.link_threshold and same_claim(cstate):
                self.store.link("topic", node.node_id, candidate.node_id, "conflicts_with", answers["conflicts"])
                links.append({"view": "topic", "to": candidate.node_id, "kind": "conflicts_with"})
            if (answers["supersedes"] >= self.config.supersede_threshold and same_claim(cstate)
                    and newer(node, candidate) and not candidate.flags.get("superseded_by")):
                self.store.flag(candidate.node_id, "superseded_by", str(node.node_id))
                self.store.link("topic", node.node_id, candidate.node_id, "supersedes", answers["supersedes"])
                links.append({"flag": "superseded_by", "node": candidate.node_id, "by": node.node_id})
        for other, kind, _ in self.store.neighbors(node.node_id, "time", scope=scope):
            links.append({"view": "time", "to": other, "kind": kind, "by": "code"})
        return {"stored": True, "node_id": node.node_id, "types": scores, "links": links, "decider": session.last_backend}

    def _candidates(self, node, session):
        k = self.config.candidate_k
        vectors = [nid for nid, _ in self.store.vector_search(node.text, k + 1, scope=node.scope, deadline=session.deadline) if nid != node.node_id]
        keywords = [nid for nid, _ in self.store.keyword_search(node.text, k + 1, scope=node.scope) if nid != node.node_id]
        fused = fuse_rankings([vectors, keywords])
        return [self.store.get(nid, node.scope) for nid in sorted(fused, key=lambda nid: (-fused[nid], nid))[:k]]
