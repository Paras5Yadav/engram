"""Scoped hybrid retrieval with relevance gates and bounded graph traversal."""
from __future__ import annotations

import math
import time
from collections import OrderedDict
from dataclasses import replace

from .config import EngramConfig
from .context import is_superseded, render_context
from .embeddings import content_terms
from .questions import routing_questions, stopping_questions, traversal_questions
from .runtime import DecisionSession, make_backend
from .store import MemoryStore
from .types import VIEWS, Candidate, Trace
from .write import fuse_rankings


def allocate_budgets(needs: dict[str, float], config: EngramConfig) -> dict[str, int]:
    budgets = dict.fromkeys(VIEWS, 0)
    active = [v for v in VIEWS if needs.get(v, 0.0) >= config.activation_threshold]
    active.sort(key=lambda v: (-needs.get(v, 0), VIEWS.index(v)))
    floor = config.minimum_graph_budget
    active = active[:config.total_graph_budget // floor]
    if not active:
        return budgets
    remaining = config.total_graph_budget - floor * len(active)
    weight = sum(needs.get(v, 0) for v in active)
    shares = {v: remaining * (needs.get(v, 0) / weight if weight else 1 / len(active)) for v in active}
    for view in active:
        budgets[view] = floor + math.floor(shares[view])
    remainder = config.total_graph_budget - sum(budgets.values())
    order = sorted(active, key=lambda v: (-(shares[v] - math.floor(shares[v])), VIEWS.index(v)))
    for view in order[:remainder]:
        budgets[view] += 1
    return budgets


class RetrievalLoop:
    def __init__(self, store: MemoryStore, backend=None, config: EngramConfig | None = None):
        self.store, self.config = store, config or EngramConfig()
        self.backend = backend if backend is not None else make_backend(self.config)
        self._cache = OrderedDict()

    def query(self, question: str, top_k: int | None = None, *, scope="default"):
        cfg = self.config
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a nonempty string")
        if len(question) > cfg.max_observation_chars:
            raise ValueError("question exceeds max_observation_chars")
        if not isinstance(scope, str) or not scope.strip():
            raise ValueError("scope must be a nonempty string")
        count = cfg.top_k if top_k is None else top_k
        if type(count) is not int or count <= 0:
            raise ValueError("top_k must be a positive integer")
        session = DecisionSession(self.backend, cfg, self._cache)
        trace = Trace(scope=scope)
        with self.store.transaction(write=False):
            route = session.ask({"query": question}, routing_questions())
            trace.routing = dict(route)
            budgets = allocate_budgets({v: route[f"need_{v}"] for v in VIEWS}, cfg)
            trace.budgets = dict(budgets)
            evidence = self._anchors(question, scope, session, trace)
            selected = self._select(evidence, count, route)
            reason = "anchors_only"
            frontier = list(evidence)
            rounds = cfg.max_rounds if route["need_multi_hop"] >= cfg.activation_threshold else min(cfg.max_rounds, 1)
            if not selected:
                if time.monotonic() >= session.deadline:
                    reason = "deadline"
                elif session.exhausted:
                    reason = "controller_budget"
                else:
                    reason = "context_budget" if evidence else "no_match"
            else:
                for round_index in range(rounds):
                    if session.exhausted:
                        reason = "deadline" if time.monotonic() >= session.deadline else "controller_budget"
                        break
                    if not any(budgets.values()):
                        reason = "budget_exhausted"
                        break
                    added = self._expand(question, scope, evidence, frontier, budgets, session, trace)
                    frontier = sorted(added)
                    selected = self._select(evidence, count, route)
                    known_conflict = self._conflicts(selected)
                    stop = session.ask({"query": question, "evidence": [c.node.text for c in selected],
                                         "evidence_ids": [c.node.node_id for c in selected],
                                         "budget_left": sum(budgets.values()),
                                         "unresolved_conflicts": known_conflict}, stopping_questions())
                    trace.rounds.append({"round": round_index + 1, "added": sorted(added),
                                         "budget_left": sum(budgets.values()), **stop})
                    if (not known_conflict and stop["enough"] >= cfg.coverage_threshold
                            and stop["missing_piece"] < cfg.conflict_threshold
                            and stop["conflicts_present"] < cfg.conflict_threshold):
                        reason = "enough_evidence"
                        break
                    if session.exhausted:
                        reason = "deadline" if time.monotonic() >= session.deadline else "controller_budget"
                        break
                    if trace.edges_examined >= cfg.maximum_edges:
                        reason = "edge_limit"
                        break
                    if trace.nodes_examined >= cfg.maximum_nodes:
                        reason = "node_limit"
                        break
                    if stop["worth_continuing"] < 0.4 or not frontier:
                        reason = "low_expected_value"
                        break
                else:
                    reason = "round_limit" if rounds else "anchors_only"
            trace.stop_reason = reason
            trace.evidence_ids = [c.node.node_id for c in selected]
            trace.budget_used = {v: trace.budgets[v] - budgets[v] for v in VIEWS}
            trace.context_bytes = len(render_context([c.node for c in selected]).encode("utf-8"))
            for name, value in session.stats().items():
                setattr(trace, name, value)
            return selected, trace

    def _supported(self, query, node, semantic_score=None):
        if content_terms(query) & content_terms(node.text):
            return True
        return bool(self.store.encoder.semantic and semantic_score is not None
                    and semantic_score >= self.config.relevance_threshold)

    @staticmethod
    def _score(answers):
        return 0.7 * answers["relevant"] + 0.2 * answers["adds_detail"] + 0.1 * answers["corroborates"]

    def _anchors(self, question, scope, session, trace):
        count = min(self.config.anchor_count, self.config.maximum_nodes)
        vectors = self.store.vector_search(question, count, scope=scope, deadline=session.deadline)
        trace.vector_nodes_scanned = self.store.last_vector_search["scanned"]
        trace.vector_search_complete = self.store.last_vector_search["complete"]
        keywords = self.store.keyword_search(question, count, scope=scope)
        fused = fuse_rankings([[nid for nid, _ in vectors], [nid for nid, _ in keywords]])
        similarities = dict(vectors)
        nodes = []
        for nid in sorted(fused, key=lambda nid: (-fused[nid], nid)):
            node = self.store.get(nid, scope)
            if self._supported(question, node, similarities.get(nid)):
                nodes.append(node)
                if len(nodes) == count:
                    break
        trace.nodes_examined = len(nodes)
        jobs = [({"query": question, "candidate": n.text, "evidence": [], "source": n.source}, traversal_questions()) for n in nodes]
        decisions = session.ask_many(jobs)
        return {n.node_id: Candidate(n, score=self._score(a), path=(n.node_id,)) for n, a in zip(nodes, decisions)
                if a["relevant"] >= self.config.relevance_threshold}

    def _select(self, evidence, count, route):
        latest = route["need_latest"] >= self.config.activation_threshold
        ranked = [replace(c, score=c.score * (0.4 if latest and is_superseded(c.node) else 1)) for c in evidence.values()]
        ranked.sort(key=lambda c: (-c.score, -(c.node.timestamp if c.node.timestamp is not None else c.node.created_at), c.node.node_id))
        selected = []
        for candidate in ranked:
            if len(render_context([c.node for c in [*selected, candidate]]).encode("utf-8")) <= self.config.max_context_bytes:
                selected.append(candidate)
            if len(selected) >= count:
                break
        return selected

    def _expand(self, question, scope, evidence, frontier, budgets, session, trace):
        discovered = {}
        for view in VIEWS:
            for node_id in frontier:
                limit = min(8, budgets[view], self.config.maximum_edges - trace.edges_examined)
                if limit <= 0 or trace.nodes_examined + len(discovered) >= self.config.maximum_nodes or session.exhausted:
                    break
                for other, kind, weight in self.store.neighbors(node_id, view, limit, scope=scope):
                    trace.edges_examined += 1
                    budgets[view] -= 1
                    if other not in evidence and other not in discovered and trace.nodes_examined + len(discovered) < self.config.maximum_nodes:
                        node = self.store.get(other, scope)
                        discovered[other] = (node, view, kind, weight, evidence[node_id].path + (other,))
        jobs, candidates = [], []
        texts = [c.node.text for c in evidence.values()]
        for candidate in discovered.values():
            node, view, kind, weight, path = candidate
            trace.nodes_examined += 1
            # For local lexical retrieval a shared term is required. Semantic
            # encoders let the typed judge assess neighbors without word overlap.
            if self.store.encoder.semantic or self._supported(question, node):
                candidates.append(candidate)
                jobs.append(({"query": question, "candidate": node.text, "evidence": texts,
                              "view": view, "relation": kind, "relation_strength": weight,
                              "via_text": evidence[path[-2]].node.text,
                              "path": path}, traversal_questions()))
        decisions = session.ask_many(jobs)
        added = set()
        for (node, view, kind, weight, path), answers in zip(candidates, decisions):
            score = self._score(answers)
            if answers["relevant"] >= self.config.relevance_threshold and score >= self.config.expand_threshold:
                evidence[node.node_id] = Candidate(node, view, kind, score, path)
                added.add(node.node_id)
        return added

    def _conflicts(self, selected):
        return self.store.has_conflicts(c.node.node_id for c in selected if not is_superseded(c.node))
