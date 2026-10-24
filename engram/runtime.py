"""Per-operation controller budgets, bounded cache, retries, and visible fallback."""
from __future__ import annotations

import json
import os
import time
from collections import OrderedDict
from dataclasses import asdict

from .config import EngramConfig
from .decisions import (
    BackendUnavailable,
    HeuristicBackend,
    LLMJudgeBackend,
    validate_answers,
)


class DecisionSession:
    def __init__(self, backend, config: EngramConfig, cache=None):
        self.backend = HeuristicBackend() if config.offline else backend
        self.config = config
        self.cache = cache if cache is not None else OrderedDict()
        self.started = time.monotonic()
        self.deadline = self.started + config.max_latency_seconds
        self.calls = self.cache_hits = self.fallbacks = 0
        self.last_backend = self.backend.name

    @property
    def exhausted(self):
        return time.monotonic() >= self.deadline or self.calls >= self.config.maximum_calls

    def _key(self, state, questions):
        return json.dumps([state, {k: asdict(q) for k, q in questions.items()}],
                          sort_keys=True, ensure_ascii=False, allow_nan=False, default=str)

    def ask(self, state, questions):
        return self.ask_many([(state, questions)])[0]

    def ask_many(self, jobs):
        results = [None] * len(jobs)
        missing, indices, keys = [], [], []
        for i, (state, questions) in enumerate(jobs):
            key = self._key(state, questions)
            if key in self.cache:
                results[i] = dict(self.cache[key])
                self.cache.move_to_end(key)
                self.cache_hits += 1
            else:
                missing.append((state, questions))
                indices.append(i)
                keys.append(key)
        if not missing:
            return results
        values = None
        is_batch = hasattr(self.backend, "ask_many")
        if is_batch:
            for _ in range(self.config.max_retries + 1):
                if self.exhausted:
                    break
                self.calls += 1
                try:
                    values = self.backend.ask_many(missing, timeout=min(self.config.timeout_seconds, max(0.001, self.deadline - time.monotonic())))
                    if not isinstance(values, list) or len(values) != len(missing):
                        raise BackendUnavailable("judge returned an invalid batch length")
                    values = [validate_answers(v, q) for v, (_, q) in zip(values, missing)]
                    if time.monotonic() >= self.deadline:
                        raise BackendUnavailable("controller deadline exceeded")
                    break
                except BackendUnavailable:
                    values = None
        else:
            values = []
            for state, questions in missing:
                answer = None
                for _ in range(self.config.max_retries + 1):
                    if self.exhausted:
                        break
                    self.calls += 1
                    try:
                        answer = validate_answers(self.backend.ask(state, questions), questions)
                        if time.monotonic() >= self.deadline:
                            answer = None
                        break
                    except BackendUnavailable:
                        continue
                if answer is None:
                    self.fallbacks += 1
                    self.last_backend = "heuristic"
                    answer = HeuristicBackend().ask(state, questions)
                values.append(answer)
        if values is None:
            self.fallbacks += len(missing)
            self.last_backend = "heuristic"
            values = [HeuristicBackend().ask(s, q) for s, q in missing]
        for i, key, value in zip(indices, keys, values):
            results[i] = value
            # Do not cache degraded decisions after a transient provider failure.
            if self.config.cache_size and not self.fallbacks:
                self.cache[key] = dict(value)
                while len(self.cache) > self.config.cache_size:
                    self.cache.popitem(last=False)
        return results

    def stats(self):
        return {"backend": self.last_backend, "calls": self.calls, "cache_hits": self.cache_hits,
                "fallbacks": self.fallbacks, "elapsed_ms": round((time.monotonic() - self.started) * 1000, 2)}


def make_backend(config: EngramConfig, name: str | None = None):
    name = name or config.decision_backend
    if config.offline or name == "heuristic":
        return HeuristicBackend()
    if name == "llm":
        return LLMJudgeBackend(model=config.control_model or config.answer_model,
                               base_url=config.endpoint("control"), api_key=config.api_key("control"),
                               timeout=config.timeout_seconds, max_request_bytes=config.max_context_bytes * 4)
    if name == "typesafe":
        from .providers import TypeSafeBackend
        return TypeSafeBackend(model=config.control_model or "jev-latest",
                               base_url=config.control_base_url,
                               api_key=os.environ.get(config.control_key_env) or os.environ.get("TYPESAFE_API_KEY"),
                               timeout=config.timeout_seconds,
                               max_request_bytes=config.max_context_bytes * 4)
    if name == "laya":
        from .providers import LayaBackend
        if config.laya_model is None:
            raise ValueError("laya backend requires laya_model with a local checkpoint path")
        return LayaBackend(model_path=config.laya_model, device=config.laya_device,
                           max_length=config.laya_max_length)
    raise ValueError(f"unsupported decision backend: {name}")
