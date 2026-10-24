"""Validated settings. Keys stay in the environment, never in a memory store."""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, fields
from urllib.parse import urlparse


@dataclass(frozen=True)
class EngramConfig:
    keep_threshold: float = 0.5
    duplicate_threshold: float = 0.85
    admission_enabled: bool = True
    link_threshold: float = 0.55
    supersede_threshold: float = 0.75
    candidate_k: int = 8
    activation_threshold: float = 0.35
    total_graph_budget: int = 24
    minimum_graph_budget: int = 2
    max_rounds: int = 3
    anchor_count: int = 8
    top_k: int = 6
    expand_threshold: float = 0.45
    relevance_threshold: float = 0.12
    coverage_threshold: float = 0.75
    conflict_threshold: float = 0.4
    maximum_nodes: int = 64
    maximum_edges: int = 256
    maximum_calls: int = 24
    max_latency_seconds: float = 15.0
    timeout_seconds: float = 5.0
    max_retries: int = 1
    cache_size: int = 256
    max_context_bytes: int = 24000
    max_observation_chars: int = 64000
    decision_backend: str = "heuristic"
    control_model: str | None = None
    answer_model: str = "gpt-4o-mini"
    llm_base_url: str | None = None
    control_base_url: str | None = None
    answer_base_url: str | None = None
    control_key_env: str = "ENGRAM_CONTROL_API_KEY"
    answer_key_env: str = "ENGRAM_ANSWER_API_KEY"
    laya_model: str | None = None
    laya_device: str = "cpu"
    laya_max_length: int = 512
    offline: bool = False

    def __post_init__(self):
        for f in fields(self):
            name, value = f.name, getattr(self, f.name)
            if name.endswith("threshold"):
                if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"{name} must be a finite number in [0, 1]")
        for name in ("candidate_k", "minimum_graph_budget", "anchor_count", "top_k",
                     "maximum_nodes", "maximum_edges", "maximum_calls", "max_context_bytes",
                     "max_observation_chars", "laya_max_length"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("total_graph_budget", "max_rounds", "max_retries", "cache_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("max_latency_seconds", "timeout_seconds"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("offline", "admission_enabled"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        for name in ("llm_base_url", "control_base_url", "answer_base_url"):
            if getattr(self, name) is not None:
                validate_endpoint(getattr(self, name))
        if self.decision_backend not in ("heuristic", "llm", "typesafe", "laya"):
            raise ValueError("decision_backend must be heuristic, llm, typesafe, or laya")
        for name in ("answer_model", "control_key_env", "answer_key_env"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty string")
        if self.control_model is not None and (not isinstance(self.control_model, str) or not self.control_model.strip()):
            raise ValueError("control_model must be a nonempty string or null")
        if self.laya_model is not None and (not isinstance(self.laya_model, str) or not self.laya_model.strip()):
            raise ValueError("laya_model must be a local checkpoint path or null")
        if not isinstance(self.laya_device, str) or not self.laya_device.strip():
            raise ValueError("laya_device must be a nonempty string")

    def endpoint(self, role: str) -> str:
        if role not in ("control", "answer"):
            raise ValueError("unknown model role")
        endpoint = (getattr(self, f"{role}_base_url") or self.llm_base_url
                    or os.environ.get(f"ENGRAM_{role.upper()}_BASE_URL")
                    or os.environ.get("ENGRAM_LLM_BASE_URL") or "https://api.openai.com/v1")
        validate_endpoint(endpoint)
        return endpoint.rstrip("/")

    def api_key(self, role: str) -> str | None:
        return (os.environ.get(getattr(self, f"{role}_key_env"))
                or os.environ.get("ENGRAM_API_KEY") or os.environ.get("OPENAI_API_KEY"))

    def to_dict(self) -> dict:
        return asdict(self)


def validate_endpoint(value: str) -> None:
    if not isinstance(value, str):
        raise ValueError("endpoint must be a URL string")
    url = urlparse(value)
    if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError("endpoint must be an HTTP(S) URL without credentials, query, or fragment")


def load_config(path: str | None = None) -> EngramConfig:
    if not path:
        return EngramConfig()
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("configuration must be a JSON object")
    unknown = set(data) - {f.name for f in fields(EngramConfig)}
    if unknown:
        raise ValueError(f"unknown configuration fields: {', '.join(sorted(unknown))}")
    return EngramConfig(**data)
