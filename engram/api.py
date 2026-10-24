"""Stable local application interface used by the CLI and agent skills."""
from __future__ import annotations

import math
from datetime import datetime, timezone

from .answer import AnswerSynthesizer
from .config import EngramConfig
from .context import render_context
from .retrieval import RetrievalLoop
from .runtime import make_backend
from .store import MemoryStore, validate_observation
from .write import WritePipeline


def parse_timestamp(value):
    if value is None:
        return None
    if type(value) in (int, float):
        if not math.isfinite(value):
            raise ValueError("timestamp must be finite")
        return float(value)
    if not isinstance(value, str):
        raise ValueError("timestamp must be Unix seconds, an ISO date, or null")
    try:
        numeric = float(value)
    except ValueError:
        pass
    else:
        if not math.isfinite(numeric):
            raise ValueError("timestamp must be finite")
        return numeric
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("timestamp must be Unix seconds, an ISO date, or null") from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


class Engram:
    def __init__(self, directory: str | None = None, *, config=None, backend=None, encoder=None, reindex=False):
        self.config = config or EngramConfig()
        self.backend = backend if backend is not None else make_backend(self.config)
        self.store = MemoryStore(directory, encoder=encoder, reindex=reindex)
        self.writer = WritePipeline(self.store, self.backend, self.config)
        self.retriever = RetrievalLoop(self.store, self.backend, self.config)
        self.synthesizer = AnswerSynthesizer(self.config)

    def ingest(self, text, timestamp=None, **kwargs):
        return self.writer.ingest(text, parse_timestamp(timestamp), **kwargs)

    def ingest_many(self, items, *, scope="default"):
        """Validate the whole batch first, then commit all or none."""
        if not isinstance(items, list):
            raise ValueError("observations must be an array of strings or objects")
        prepared = []
        allowed = {"text", "timestamp", "source", "entities", "metadata", "idempotency_key"}
        for item in items:
            if isinstance(item, str):
                item = {"text": item, "source": "import"}
            if not isinstance(item, dict) or "text" not in item or set(item) - allowed:
                raise ValueError("observation must have text and only supported fields")
            item = dict(item)
            item["timestamp"] = parse_timestamp(item.get("timestamp"))
            validate_observation(item["text"], item["timestamp"], item.get("source", "conversation"),
                                 item.get("entities"), scope, item.get("metadata"))
            if len(item["text"]) > self.config.max_observation_chars:
                raise ValueError("observation exceeds max_observation_chars")
            prepared.append(item)
        with self.store.transaction():
            return [self.writer.ingest(**item, scope=scope) for item in prepared]

    def query(self, question, *, top_k=None, scope="default"):
        evidence, trace = self.retriever.query(question, top_k, scope=scope)
        return {"evidence": [c.to_dict() for c in evidence], "trace": trace.to_dict(),
                "context": render_context([c.node for c in evidence])}

    def ask(self, question, *, top_k=None, scope="default"):
        evidence, trace = self.retriever.query(question, top_k, scope=scope)
        return {**self.synthesizer.answer(question, [c.node for c in evidence]), "trace": trace.to_dict()}

    def forget(self, node_id, *, scope="default"):
        return self.store.forget(node_id, scope=scope)

    def export(self, path, *, scope="default"):
        return self.store.export_file(path, scope=scope)

    def close(self):
        self.store.close()
        close = getattr(self.backend, "close", None)
        if close:
            close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
