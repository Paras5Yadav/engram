"""One evidence format shared by retrieval, answer synthesis, and agent clients."""
from __future__ import annotations

import json


def is_superseded(node) -> bool:
    return bool(node.flags.get("superseded_by") or node.flags.get("superseded"))


def memory_line(node) -> str:
    return json.dumps({"id": node.node_id, "timestamp": node.timestamp, "source": node.source,
                       "text": node.text, "superseded": is_superseded(node),
                       "superseded_by": node.flags.get("superseded_by")},
                      ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def render_context(nodes) -> str:
    return "\n".join(memory_line(node) for node in nodes)
