"""Conservative claim identity for updates; callers can supply metadata.fact_key.

An entity name alone never identifies a fact. Unknown predicates stay separate.
"""
from __future__ import annotations

import re

_PROPERTIES = (
    ("gym", r"\bgym\b|\bclimb(?:s|ing)?\b"),
    ("log_storage", r"\blogs?\b|\blog storage\b"),
    ("timeout", r"\btimeout\b"),
    ("residence", r"\blives?\b|\bmoved to\b"),
    ("diet", r"\bdiet\b|\bvegan\b|\bvegetarian\b"),
    ("favorite_cuisine", r"\bfavou?rite cuisine\b"),
    ("reminder_channel", r"\breminders?\b"),
    ("timezone", r"\btime\s?zone\b"),
)
_UPDATE = re.compile(r"\b(now|currently|switched|changed|updated|replaced|corrected|correction)\b|no longer|not anymore", re.I)


def fact_key(text: str, entities: list[str], metadata: dict | None = None) -> str | None:
    metadata = metadata or {}
    explicit = metadata.get("fact_key")
    if isinstance(explicit, str) and explicit.strip():
        return "explicit:" + explicit
    subject = metadata.get("subject") or (entities[0].casefold() if entities else None)
    if not subject:
        return None
    for name, pattern in _PROPERTIES:
        if re.search(pattern, text, re.I):
            return f"inferred:{str(subject).casefold()}:{name}"
    return None


def same_claim(state: dict) -> bool:
    left = state.get("observation_fact_key")
    right = state.get("candidate_fact_key")
    if left is None:
        left = fact_key(state.get("observation", ""), state.get("observation_entities", []))
    if right is None:
        right = fact_key(state.get("candidate", ""), state.get("candidate_entities", []))
    return bool(left and left == right)


def explicit_update(text: str) -> bool:
    return bool(_UPDATE.search(text))


def newer(observation, candidate) -> bool:
    # Mixed known/unknown dates cannot establish replacement order safely.
    if observation.timestamp is None or candidate.timestamp is None:
        return observation.timestamp is candidate.timestamp and observation.node_id > candidate.node_id
    return observation.timestamp > candidate.timestamp
