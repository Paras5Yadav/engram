"""Engram — a dependency-free agentic memory system with a dual-process controller."""

from .answer import AnswerSynthesizer
from .api import Engram
from .config import EngramConfig, load_config
from .decisions import (
                        BackendUnavailable,
                        HeuristicBackend,
                        LLMJudgeBackend,
                        ask_with_fallback,
                        extract_entities,
)
from .embeddings import CallableEncoder, HashedEncoder, SentenceEncoder
from .providers import LayaBackend, TypeSafeBackend
from .retrieval import RetrievalLoop
from .store import MemoryStore
from .types import Candidate, Edge, MemoryNode, Trace
from .write import WritePipeline

__version__ = "0.2.0"

__all__ = [
    "AnswerSynthesizer", "BackendUnavailable", "Candidate", "Edge",
    "EngramConfig", "HeuristicBackend", "LLMJudgeBackend", "MemoryNode",
    "MemoryStore", "RetrievalLoop", "Trace", "WritePipeline",
    "ask_with_fallback", "extract_entities", "load_config",
    "Engram", "CallableEncoder", "HashedEncoder", "SentenceEncoder", "TypeSafeBackend", "LayaBackend",
]
