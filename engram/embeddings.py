"""Versioned encoders. The default lexical encoder works without downloads."""
from __future__ import annotations

import hashlib
import math
import unicodedata
from typing import Callable

DIM = 256
STOPWORDS = frozenset(
    "a an the and or but if then else of to in on at for with by from as is are was "
    "were be been being it its this that these those i you he she we they me him her "
    "them my your his their our so do does did have has had will would can "
    "could just about what which who whom when where why how".split())


def _canon(token: str) -> str:
    token = token.replace("’", "'")
    if not token.isascii():
        return token
    if token.endswith("'s") and len(token) > 3:
        token = token[:-2]
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[:-len(suffix)]
    return token


def tokenize(text: str, drop_stopwords: bool = False) -> list[str]:
    tokens, current = [], []
    for char in unicodedata.normalize("NFC", text).lower():
        if unicodedata.category(char)[0] in ("L", "M", "N") or (char in "'’" and current):
            current.append(char)
        elif current:
            tokens.append("".join(current).strip("'’"))
            current = []
    if current:
        tokens.append("".join(current).strip("'’"))
    return [_canon(t) for t in tokens if not drop_stopwords or t not in STOPWORDS]


def content_terms(text: str) -> set[str]:
    return set(tokenize(text, True))


def normalize(vector) -> list[float]:
    try:
        values = [float(v) for v in vector]
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError("encoder must return a numeric vector") from error
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError("encoder vector must be nonempty and finite")
    norm = math.hypot(*values)
    if not math.isfinite(norm):
        raise ValueError("encoder vector norm overflow")
    return [v / norm for v in values] if norm else values


class HashedEncoder:
    semantic = False

    def __init__(self, dim: int = DIM):
        if type(dim) is not int or dim <= 0:
            raise ValueError("encoder dimension must be positive")
        self.dim = dim
        self.model_id = f"engram-lexical-v2:{dim}"

    def encode(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        tokens = tokenize(text)
        for feature in tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]:
            digest = hashlib.md5(feature.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            vector[index] += 1 if digest[4] % 2 == 0 else -1
        return normalize(vector)


class CallableEncoder:
    """The model_id must change whenever weights or preprocessing change."""
    semantic = True

    def __init__(self, function: Callable, model_id: str):
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("a stable encoder model_id is required")
        self.function, self.model_id = function, model_id

    def encode(self, text: str) -> list[float]:
        return normalize(self.function(text))


class SentenceEncoder:
    """Optional local sentence-transformers encoder; loading is explicit."""
    semantic = True

    def __init__(self, model: str, *, device: str = "cpu", local_files_only: bool = True):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model, device=device, local_files_only=local_files_only)
        self.model_id = f"sentence-transformers:{model}"

    def encode(self, text: str) -> list[float]:
        return normalize(self.model.encode(text, normalize_embeddings=True))


_encoder = HashedEncoder()


def current_encoder():
    return _encoder


def set_encoder(encoder, model_id: str | None = None) -> None:
    """Select the default for NEW stores. Existing stores keep their encoder."""
    global _encoder
    if hasattr(encoder, "encode") and hasattr(encoder, "model_id"):
        _encoder = encoder
    else:
        if not model_id:
            raise ValueError("custom encoders require an explicit model_id")
        _encoder = CallableEncoder(encoder, model_id)


def encode(text: str) -> list[float]:
    return normalize(_encoder.encode(text))


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vector length mismatch; reindex with the configured encoder")
    left, right = normalize(a), normalize(b)
    return max(-1.0, min(1.0, sum(x * y for x, y in zip(left, right))))


def overlap(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a or b else 0.0
