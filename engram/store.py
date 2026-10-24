"""SQLite memory plane with transactional writes and indexed graph/term lookups.

Opening a directory binds this instance to a live database: save() never writes
an old snapshot over another writer. Legacy JSON is imported once and retained.
"""
from __future__ import annotations

import heapq
import json
import math
import os
import sqlite3
import tempfile
import threading
import time
import weakref
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from .embeddings import content_terms, cosine, current_encoder, normalize, tokenize
from .types import VIEWS, Edge, MemoryNode

_STORE_FILE = "memories.sqlite3"
_SCHEMA = 1


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def validate_observation(text, timestamp=None, source="conversation", entities=None,
                         scope="default", metadata=None):
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    if timestamp is not None and (type(timestamp) not in (int, float) or not math.isfinite(timestamp)):
        raise ValueError("timestamp must be finite Unix seconds or null")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("source must be a nonempty string")
    if not isinstance(scope, str) or not scope.strip() or len(scope) > 256:
        raise ValueError("scope must be a nonempty string of at most 256 characters")
    if entities is not None and (not isinstance(entities, list) or any(not isinstance(e, str) for e in entities)):
        raise ValueError("entities must be a list of strings")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("metadata must be an object")
    _json(metadata or {})


class MemoryStore:
    def __init__(self, directory: str | None = None, *, encoder=None, reindex=False):
        self.encoder = encoder or current_encoder()
        self.dimension = len(normalize(self.encoder.encode("")))
        self.directory = str(Path(directory).resolve()) if directory else None
        self._lock = threading.RLock()
        self._depth = 0
        if self.directory:
            Path(self.directory).mkdir(parents=True, exist_ok=True, mode=0o700)
        path = str(Path(self.directory) / _STORE_FILE) if self.directory else ":memory:"
        if self.directory:
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                os.close(fd)
            except FileExistsError:
                pass
        self._conn = sqlite3.connect(path, timeout=10, isolation_level=None, check_same_thread=False)
        self._finalizer = weakref.finalize(self, self._conn.close)
        self._conn.row_factory = sqlite3.Row
        try:
            self._initialize(reindex)
        except BaseException:
            self.close()
            raise

    def _initialize(self, reindex):
        existing_meta = self._conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='meta'").fetchone()
        if existing_meta:
            version = self._meta("schema")
            if version and version != str(_SCHEMA):
                self.close()
                raise ValueError(f"unsupported store schema {version}")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=10000")
        if self.directory:
            if self._conn.execute("PRAGMA journal_mode").fetchone()[0].lower() != "wal":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=FULL")
        manifest = _json({"model_id": self.encoder.model_id, "dimension": self.dimension})
        legacy = Path(self.directory) / "memories.json" if self.directory else None
        if existing_meta and self._meta("schema") == str(_SCHEMA) and not reindex:
            if self._meta("encoder") != manifest:
                raise ValueError("encoder differs from store manifest; explicitly reindex to change it")
            if not legacy or not legacy.exists() or self._meta("legacy_imported"):
                return
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS memories(
                id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL, timestamp REAL,
                source TEXT NOT NULL, entities TEXT NOT NULL, types TEXT NOT NULL,
                embedding TEXT NOT NULL, flags TEXT NOT NULL, created_at REAL NOT NULL,
                scope TEXT NOT NULL, metadata TEXT NOT NULL, terms_count INTEGER NOT NULL);
            CREATE INDEX IF NOT EXISTS memory_scope_time ON memories(scope,timestamp,id);
            CREATE INDEX IF NOT EXISTS memory_scope_text ON memories(scope,text);
            CREATE TABLE IF NOT EXISTS links(
                view TEXT NOT NULL, src INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                dst INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                kind TEXT NOT NULL, weight REAL NOT NULL, note TEXT NOT NULL,
                PRIMARY KEY(view,src,dst,kind));
            CREATE INDEX IF NOT EXISTS link_dst ON links(view,dst);
            CREATE INDEX IF NOT EXISTS link_kind ON links(view,kind,src,dst);
            CREATE TABLE IF NOT EXISTS terms(
                term TEXT NOT NULL, node_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                count INTEGER NOT NULL, PRIMARY KEY(term,node_id));
            CREATE TABLE IF NOT EXISTS observations(
                id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL, timestamp REAL,
                source TEXT NOT NULL, scope TEXT NOT NULL, metadata TEXT NOT NULL,
                idempotency_key TEXT, memory_id INTEGER REFERENCES memories(id) ON DELETE CASCADE,
                result TEXT NOT NULL, created_at REAL NOT NULL, UNIQUE(scope,idempotency_key));
            CREATE INDEX IF NOT EXISTS observation_scope ON observations(scope,id);
        """)
        try:
            with self.transaction():
                version = self._meta("schema")
                if version and version != str(_SCHEMA):
                    raise ValueError(f"unsupported store schema {version}; open with the matching Engram version")
                self._set_meta("schema", str(_SCHEMA))
                identity = self._meta("encoder")
                manifest = _json({"model_id": self.encoder.model_id, "dimension": self.dimension})
                if identity and (reindex or identity != manifest):
                    if not reindex:
                        raise ValueError("encoder differs from store manifest; explicitly reindex to change it")
                    self.reindex(self.encoder)
                else:
                    self._set_meta("encoder", manifest)
                legacy = Path(self.directory) / "memories.json" if self.directory else None
                if legacy and legacy.exists() and not self._meta("legacy_imported"):
                    if self._conn.execute("SELECT count(*) FROM memories").fetchone()[0] == 0:
                        self._import_legacy(legacy)
                    self._set_meta("legacy_imported", "1")
        except BaseException:
            self.close()
            raise

    @contextmanager
    def transaction(self, *, write=True):
        """Nested savepoints; a failed ingest rolls back nodes, links, and raw input."""
        with self._lock:
            nested = self._depth > 0
            name = f"engram_{self._depth}"
            self._conn.execute(f"SAVEPOINT {name}" if nested else ("BEGIN IMMEDIATE" if write else "BEGIN"))
            self._depth += 1
            previous_encoder, previous_dimension = self.encoder, self.dimension
            try:
                yield self
                self._conn.execute(f"RELEASE SAVEPOINT {name}" if nested else "COMMIT")
            except BaseException:
                self.encoder, self.dimension = previous_encoder, previous_dimension
                if nested:
                    self._conn.execute(f"ROLLBACK TO SAVEPOINT {name}")
                    self._conn.execute(f"RELEASE SAVEPOINT {name}")
                else:
                    self._conn.execute("ROLLBACK")
                raise
            finally:
                self._depth -= 1

    def _meta(self, key):
        row = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def _set_meta(self, key, value):
        self._conn.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", (key, value))

    @staticmethod
    def _node(row) -> MemoryNode:
        return MemoryNode(row["id"], row["text"], row["timestamp"], row["source"],
                          json.loads(row["entities"]), json.loads(row["types"]),
                          json.loads(row["embedding"]), json.loads(row["flags"]), row["created_at"],
                          row["scope"], json.loads(row["metadata"]))

    @property
    def nodes(self) -> dict[int, MemoryNode]:
        return {n.node_id: n for n in self.all()}

    @property
    def edges(self) -> dict[str, list[Edge]]:
        with self._lock:
            result = {v: [] for v in VIEWS}
            for row in self._conn.execute("SELECT view,src,dst,kind,weight,note FROM links ORDER BY view,src,dst,kind"):
                edge = Edge(**dict(row))
                result[edge.view].append(edge)
            return result

    def all(self, scope: str | None = None) -> list[MemoryNode]:
        with self._lock:
            sql, args = ("SELECT * FROM memories ORDER BY id", ()) if scope is None else (
                "SELECT * FROM memories WHERE scope=? ORDER BY id", (scope,))
            return [self._node(row) for row in self._conn.execute(sql, args)]

    def get(self, node_id: int, scope: str | None = None) -> MemoryNode | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM memories WHERE id=?", (node_id,)).fetchone()
            return self._node(row) if row and (scope is None or row["scope"] == scope) else None

    def exact_match(self, text: str, scope="default") -> MemoryNode | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM memories WHERE scope=? AND text=? ORDER BY id DESC LIMIT 1", (scope, text)).fetchone()
            return self._node(row) if row else None

    def add(self, text: str, timestamp: float | None = None, source="conversation",
            entities=None, type_scores=None, *, scope="default", metadata=None) -> MemoryNode:
        validate_observation(text, timestamp, source, entities, scope, metadata)
        if not text.strip():
            raise ValueError("memory text must not be empty")
        with self.transaction():
            vector = normalize(self.encoder.encode(text))
            if len(vector) != self.dimension:
                raise ValueError("encoder dimension changed; reindex explicitly")
            terms = Counter(tokenize(text, True))
            cursor = self._conn.execute("""INSERT INTO memories
                (text,timestamp,source,entities,types,embedding,flags,created_at,scope,metadata,terms_count)
                VALUES(?,?,?,?,?,?,?,?,?,?,?)""", (text, timestamp, source, _json(entities or []),
                _json(type_scores or {}), _json(vector), "{}", time.time(), scope, _json(metadata or {}), max(1, sum(terms.values()))))
            node_id = cursor.lastrowid
            self._conn.executemany("INSERT INTO terms VALUES(?,?,?)", [(term, node_id, count) for term, count in terms.items()])
            self._insert_time(node_id, scope, timestamp)
            return self.get(node_id)

    def _insert_time(self, node_id, scope, timestamp):
        if timestamp is None:
            return
        before = self._conn.execute("SELECT id FROM memories WHERE scope=? AND timestamp IS NOT NULL AND (timestamp<? OR (timestamp=? AND id<?)) ORDER BY timestamp DESC,id DESC LIMIT 1", (scope, timestamp, timestamp, node_id)).fetchone()
        after = self._conn.execute("SELECT id FROM memories WHERE scope=? AND timestamp IS NOT NULL AND (timestamp>? OR (timestamp=? AND id>?)) ORDER BY timestamp,id LIMIT 1", (scope, timestamp, timestamp, node_id)).fetchone()
        if before and after:
            self._conn.execute("DELETE FROM links WHERE view='time' AND src=? AND dst=? AND kind='before'", (before[0], after[0]))
        if before:
            self.link("time", before[0], node_id, "before")
        if after:
            self.link("time", node_id, after[0], "before")

    def link(self, view: str, src: int, dst: int, kind: str, weight=1.0, note="") -> Edge | None:
        if view not in VIEWS:
            raise ValueError(f"unknown view {view!r}")
        if type(weight) not in (int, float) or not math.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError("link weight must be finite in [0,1]")
        with self.transaction():
            left, right = self.get(src), self.get(dst)
            if not left or not right or src == dst:
                return None
            if left.scope != right.scope:
                raise ValueError("cannot link memories across scopes")
            self._conn.execute("""INSERT INTO links VALUES(?,?,?,?,?,?) ON CONFLICT(view,src,dst,kind)
                DO UPDATE SET weight=max(weight,excluded.weight)""", (view, src, dst, kind, weight, note))
            return Edge(view, src, dst, kind, weight, note)

    def flag(self, node_id: int, key: str, value: str) -> None:
        with self.transaction():
            node = self.get(node_id)
            if node:
                node.flags[key] = value
                self._conn.execute("UPDATE memories SET flags=? WHERE id=?", (_json(node.flags), node_id))

    def neighbors(self, node_id: int, view: str, limit=8, *, scope=None):
        if view not in VIEWS:
            raise ValueError(f"unknown view {view!r}")
        with self._lock:
            node = self.get(node_id, scope)
            if not node:
                return []
            rows = self._conn.execute("""SELECT src,dst,kind,weight FROM links WHERE view=?
                AND (src=? OR dst=?) ORDER BY weight DESC,src,dst LIMIT ?""", (view, node_id, node_id, max(0, limit)))
            return [(r["dst"], r["kind"], r["weight"]) if r["src"] == node_id
                    else (r["src"], "~" + r["kind"], r["weight"]) for r in rows]

    def vector_search(self, query: str, k=8, *, scope="default", deadline=None):
        with self._lock:
            vector = normalize(self.encoder.encode(query))
            self.last_vector_search = {"scanned": 0, "complete": True}
            def scores():
                rows = self._conn.execute("SELECT id,embedding FROM memories WHERE scope=? ORDER BY id", (scope,))
                for row in rows:
                    if deadline is not None and time.monotonic() >= deadline:
                        self.last_vector_search["complete"] = False
                        break
                    self.last_vector_search["scanned"] += 1
                    yield row["id"], cosine(vector, json.loads(row["embedding"]))
            return heapq.nsmallest(k, scores(), key=lambda item: (-item[1], item[0]))

    def keyword_search(self, query: str, k=8, *, scope="default"):
        terms = sorted(content_terms(query))
        if not terms:
            return []
        placeholders = ",".join("?" for _ in terms)
        with self._lock:
            return [tuple(r) for r in self._conn.execute(f"""SELECT m.id,sum(1.0*t.count/m.terms_count) AS score
                FROM terms t JOIN memories m ON m.id=t.node_id WHERE t.term IN ({placeholders}) AND m.scope=?
                GROUP BY m.id ORDER BY score DESC,m.id LIMIT ?""", (*terms, scope, k))]

    def previous_in_time(self, node_id):
        node = self.get(node_id)
        if not node or node.timestamp is None:
            return None
        with self._lock:
            row = self._conn.execute("SELECT * FROM memories WHERE scope=? AND timestamp<? ORDER BY timestamp DESC,id DESC LIMIT 1", (node.scope, node.timestamp)).fetchone()
            return self._node(row) if row else None

    def has_conflicts(self, node_ids) -> bool:
        """Indexed consistency check over the selected, nonhistorical evidence."""
        ids = sorted(set(node_ids))
        if len(ids) < 2:
            return False
        placeholders = ",".join("?" for _ in ids)
        with self._lock:
            row = self._conn.execute(f"SELECT 1 FROM links WHERE view='topic' AND kind='conflicts_with' AND src IN ({placeholders}) AND dst IN ({placeholders}) LIMIT 1", (*ids, *ids)).fetchone()
            return row is not None

    def recent(self, count=5, *, scope="default"):
        with self._lock:
            rows = self._conn.execute("SELECT * FROM memories WHERE scope=? ORDER BY id DESC LIMIT ?", (scope, count)).fetchall()
            return [self._node(row) for row in reversed(rows)]

    def cluster_for(self, node_id, view="topic", *, limit=128):
        cluster, frontier = {node_id}, [node_id]
        while frontier and len(cluster) < limit:
            for other, _, _ in self.neighbors(frontier.pop(), view, limit=limit - len(cluster)):
                if other not in cluster:
                    cluster.add(other)
                    frontier.append(other)
        return cluster

    def record_observation(self, text, timestamp, source, scope, metadata, idempotency_key=None):
        cursor = self._conn.execute("INSERT INTO observations(text,timestamp,source,scope,metadata,idempotency_key,result,created_at) VALUES(?,?,?,?,?,?,?,?)", (text, timestamp, source, scope, _json(metadata or {}), idempotency_key, "{}", time.time()))
        return cursor.lastrowid

    def finish_observation(self, observation_id, result):
        self._conn.execute("UPDATE observations SET memory_id=?,result=? WHERE id=?", (result.get("node_id"), _json(result), observation_id))

    def replay(self, scope, key):
        with self._lock:
            row = self._conn.execute("SELECT text,timestamp,source,metadata,result FROM observations WHERE scope=? AND idempotency_key=?", (scope, key)).fetchone()
            return dict(row) if row else None

    def observations(self, scope="default"):
        with self._lock:
            rows = self._conn.execute("SELECT * FROM observations WHERE scope=? ORDER BY id", (scope,)).fetchall()
            return [{**dict(r), "metadata": json.loads(r["metadata"]), "result": json.loads(r["result"])} for r in rows]

    def forget(self, node_id: int, *, scope="default") -> bool:
        with self.transaction():
            node = self.get(node_id, scope)
            if not node:
                return False
            self._conn.execute("DELETE FROM memories WHERE id=? AND scope=?", (node_id, scope))
            for other in self.all(scope):
                if other.flags.get("superseded_by") == str(node_id):
                    other.flags.pop("superseded_by")
                    other.flags["superseded"] = "true"
                    self._conn.execute("UPDATE memories SET flags=? WHERE id=?", (_json(other.flags), other.node_id))
            self._rebuild_time(scope)
            return True

    def clear(self, *, scope="default") -> int:
        """Delete a scope's curated nodes and all raw/filtered observations."""
        with self.transaction():
            count = self._conn.execute("SELECT count(*) FROM memories WHERE scope=?", (scope,)).fetchone()[0]
            self._conn.execute("DELETE FROM observations WHERE scope=?", (scope,))
            self._conn.execute("DELETE FROM memories WHERE scope=?", (scope,))
            return count

    def _rebuild_time(self, scope):
        self._conn.execute("DELETE FROM links WHERE view='time' AND src IN (SELECT id FROM memories WHERE scope=?)", (scope,))
        rows = self._conn.execute("SELECT id FROM memories WHERE scope=? AND timestamp IS NOT NULL ORDER BY timestamp,id", (scope,)).fetchall()
        for left, right in zip(rows, rows[1:]):
            self.link("time", left[0], right[0], "before")

    def export(self, *, scope="default") -> dict:
        with self.transaction(write=False):
            nodes = self.all(scope)
            ids = {n.node_id for n in nodes}
            return {"schema_version": _SCHEMA, "encoder": json.loads(self._meta("encoder")),
                    "nodes": [n.to_dict() for n in nodes],
                    "edges": {v: [e.to_dict() for e in edges if e.src in ids and e.dst in ids] for v, edges in self.edges.items()},
                    "observations": self.observations(scope)}

    def export_file(self, path: str, *, scope="default") -> str:
        payload = self.export(scope=scope)
        target = Path(path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=target.parent, prefix=".engram-export-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, allow_nan=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return str(target)

    def reindex(self, encoder):
        dimension = len(normalize(encoder.encode("")))
        with self.transaction():
            for node in self.all():
                vector = normalize(encoder.encode(node.text))
                if len(vector) != dimension:
                    raise ValueError("encoder returned inconsistent dimensions")
                self._conn.execute("UPDATE memories SET embedding=? WHERE id=?", (_json(vector), node.node_id))
            self._set_meta("encoder", _json({"model_id": encoder.model_id, "dimension": dimension}))
        self.encoder, self.dimension = encoder, dimension

    def _import_legacy(self, path):
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        for data in payload["nodes"]:
            node = MemoryNode.from_dict(data)
            validate_observation(node.text, node.timestamp, node.source, node.entities, node.scope, node.metadata)
            vector = normalize(self.encoder.encode(node.text))
            if len(vector) != self.dimension:
                raise ValueError("legacy import encoder dimension mismatch")
            terms = Counter(tokenize(node.text, True))
            self._conn.execute("INSERT INTO memories VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (node.node_id, node.text, node.timestamp, node.source, _json(node.entities), _json(node.type_scores), _json(vector), _json(node.flags), node.created_at, node.scope, _json(node.metadata), max(1, sum(terms.values()))))
            self._conn.executemany("INSERT INTO terms VALUES(?,?,?)", [(term, node.node_id, count) for term, count in terms.items()])
        for view, edges in payload.get("edges", {}).items():
            if view != "time":
                for data in edges:
                    self.link(**data)
        for scope in {n.scope for n in self.all()}:
            self._rebuild_time(scope)

    def save(self, directory: str) -> str:
        target = str(Path(directory).resolve())
        if self.directory:
            if target != self.directory:
                raise ValueError("store already bound; use export_file for a separate snapshot")
            return str(Path(target) / _STORE_FILE)
        destination = MemoryStore.load(target, encoder=self.encoder)
        try:
            with self.transaction(write=False), destination.transaction():
                if destination._conn.execute("SELECT count(*) FROM memories").fetchone()[0] or destination._conn.execute("SELECT count(*) FROM observations").fetchone()[0]:
                    raise ValueError("refusing to replace a nonempty store; open it with MemoryStore.load")
                for table in ("memories", "terms", "links", "observations"):
                    rows = self._conn.execute(f"SELECT * FROM {table}").fetchall()
                    if rows:
                        placeholders = ",".join("?" for _ in rows[0])
                        destination._conn.executemany(f"INSERT INTO {table} VALUES({placeholders})", [tuple(r) for r in rows])
            self.close()
            self._conn, destination._conn = destination._conn, None
            destination._finalizer.detach()
            self._finalizer = weakref.finalize(self, self._conn.close)
            self.directory = target
        finally:
            destination.close()
        return str(Path(target) / _STORE_FILE)

    @classmethod
    def load(cls, directory: str, *, encoder=None, reindex=False):
        return cls(directory, encoder=encoder, reindex=reindex)

    def close(self):
        with self._lock:
            if self._conn is not None:
                self._finalizer()
                self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
