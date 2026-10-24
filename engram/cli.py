"""Local command-line and JSON interface. No server is started."""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from dataclasses import replace

from .api import Engram, parse_timestamp
from .config import load_config
from .decisions import BackendUnavailable
from .runtime import make_backend


def _load_env(path=".env"):
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _parse_timestamp(value):
    return parse_timestamp(value)


def _make_backend(name, config):
    return make_backend(config, name)


def _config(args):
    config = load_config(args.config)
    overrides = {}
    if args.backend:
        overrides["decision_backend"] = args.backend
    if args.offline:
        overrides["offline"] = True
    return replace(config, **overrides)


def _encoder(args):
    if args.embedding_model:
        from .embeddings import SentenceEncoder
        return SentenceEncoder(args.embedding_model, local_files_only=not args.allow_model_download)
    if args.allow_model_download:
        raise ValueError("--allow-model-download requires --embedding-model")
    return None


def _emit(payload, args):
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, allow_nan=False))
    elif "answer" in payload:
        print(payload["answer"])
    elif "context" in payload:
        print(payload["context"] or "No stored memories match this question yet.")
        print(f"\nStop: {payload['trace']['stop_reason']}")
    else:
        print(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2))


def main(argv=None):
    raw_arguments = list(argv) if argv is not None else sys.argv[1:]
    class JsonArgumentParser(argparse.ArgumentParser):
        def error(self, message):
            if "--json" in raw_arguments:
                print(json.dumps({"error": {"type": "ArgumentError", "message": message}}))
                raise SystemExit(2)
            super().error(message)
    parser = JsonArgumentParser(prog="engram", description="Local persistent agent memory")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("demo", help="fully offline synthetic demo")
    for name, help_text in (("build", "atomically import a JSON observation array"),
                            ("remember", "store one observation"), ("query", "retrieve evidence and trace"),
                            ("ask", "retrieve and optionally generate an answer"),
                            ("forget", "delete a node and its associated raw observations"),
                            ("export", "export one scope as JSON"), ("reindex", "explicitly rebuild vectors")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--persist", default="./engram_store")
        p.add_argument("--scope", default="default")
        p.add_argument("--config")
        p.add_argument("--backend", choices=("heuristic", "llm", "typesafe", "laya"))
        p.add_argument("--offline", action="store_true")
        p.add_argument("--json", action="store_true", help="one JSON object on stdout, including errors")
        p.add_argument("--env-file", help="explicit .env file; process environment takes precedence")
        p.add_argument("--embedding-model", help="local path or cached sentence-transformers model id")
        p.add_argument("--allow-model-download", action="store_true")
        if name == "build":
            p.add_argument("--input", required=True)
        elif name == "remember":
            p.add_argument("--text", required=True)
            p.add_argument("--timestamp")
            p.add_argument("--source", default="conversation")
            p.add_argument("--metadata", default="{}", help="JSON object; fact_key can identify an updatable claim")
            p.add_argument("--idempotency-key")
        elif name in ("query", "ask"):
            p.add_argument("--question", required=True)
            p.add_argument("--top-k", type=int)
        elif name == "forget":
            group = p.add_mutually_exclusive_group(required=True)
            group.add_argument("--id", type=int)
            group.add_argument("--all", action="store_true", help="clear this scope including filtered raw input")
        elif name == "export":
            p.add_argument("--output", required=True)
    args = parser.parse_args(raw_arguments)
    if args.command == "demo":
        from .demo import run
        run()
        return 0
    try:
        if args.env_file:
            _load_env(args.env_file)
        config = _config(args)
        with Engram(args.persist, config=config, encoder=_encoder(args), reindex=args.command == "reindex") as app:
            if args.command == "build":
                with open(args.input, encoding="utf-8") as handle:
                    payload = json.load(handle)
                if isinstance(payload, dict):
                    if set(payload) != {"observations"}:
                        raise ValueError("input object must contain only observations")
                    payload = payload["observations"]
                results = app.ingest_many(payload, scope=args.scope)
                result = {"stored": sum(r["stored"] for r in results), "total": len(results), "results": results}
            elif args.command == "remember":
                result = app.ingest(args.text, args.timestamp, source=args.source, scope=args.scope,
                                    metadata=json.loads(args.metadata), idempotency_key=args.idempotency_key)
            elif args.command in ("query", "ask"):
                method = app.query if args.command == "query" else app.ask
                result = method(args.question, top_k=args.top_k, scope=args.scope)
            elif args.command == "forget":
                result = ({"deleted": app.store.clear(scope=args.scope), "scope": args.scope} if args.all
                          else {"deleted": app.forget(args.id, scope=args.scope), "node_id": args.id})
            elif args.command == "export":
                result = {"path": app.export(args.output, scope=args.scope)}
            else:
                result = {"reindexed": True, "encoder": app.store.encoder.model_id}
            _emit(result, args)
        return 0
    except (ValueError, TypeError, OSError, sqlite3.Error, ImportError, BackendUnavailable) as error:
        payload = {"error": {"type": type(error).__name__, "message": str(error)}}
        if args.json:
            print(json.dumps(payload, ensure_ascii=False))
        else:
            print(f"engram: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
