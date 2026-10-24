# Engram

Persistent agent memory that runs on the user's computer. The base engine uses
Python 3.11+ and the standard library. Optional adapters add TypeSafe decisions,
OpenAI-compatible models, local Laya decisions, or local semantic embeddings.
No Engram server or hosted database is required.

Version 0.2 adds a transactional SQLite engine and a distributable local skill.
This is an engineering foundation. Quality and performance still require
evaluation with the selected models, data, and budgets.

## Run locally

```bash
python -m engram demo
python -m engram remember --persist ./engram_store --scope project-a \
  --text "The project API timeout is 30 seconds." --timestamp 2026-10-24 --json
python -m engram query --persist ./engram_store --scope project-a \
  --question "What is the API timeout?" --json
```

`query` returns evidence, a bounded `context` block, and a decision trace. It
never generates an answer or starts a server. The agent using the skill can
answer from that evidence with its existing model. `ask` optionally calls a
separate answer model; `--offline` forces heuristic decisions and template
answers even when provider keys exist in the environment.

To install the CLI in a virtual environment:

```bash
python -m venv .venv
# Activate .venv using your platform's activation command.
python -m pip install .
engram demo
```

## Local skill distribution

Build the wheel and bundle it with the skill:

```bash
python -m pip install build
python -m build --wheel
python scripts/package_skill.py --wheel dist/engram-0.2.0-py3-none-any.whl
python scripts/check_skill.py --bundle dist/engram-memory.zip
```

The resulting `dist/engram-memory.zip` contains the skill, bootstrap script,
and Engram wheel. Extract its `engram-memory` directory into the agent's
skills directory. The source skill is [skills/engram-memory/SKILL.md](skills/engram-memory/SKILL.md).

Run the installed skill's `scripts/run.py` with the same CLI arguments:

```bash
python /path/to/engram-memory/scripts/run.py remember \
  --persist /path/to/project/.engram-memory --text "Mira prefers tea." --json
python /path/to/engram-memory/scripts/run.py query \
  --persist /path/to/project/.engram-memory --question "What does Mira prefer?" --json
```

First use creates a private `.runtime` virtual environment and installs the
included wheel without contacting a package index. Python itself must already
be installed with `venv`/`ensurepip` support. Optional `setup --provider typesafe`,
`setup --provider semantic`, and `setup --provider laya` install provider
packages from the user's package index. Model weights are separate.

This repository does not install a skill into your global agent configuration
automatically. The same CLI can be used by any agent that can run local commands.

## Storage and updates

The store is `memories.sqlite3`, with WAL journaling, full synchronous commits,
foreign keys, unique node IDs, indexed terms/edges, and atomic ingestion. Writes
from independent processes append to the live database. There is no final
whole-file JSON overwrite. One writer decides and commits its links at a time;
readers use a consistent snapshot and can open during an active writer.

Each valid input has a raw observation record. Greetings may be excluded from
the curated graph; exact repeats reuse a node while preserving their raw record.
Fuzzy duplicate scores cannot drop numerical corrections. An idempotency key
makes a retried request return its original result without another raw record;
reusing that key for different input raises an error.

Explicit updates supersede only an older observation about the same claim.
For application-specific properties, supply stable `metadata.fact_key` values:

```bash
python -m engram remember --persist ./engram_store \
  --text "The preferred compiler is A." --timestamp 2026-10-23 \
  --metadata '{"fact_key":"project/compiler"}' --json
python -m engram remember --persist ./engram_store \
  --text "The preferred compiler is now B." --timestamp 2026-10-24 \
  --metadata '{"fact_key":"project/compiler"}' --json
```

The offline heuristic recognizes a limited set of common properties. An entity
name alone cannot establish supersession. Historical backfills cannot replace
newer facts, and unknown dates stay unknown. Time links reconnect on backfill.
Deleting a replacement leaves its earlier fact marked historical.

Opening a directory containing legacy `memories.json` imports it transactionally
once, rebuilds embeddings/time links, and leaves the original JSON untouched.
Raw observation logging begins with the new engine; it cannot reconstruct inputs
that an old version discarded. `store.nodes` and `store.edges` now return
snapshots; use store methods to persist modifications.

## Providers and bring your own key

The default controller is the offline heuristic. Configuration is validated;
unknown settings, nonfinite probabilities, and invalid choice values fail
validation. Provider errors use a visible heuristic fallback, reported in
`runtime` or the query trace. Retries and controller calls are capped.

TypeSafe uses its native decision SDK:

```bash
python -m pip install '.[typesafe]'
# Set TYPESAFE_API_KEY in the process environment.
python -m engram query --backend typesafe --persist ./engram_store \
  --question "What is the API timeout?" --json
```

`--backend llm` uses a chat endpoint that supports the OpenAI-compatible API.
Set `ENGRAM_API_KEY` and optionally `ENGRAM_LLM_BASE_URL`; loopback endpoints can
be keyless. The same generic endpoint is used by decisions and answers unless
role-specific settings override it. `ENGRAM_CONTROL_API_KEY` and
`ENGRAM_ANSWER_API_KEY` separate keys. TypeSafe control uses
`ENGRAM_CONTROL_API_KEY` or `TYPESAFE_API_KEY`, never an unrelated OpenAI key.

Changing the query controller reuses existing stored links; it does not regrade
previous writes. Use the intended controller when building memory.

Role-specific config fields are `control_model`, `answer_model`,
`control_base_url`, and `answer_base_url`. For a chat backend, URL precedence is
role config, generic config, role environment URL, generic environment URL,
then the default endpoint. For TypeSafe, `control_base_url` overrides the native
SDK endpoint. Load an environment file explicitly with `--env-file PATH`;
arbitrary working-directory `.env` files are not loaded automatically.

For local Laya, install `.[laya]`, prepare a local checkpoint, and set
`decision_backend` to `laya`, `laya_model` to its directory, and optionally
`laya_device` and `laya_max_length` in a config JSON. The adapter rejects
truncated or unverifiable judgments. Set `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` when requiring the model SDK to use local files only.
Its weights/runtime have not been exercised in this repository's automated
checks; adapter tests use a fake model. See the official
[Laya documentation](https://github.com/NandhaKishorM/laya) and
[TypeSafe SDK documentation](https://docs.typesafe.ai/sdk/python).

## Semantic embeddings

The default encoder is deterministic feature hashing, suitable for lexical
recall and offline smoke tests. For semantic recall, install `.[semantic]` and
pass `--embedding-model /path/to/sentence-transformers-checkpoint`. Cached model
IDs also work. Downloads require explicit `--allow-model-download`.

Stores record encoder identity and vector dimension. A different encoder
requires an explicit `reindex` command. Keep the same embedding option on every
subsequent invocation. Reindex after changing weights at the same model path.

```bash
python -m engram reindex --persist ./engram_store \
  --embedding-model /path/to/checkpoint --json
```

Library callers can use `HashedEncoder`, `SentenceEncoder`, or
`CallableEncoder(function, model_id="my-model-revision")`. A store captures its
encoder; changing the process default affects only new stores.

## Import, export, and forget

```bash
python -m engram build --input examples/observations.json --persist ./engram_store --json
python -m engram export --persist ./engram_store --output memories-export.json --json
python -m engram forget --persist ./engram_store --id 12 --json
python -m engram forget --persist ./engram_store --scope project-a --all --json
```

Imports validate the whole batch and commit all or none. Inputs accept strings
or objects with `text`, `timestamp`, `source`, `entities`, `metadata`, and
`idempotency_key`. ISO timestamps without a timezone use UTC. Every command
supports `--scope`; lookups, links, exports, and deletion respect it. Scopes are
local organizational boundaries, not authentication for hostile tenants.

Forgetting a node removes its links, term postings, and associated raw records.
`forget --all` also removes filtered raw inputs from that scope. This is logical
deletion from the active database; previously exported files/backups are separate.
Exports are consistent snapshots written by atomic file replacement.

## Library

```python
from engram import Engram, EngramConfig

with Engram("./engram_store", config=EngramConfig(offline=True)) as memory:
    memory.ingest("Dev borrowed Noor's tripod.", source="chat-42", scope="project-a")
    result = memory.query("Who has Noor's tripod?", scope="project-a")
    print(result["context"])
```

## Limits and verification

Retrieval checks relevance before novelty, respects graph/call/node/edge/context
budgets, preserves the initial allocation in its trace, and grades the evidence
that actually fits the returned top-k/context. Missing facts or unresolved
conflicts prevent a sufficient-evidence stop. Multi-hop routing controls depth.
Context limits count UTF-8 evidence bytes, not model tokens.

Vector search is a streaming exact scan, not an approximate vector index.
Scans check deadlines between rows; model and SQLite operations are synchronous.
A deadline rejects late results but cannot interrupt an in-flight native model
forward or guarantee a hard wall-clock SLA. Writers serialize decisions inside
a transaction, so remote latency can delay other writers.

```bash
python -m unittest discover -s tests -t .
```

Run the labeled retrieval smoke evaluation with:

```bash
python -m engram.evaluate --input examples/evaluation.json --output evaluation.json
```

The JSON report records dataset/source hashes, configuration, evidence recall,
gold precision, no-match accuracy, forbidden evidence, latency, and fallback
counts. Its small synthetic fixture is not a long-conversation benchmark.

Tests cover durability across killed/concurrent processes, update history,
rollback, scopes, endpoint routing, malformed outputs, bounded retrieval, and
CLI round trips. The optional real TypeSafe SDK is tested over an intercepted
HTTP transport, without a live API call. No paid-model accuracy or hardware
inference benchmark is implied. Remaining engineering/evaluation work is in
[ENGINEERING.md](ENGINEERING.md).
