# Engine foundation — 0.2

Implemented in the Engram repository on `codex/engram-engine-foundation`.

## Completed

- SQLite live storage, WAL/full commits, nested transactions, serialized writers,
  reader snapshots, secure permissions for newly created stores, and JSON migration.
- Raw observation history, exact-only curated deduplication, retry idempotency,
  atomic imports, claim-scoped updates, temporal backfills, and explicit deletion.
- Strict config/decision validation; byte, graph, candidate, edge, call and
  cooperative time budgets; batched decision requests, bounded cache and retries.
- Relevance-gated anchors/neighbors, missing/conflict checks, stopping over final
  evidence, latest-state ranking, and complete budget/fallback traces.
- Separate controller/answer provider settings, native TypeSafe decisions,
  OpenAI-compatible chat, local Laya adapter, and optional sentence embeddings.
- Local library facade, JSON CLI, portable skill, wheel/skill packaging, and a
  labeled retrieval evaluation runner with source/dataset hashes.

## Release checks

Run unit/regression tests with Python 3.11+; installing the TypeSafe extra also
activates the SDK wire-contract test. Base tests require only the standard
library. A wheel should be installed in a fresh environment and a packaged skill
extracted outside the checkout, then exercised through remember → query →
export → forget. Validate the SKILL.md frontmatter as part of packaging.

No live provider requests or downloaded model weights were used to verify this
milestone. Laya and sentence-transformer accuracy/latency remain unmeasured.

## Quality evaluation and scaling work

1. Run matched retrieval/answer evaluations on held-out long conversations,
   corrections, temporal questions, multilingual facts, unanswerable questions,
   and adversarial retrieved text. Record dataset hashes, model revisions,
   configs, recall/precision, abstention errors, latency, fallback rate, and cost.
2. Calibrate admission, relevance, relation, stopping and supersession thresholds
   for each real backend. The offline heuristics use lexical/entity shortcuts;
   their probabilities are not calibrated semantic judgments.
3. Qualify the Laya checkpoint and actual sentence encoder on target laptops;
   cover model loading, short contexts, truncation rejection, and memory use.
4. Profile large stores and add an incrementally maintained approximate vector
   index if exact scans are insufficient. Preserve cross-process consistency and
   encoder manifests when doing so. Topic clustering currently has no materialized
   summaries; `pattern`/`same_fact` edges retain the original observations.
5. Record decision model/config revisions on persistent links and support an
   explicit graph rebuild when policies change. Switching the query controller
   currently reuses the graph constructed by previous writes.
6. Add long-conversation dataset adapters and run a reproducible comparative evaluation. The
   synthetic regression suite verifies engineering behavior; it does not establish
   measured performance on real conversations.

## Work required for a commercial release

- Test installation/upgrade and SQLite recovery on Windows and Linux as well as
  macOS. Preserve configuration and provide deliberate data/schema migrations.
- Design and verify explicit backup/restore and any encryption/key-management
  requirements. Logical deletion does not erase old exports or backups.
- Add the chosen product identity, distribution/update policy, and support flow.
  Hosted billing/accounts/tenant security would be additional work if a hosted
  service is selected; they are not part of this local runtime.

The skill is a local integration mechanism. Automatic capture of every chat,
MCP adapters, browser UI, hosted APIs, billing, and a real-conversation quality assessment have
not been implemented by this milestone.
