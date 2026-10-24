"""Engine regression contracts: durability, relevance, updates, and provider faults."""
from __future__ import annotations

import io
import json
import math
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import closing, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from engram import (
    AnswerSynthesizer,
    CallableEncoder,
    Engram,
    EngramConfig,
    HashedEncoder,
    HeuristicBackend,
    LLMJudgeBackend,
    MemoryStore,
    TypeSafeBackend,
    WritePipeline,
    load_config,
)
from engram.api import parse_timestamp
from engram.decisions import BackendUnavailable, ask_with_fallback
from engram.embeddings import cosine, current_encoder, set_encoder
from engram.questions import (
    admission_questions,
    consolidation_questions,
)
from engram.retrieval import allocate_budgets
from engram.runtime import DecisionSession, make_backend

ROOT = Path(__file__).resolve().parents[1]


def pair(first, second, times=(1, 2)):
    store = MemoryStore()
    writer = WritePipeline(store)
    return store, writer.ingest(first, times[0]), writer.ingest(second, times[1])


class WriteContracts(unittest.TestCase):
    def test_numeric_correction_is_retained_with_raw_history(self):
        store, old, new = pair('Production timeout is 30 seconds for all user requests.',
                               'Production timeout is 60 seconds for all user requests.')
        self.assertTrue(new['stored'])
        self.assertEqual(len(store.observations()), 2)
        self.assertEqual(len(store.nodes), 2)

    def test_exact_repeat_is_deduplicated_but_raw_observation_survives(self):
        store, old, new = pair('Mira prefers weekly email reminders.', 'Mira prefers weekly email reminders.')
        self.assertFalse(new['stored'])
        self.assertEqual(new['node_id'], old['node_id'])
        self.assertEqual(len(store.nodes), 1)
        self.assertEqual(len(store.observations()), 2)

    def test_greetings_remain_in_raw_history(self):
        with Engram() as app:
            result = app.ingest('hi')
            self.assertEqual(result['reason'], 'filtered')
            self.assertEqual(app.store.observations()[0]['text'], 'hi')

    def test_unicode_fact_is_recallable(self):
        with Engram() as app:
            result = app.ingest('मुझे मूंगफली से एलर्जी है।')
            self.assertTrue(result['stored'])
            self.assertTrue(app.query('मूंगफली एलर्जी')['evidence'])

    def test_unrelated_change_cannot_supersede_another_fact_about_same_entity(self):
        store, old, new = pair('Atlas uses PostgreSQL to store user profiles.',
                               'Atlas now supports image uploads for profile pictures.')
        self.assertNotIn('superseded_by', store.get(old['node_id']).flags)

    def test_backfilled_update_cannot_replace_a_newer_fact(self):
        store, old, new = pair('Sam now climbs at Harbor gym.', 'Sam switched to Riverside gym.', (200, 100))
        self.assertNotIn('superseded_by', store.get(old['node_id']).flags)

    def test_repeated_updates_keep_a_chain(self):
        with Engram() as app:
            rows = ['Atlas stores logs in Redis.', 'Atlas switched log storage to PostgreSQL.', 'Atlas switched log storage to SQLite.']
            ids = [app.ingest(text, i + 1)['node_id'] for i, text in enumerate(rows)]
            self.assertEqual(app.store.get(ids[0]).flags['superseded_by'], str(ids[1]))
            self.assertEqual(app.store.get(ids[1]).flags['superseded_by'], str(ids[2]))
            self.assertNotIn('superseded_by', app.store.get(ids[2]).flags)

    def test_update_can_supersede_without_conflict(self):
        class Update(HeuristicBackend):
            def ask(self, state, questions):
                values = super().ask(state, questions)
                if 'supersedes' in questions:
                    values.update(conflicts=0.0, supersedes=1.0)
                return values
        store = MemoryStore()
        writer = WritePipeline(store, Update())
        old = writer.ingest('Sam climbs at Riverside gym.', 1)
        new = writer.ingest('Sam now climbs at Harbor gym.', 2)
        self.assertEqual(store.get(old['node_id']).flags['superseded_by'], str(new['node_id']))

    def test_metadata_identifies_custom_updatable_claim(self):
        with Engram() as app:
            old = app.ingest('The preferred compiler is A.', 1, metadata={'fact_key': 'project/compiler'})
            new = app.ingest('The preferred compiler is now B.', 2, metadata={'fact_key': 'project/compiler'})
            self.assertEqual(app.store.get(old['node_id']).flags['superseded_by'], str(new['node_id']))

    def test_unknown_and_known_dates_do_not_create_false_update_order(self):
        store, old, new = pair('Sam climbs at Riverside gym.', 'Sam now climbs at Harbor gym.', (100, None))
        self.assertNotIn('superseded_by', store.get(old['node_id']).flags)

    def test_chronology_alone_is_not_causation(self):
        store, *_ = pair('Sam attended the Riverside gym class.', 'Sam bought shoes after attending the Riverside gym class.')
        self.assertFalse(store.edges['cause'])

    def test_backfill_reconnects_time_chain(self):
        store = MemoryStore()
        ids = {ts: store.add(f'Observation dated {ts}.', ts).node_id for ts in (30, 10, 20)}
        self.assertEqual({(e.src, e.dst) for e in store.edges['time']}, {(ids[10], ids[20]), (ids[20], ids[30])})

    def test_unknown_dates_have_no_time_edges(self):
        store = MemoryStore()
        store.add('Undated event.', None)
        store.add('Dated event.', 1)
        self.assertFalse(store.edges['time'])

    def test_idempotent_retry_does_not_duplicate_raw_history(self):
        with Engram() as app:
            first = app.ingest('Remember the conference code.', 1, idempotency_key='event-1')
            second = app.ingest('Remember the conference code.', 1, idempotency_key='event-1')
            self.assertEqual(first['node_id'], second['node_id'])
            self.assertTrue(second['replayed'])
            self.assertEqual(len(app.store.observations()), 1)
            with self.assertRaises(ValueError):
                app.ingest('Different event.', 1, idempotency_key='event-1')
            self.assertEqual(len(app.store.nodes), 1)

    def test_failed_ingest_is_atomic(self):
        class Broken(HeuristicBackend):
            def ask(self, state, questions):
                if 'topic_link' in questions:
                    raise RuntimeError('unexpected programming failure')
                return super().ask(state, questions)
        store = MemoryStore()
        old = WritePipeline(store).ingest('Sam climbs at Riverside gym.', 1)
        with self.assertRaises(RuntimeError):
            WritePipeline(store, Broken()).ingest('Sam now climbs at Harbor gym.', 2)
        self.assertEqual(len(store.nodes), 1)
        self.assertEqual(len(store.observations()), 1)
        self.assertFalse(store.get(old['node_id']).flags)

    def test_invalid_batch_has_no_partial_writes(self):
        with Engram() as app:
            with self.assertRaises(ValueError):
                app.ingest_many([{'text': 'Valid first fact.'}, {'text': 'Invalid timestamp.', 'timestamp': float('nan')}])
            self.assertFalse(app.store.nodes)
            self.assertFalse(app.store.observations())


class StorageContracts(unittest.TestCase):
    def test_two_live_writers_cannot_overwrite_each_other(self):
        with tempfile.TemporaryDirectory() as directory:
            with MemoryStore.load(directory) as first, MemoryStore.load(directory) as second:
                a = first.add('Alpha independent result.', 1)
                b = second.add('Beta independent result.', 2)
                first.save(directory)
                second.save(directory)
                self.assertNotEqual(a.node_id, b.node_id)
                self.assertEqual(len(first.nodes), 2)
            with MemoryStore.load(directory) as restored:
                self.assertEqual(len(restored.nodes), 2)

    def test_parallel_processes_keep_all_commits_and_unique_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            with MemoryStore.load(directory):
                pass
            script = ('import sys; from engram import MemoryStore; '
                      's=MemoryStore.load(sys.argv[1]); '
                      '[s.add(f"Agent {sys.argv[2]} result {i}", int(sys.argv[2])*100+i) for i in range(12)]; s.close()')
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
            processes = [subprocess.Popen([sys.executable, '-c', script, directory, str(i)], cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for i in range(3)]
            for process in processes:
                out, err = process.communicate(timeout=20)
                self.assertEqual(process.returncode, 0, err)
            with MemoryStore.load(directory) as restored:
                self.assertEqual(len(restored.nodes), 36)
                self.assertEqual(len({n.text for n in restored.nodes.values()}), 36)
                self.assertEqual(len(restored.edges['time']), 35)

    def test_killed_writer_leaves_committed_store_readable(self):
        with tempfile.TemporaryDirectory() as directory:
            with MemoryStore.load(directory) as store:
                store.add('Committed before crash.', 1)
            script = '''import sys,time
from engram import MemoryStore
s=MemoryStore.load(sys.argv[1])
with s.transaction():
    s.add("Uncommitted before crash.", 2)
    print("transaction-open", flush=True)
    time.sleep(30)
'''
            process = subprocess.Popen([sys.executable, '-c', script, directory], cwd=ROOT,
                                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                self.assertEqual(process.stdout.readline().strip(), 'transaction-open')
                with MemoryStore.load(directory) as reader:
                    self.assertEqual([n.text for n in reader.nodes.values()], ['Committed before crash.'])
            finally:
                process.kill()
                process.communicate(timeout=10)
            with MemoryStore.load(directory) as store:
                self.assertEqual([n.text for n in store.nodes.values()], ['Committed before crash.'])
                store.add('Committed after recovery.', 3)
                self.assertEqual(len(store.nodes), 2)

    def test_legacy_json_migration_keeps_original_and_reindexes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'memories.json'
            old = MemoryStore()
            node = old.add('Mira prefers tea.', 1)
            payload = {'next_id': 2, 'nodes': [node.to_dict()], 'edges': {v: [] for v in old.edges}}
            payload['nodes'][0]['embedding'] = [999.0]
            original = json.dumps(payload)
            path.write_text(original)
            with MemoryStore.load(directory) as migrated:
                self.assertEqual(migrated.get(1).text, node.text)
                self.assertEqual(len(migrated.get(1).embedding), 256)
                migrated.add('Mira also prefers coffee.', 2)
            self.assertEqual(path.read_text(), original)
            with MemoryStore.load(directory) as migrated:
                self.assertEqual(len(migrated.nodes), 2)

    def test_failed_migration_can_be_retried_without_data_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'memories.json'
            path.write_text('{broken')
            with self.assertRaises(ValueError):
                MemoryStore.load(directory)
            self.assertEqual(path.read_text(), '{broken')
            path.write_text('{"nodes":[],"edges":{}}')
            with MemoryStore.load(directory) as store:
                store.add('Recovered migration.', 1)
                self.assertEqual(len(store.nodes), 1)

    def test_memory_snapshot_cannot_replace_a_nonempty_database(self):
        with tempfile.TemporaryDirectory() as directory:
            with MemoryStore.load(directory) as store:
                store.add('Keep the existing database.', 1)
            snapshot = MemoryStore()
            snapshot.add('Different snapshot.', 2)
            with self.assertRaises(ValueError):
                snapshot.save(directory)
            with MemoryStore.load(directory) as store:
                self.assertEqual([n.text for n in store.nodes.values()], ['Keep the existing database.'])

    def test_future_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with MemoryStore.load(directory):
                pass
            with closing(sqlite3.connect(Path(directory) / 'memories.sqlite3')) as db, db:
                db.execute("UPDATE meta SET value='999' WHERE key='schema'")
            with self.assertRaises(ValueError):
                MemoryStore.load(directory)

    def test_scopes_filter_search_links_export_and_deletion(self):
        with Engram() as app:
            a = app.ingest('Mira prefers weekly email reminders.', 1, scope='alpha')['node_id']
            b = app.ingest('Mira prefers daily SMS reminders.', 2, scope='beta')['node_id']
            result = app.query('Mira reminders', scope='alpha')
            self.assertEqual({c['node']['node_id'] for c in result['evidence']}, {a})
            self.assertEqual([n['node_id'] for n in app.store.export(scope='beta')['nodes']], [b])
            with self.assertRaises(ValueError):
                app.store.link('topic', a, b, 'shares_topic')
            self.assertFalse(app.forget(b, scope='alpha'))
            self.assertTrue(app.forget(a, scope='alpha'))
            self.assertFalse(app.store.observations('alpha'))
            self.assertTrue(app.store.get(b))

    def test_forget_removes_links_terms_and_duplicate_raw_inputs(self):
        with Engram() as app:
            first = app.ingest('Mira prefers tea.', 1)['node_id']
            app.ingest('Mira prefers tea.', 2)
            app.ingest('Mira likes coffee.', 3)
            self.assertTrue(app.forget(first))
            self.assertFalse(app.store.keyword_search('tea'))
            self.assertEqual(len(app.store.observations()), 1)
            self.assertFalse(any(e.src == first or e.dst == first for edges in app.store.edges.values() for e in edges))

    def test_clear_removes_filtered_input_only_in_selected_scope(self):
        with Engram() as app:
            app.ingest('hi', scope='alpha')
            app.ingest('Mira likes coffee.', scope='beta')
            app.store.clear(scope='alpha')
            self.assertFalse(app.store.observations('alpha'))
            self.assertEqual(len(app.store.observations('beta')), 1)

    def test_export_is_consistent_json_and_contains_provenance(self):
        with tempfile.TemporaryDirectory() as directory, Engram() as app:
            app.ingest('Mira likes tea.', '2026-10-24T12:00:00Z', source='chat-42', metadata={'tag': 'preference'})
            path = app.export(str(Path(directory) / 'export.json'))
            payload = json.loads(Path(path).read_text())
            self.assertEqual(payload['nodes'][0]['source'], 'chat-42')
            self.assertEqual(payload['observations'][0]['metadata'], {'tag': 'preference'})
            self.assertIn('encoder', payload)


class RetrievalContracts(unittest.TestCase):
    def test_unrelated_query_has_no_match(self):
        with Engram() as app:
            app.ingest('Mira prefers concise weekly email reminders.', 1)
            result = app.query('What is the production database password?')
            self.assertFalse(result['evidence'])
            self.assertEqual(result['trace']['stop_reason'], 'no_match')

    def test_novel_irrelevant_time_neighbor_is_rejected(self):
        with Engram(config=EngramConfig(anchor_count=1, top_k=2)) as app:
            app.ingest('Atlas was deployed on 2026-10-20.', 1)
            noise = app.ingest('Bananas thrive beside tropical rivers during summer.', 2)['node_id']
            result = app.query('When was Atlas deployed?')
            self.assertNotIn(noise, [c['node']['node_id'] for c in result['evidence']])

    def test_budget_allocation_reserves_floors_without_overspending(self):
        for total in range(30):
            for floor in (1, 2, 3):
                cfg = EngramConfig(total_graph_budget=total, minimum_graph_budget=floor, activation_threshold=0)
                budgets = allocate_budgets({'topic': 1, 'time': .01, 'cause': .01, 'entity': .01}, cfg)
                self.assertLessEqual(sum(budgets.values()), total)
                self.assertTrue(all(v == 0 or v >= floor for v in budgets.values()))

    def test_stopping_grades_returned_top_k_and_conflict_blocks_sufficiency(self):
        class Stop(HeuristicBackend):
            def __init__(self):
                self.graded = None
            def ask(self, state, questions):
                values = super().ask(state, questions)
                if 'enough' in questions:
                    self.graded = list(state['evidence_ids'])
                    values.update(enough=1.0, missing_piece=1.0, conflicts_present=1.0, worth_continuing=1.0)
                return values
        backend = Stop()
        with Engram(backend=backend) as app:
            app.ingest('The API team owns HTTP services.', 1)
            app.ingest('The Data team owns database storage.', 2)
            result = app.query('Which teams own the API and database?', top_k=1)
            self.assertEqual(backend.graded, result['trace']['evidence_ids'])
            self.assertNotEqual(result['trace']['stop_reason'], 'enough_evidence')

    def test_context_bytes_and_graph_work_are_bounded(self):
        cfg = EngramConfig(max_context_bytes=190, maximum_nodes=3, maximum_edges=2, maximum_calls=3)
        with Engram(config=cfg) as app:
            for i in range(8):
                app.ingest(f'Mira likes tea flavor {i}.', i + 1)
            result = app.query('What tea does Mira like now and before?')
            trace = result['trace']
            self.assertLessEqual(len(result['context'].encode('utf-8')), 190)
            self.assertEqual(trace['context_bytes'], len(result['context'].encode('utf-8')))
            self.assertLessEqual(trace['nodes_examined'], 3)
            self.assertLessEqual(trace['edges_examined'], 2)
            self.assertLessEqual(trace['calls'], 3)
            self.assertTrue(all(trace['budget_used'][v] <= trace['budgets'][v] for v in trace['budgets']))

    def test_initial_budgets_remain_in_trace(self):
        with Engram() as app:
            app.ingest('Mira likes tea.', 1)
            app.ingest('Mira likes coffee.', 2)
            trace = app.query('What does Mira like?')['trace']
            self.assertEqual(sum(trace['budgets'].values()), app.config.total_graph_budget)
            self.assertGreater(sum(trace['budget_used'].values()), 0)

    def test_zero_negative_and_boolean_top_k_are_rejected(self):
        with Engram() as app:
            for value in (0, -1, True):
                with self.assertRaises(ValueError):
                    app.query('Which fact?', top_k=value)


class ProviderContracts(unittest.TestCase):
    def test_malformed_and_nonfinite_probabilities_fall_back(self):
        for value in ('"likely"', 'NaN', 'Infinity', 'true', '-0.1', '1.1', 'null'):
            with self.subTest(value=value), patch('engram.decisions.chat_completion', return_value='{"keep":' + value + '}'):
                answers, backend = ask_with_fallback({}, {'keep': admission_questions()['keep']}, LLMJudgeBackend(), HeuristicBackend())
                self.assertEqual(backend, 'heuristic')
                self.assertTrue(math.isfinite(answers['keep']))

    def test_choice_descriptions_and_batch_states_reach_the_provider(self):
        questions = {'representation': consolidation_questions()['representation']}
        backend = LLMJudgeBackend()
        with patch('engram.decisions.chat_completion', return_value='{"0":{"representation":"keep_apart"},"1":{"representation":"combine"}}') as call:
            values = backend.ask_many([({'candidate': 'A'}, questions), ({'candidate': 'B'}, questions)])
            self.assertEqual([v['representation'] for v in values], ['keep_apart', 'combine'])
            self.assertEqual(call.call_count, 1)
            self.assertIn('Nothing is lost'.lower(), call.call_args.args[0][1]['content'].lower())

    def test_relation_pairs_are_one_physical_batch_request(self):
        class Batch:
            name = 'batch-spy'
            def __init__(self):
                self.jobs = []
            def ask_many(self, jobs, **kwargs):
                self.jobs.append(jobs)
                return [HeuristicBackend().ask(s, q) for s, q in jobs]
        backend = Batch()
        with Engram(backend=backend) as app:
            app.ingest('Mira enjoys tea.', 1)
            app.ingest('Mira enjoys coffee.', 2)
            backend.jobs.clear()
            app.ingest('Mira now enjoys cocoa.', 3)
            relations = [batch for batch in backend.jobs if 'topic_link' in batch[0][1]]
            self.assertEqual(len(relations), 1)
            self.assertEqual(len(relations[0]), 2)
            self.assertIn('supersedes', relations[0][0][1])

    def test_retries_and_calls_are_capped_and_failure_is_visible(self):
        class Down:
            name = 'down'
            def __init__(self):
                self.calls = 0
            def ask_many(self, jobs, **kwargs):
                self.calls += 1
                raise BackendUnavailable('offline')
        backend = Down()
        session = DecisionSession(backend, EngramConfig(maximum_calls=2, max_retries=8))
        result = session.ask({'observation': 'Mira likes tea.'}, admission_questions())
        self.assertEqual(backend.calls, 2)
        self.assertEqual(session.fallbacks, 1)
        self.assertEqual(session.last_backend, 'heuristic')
        self.assertGreater(result['keep'], .5)

    def test_cached_queries_do_not_repeat_primary_calls(self):
        backend = LLMJudgeBackend()
        from collections import OrderedDict
        cache = OrderedDict()
        with patch('engram.decisions.chat_completion', return_value='{"0":{"keep":0.9,"fresh":0.9,"duplicate":0.1}}') as call:
            for _ in range(2):
                DecisionSession(backend, EngramConfig(), cache).ask({'observation': 'Mira likes tea.'}, admission_questions())
            self.assertEqual(call.call_count, 1)

    def test_late_results_are_discarded(self):
        class Slow:
            name = 'slow'
            def ask_many(self, jobs, **kwargs):
                time.sleep(.015)
                return [{'keep': 0, 'fresh': 0, 'duplicate': 1}]
        session = DecisionSession(Slow(), EngramConfig(max_latency_seconds=.005, max_retries=0))
        result = session.ask({'observation': 'Mira likes tea.'}, admission_questions())
        self.assertGreater(result['keep'], .5)
        self.assertTrue(session.exhausted)
        self.assertEqual(session.fallbacks, 1)

    def test_answer_and_control_share_env_endpoint_unless_roles_override(self):
        with patch.dict(os.environ, {'ENGRAM_LLM_BASE_URL': 'http://localhost:8123/v1', 'ENGRAM_API_KEY': 'local-test'}, clear=True):
            config = EngramConfig()
            backend = make_backend(config, 'llm')
            self.assertEqual(backend.base_url, 'http://localhost:8123/v1')
            node = MemoryStore().add('Mira likes tea.', 1)
            with patch('engram.answer.chat_completion', return_value='Tea.') as call:
                AnswerSynthesizer(config).answer('What does Mira like?', [node])
                self.assertEqual(call.call_args.args[2], backend.base_url)
            roles = EngramConfig(control_base_url='http://localhost:8124/v1', answer_base_url='http://localhost:8125/v1')
            self.assertNotEqual(roles.endpoint('control'), roles.endpoint('answer'))

    def test_type_safe_adapter_uses_native_types_and_separate_batch_states(self):
        class Fake:
            def system_one(self, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(nouls={k: SimpleNamespace(noul=.9) for k, q in kwargs['questions'].items() if q['type'] == 'noul'},
                                       choices={k: SimpleNamespace(choice='keep_apart') for k, q in kwargs['questions'].items() if q['type'] == 'choice'})
        client = Fake()
        backend = TypeSafeBackend(client=client)
        values = backend.ask_many([({'candidate': 'A'}, admission_questions()), ({'candidate': 'B'}, consolidation_questions())])
        self.assertEqual(len(values), 2)
        self.assertEqual(values[1]['representation'], 'keep_apart')
        self.assertEqual(client.kwargs['state']['jobs'][1]['candidate'], 'B')
        self.assertIn('jobs[1]', client.kwargs['questions']['j1__supersedes']['instructions'])
        self.assertIn('Distinct facts', client.kwargs['questions']['j1__representation']['criteria']['keep_apart'])

    def test_typesafe_does_not_inherit_an_unrelated_openai_key(self):
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'wrong-provider'}, clear=True):
            self.assertIsNone(make_backend(EngramConfig(decision_backend='typesafe')).api_key)

    def test_offline_demo_never_uses_exported_credentials(self):
        from engram.demo import run
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key', 'ENGRAM_API_KEY': 'test-key'}), patch('engram.answer.chat_completion') as call, redirect_stdout(io.StringIO()):
            run()
            self.assertEqual(call.call_count, 0)

    def test_fallback_summary_never_selects_superseded_memory_as_current(self):
        store, old, new = pair('Sam climbs at Riverside gym.', 'Sam now climbs at Harbor gym.')
        result = AnswerSynthesizer(EngramConfig(offline=True)).answer('Which gym does Sam climb at now?', list(store.nodes.values()))
        self.assertEqual(result['answer'].split('Most relevant: ')[-1], 'Sam now climbs at Harbor gym.')
        self.assertIn('superseded_by', result['answer'])


class EncoderAndConfigContracts(unittest.TestCase):
    def test_custom_hash_dimension_and_unnormalized_cosine(self):
        self.assertEqual(len(HashedEncoder(8).encode('memory graph')), 8)
        self.assertAlmostEqual(cosine([10, 0], [1, 1]), 1 / math.sqrt(2))
        for vector in ([float('nan')], [float('inf')], []):
            with self.assertRaises(ValueError):
                cosine(vector, vector)

    def test_nonfinite_invalid_and_unknown_settings_are_rejected(self):
        for kwargs in ({'anchor_count': -1}, {'total_graph_budget': -1}, {'top_k': True},
                       {'coverage_threshold': float('nan')}, {'timeout_seconds': 0},
                       {'llm_base_url': 'https://user:secret@example.com/v1'}, {'offline': 'true'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                EngramConfig(**kwargs)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_text('{"unknown_setting":true}')
            with self.assertRaises(ValueError):
                load_config(str(path))

    def test_encoder_change_requires_explicit_transactional_reindex(self):
        with tempfile.TemporaryDirectory() as directory:
            with MemoryStore.load(directory, encoder=HashedEncoder(8)) as store:
                store.add('Mira likes tea.', 1)
            with self.assertRaises(ValueError):
                MemoryStore.load(directory, encoder=HashedEncoder(16))
            with MemoryStore.load(directory, encoder=HashedEncoder(16), reindex=True) as store:
                self.assertEqual(len(store.get(1).embedding), 16)
                self.assertTrue(store.vector_search('tea'))
                broken = CallableEncoder(lambda text: [float('nan')] if text else [0], 'broken')
                with self.assertRaises(ValueError):
                    store.reindex(broken)
                self.assertEqual(store.encoder.model_id, HashedEncoder(16).model_id)
                self.assertTrue(store.vector_search('tea'))

    def test_default_encoder_changes_do_not_mutate_existing_stores(self):
        previous = current_encoder()
        try:
            store = MemoryStore()
            store.add('Mira likes tea.', 1)
            set_encoder(HashedEncoder(8))
            self.assertEqual(len(store.get(1).embedding), 256)
            self.assertTrue(store.vector_search('tea'))
            self.assertEqual(MemoryStore().dimension, 8)
        finally:
            set_encoder(previous)

    def test_timestamp_conversion_is_explicit_and_deterministic(self):
        self.assertIsNone(parse_timestamp(None))
        self.assertEqual(parse_timestamp('123'), 123)
        self.assertEqual(parse_timestamp('2026-10-24'), parse_timestamp('2026-10-24T00:00:00Z'))
        for value in (True, 'NaN', float('inf'), 'not-a-date'):
            with self.assertRaises(ValueError):
                parse_timestamp(value)


if __name__ == '__main__':
    unittest.main()
