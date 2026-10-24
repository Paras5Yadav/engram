"""Adapter contracts without paid calls or weight downloads, plus process CLI tests."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from engram import Engram, EngramConfig, HeuristicBackend, LayaBackend, TypeSafeBackend
from engram.embeddings import content_terms, set_encoder
from engram.questions import admission_questions, consolidation_questions
from engram.runtime import DecisionSession

ROOT = Path(__file__).resolve().parents[1]


class LayaContracts(unittest.TestCase):
    class Agent:
        def __init__(self, usage):
            self.usage = usage
        def predict(self, state, questions, **kwargs):
            self.state, self.questions = state, questions
            return {'answers': {k: {'choice': 'keep_apart'} if q['type'] == 'choice' else {'noul': .8}
                                for k, q in questions.items()}, 'usage': self.usage}

    def test_local_adapter_preserves_choice_and_boolean_semantics(self):
        agent = self.Agent({'truncated': False, 'state_tokens_dropped': 0})
        backend = LayaBackend(agent=agent)
        answers = backend.ask({'observation': 'New fact.'}, consolidation_questions())
        self.assertEqual(answers['representation'], 'keep_apart')
        self.assertEqual(answers['supersedes'], .8)
        self.assertEqual(set(agent.questions['supersedes']['criteria']), {'true', 'false'})
        self.assertEqual(agent.questions['representation']['criteria'], consolidation_questions()['representation'].options)

    def test_truncation_and_missing_telemetry_cause_visible_fallback(self):
        for usage in ({}, {'truncated': True}, {'truncated': False, 'state_tokens_dropped': 20}):
            session = DecisionSession(LayaBackend(agent=self.Agent(usage)), EngramConfig(max_retries=0))
            answer = session.ask({'observation': 'Mira likes tea.'}, admission_questions())
            self.assertEqual(session.fallbacks, 1)
            self.assertEqual(session.last_backend, 'heuristic')
            self.assertEqual(answer['keep'], .9)

    def test_missing_local_checkpoint_fails_before_model_import(self):
        with self.assertRaises(ValueError):
            LayaBackend(model_path='/does/not/exist')


class AdditionalContracts(unittest.TestCase):
    def test_unicode_combining_marks_are_part_of_words(self):
        self.assertIn('मूंगफली', content_terms('मुझे मूंगफली से एलर्जी है।'))
        self.assertIn('not', content_terms('Sam is not allergic.'))

    def test_anonymous_custom_encoder_cannot_change_persistent_identity(self):
        with self.assertRaises(ValueError):
            set_encoder(lambda text: [1, 2])

    def test_forgetting_an_update_does_not_make_its_old_fact_current(self):
        with Engram(config=EngramConfig(offline=True)) as app:
            old = app.ingest('Sam climbs at Riverside gym.', 1)['node_id']
            new = app.ingest('Sam now climbs at Harbor gym.', 2)['node_id']
            app.forget(new)
            result = app.ask('Which gym does Sam climb at now?')
            self.assertNotIn('Most relevant:', result['answer'])
            self.assertIn('Current information is not established', result['answer'])
            self.assertEqual(app.store.get(old).flags.get('superseded'), 'true')


class CliContracts(unittest.TestCase):
    def run_cli(self, *args, cwd=None):
        env = {k: v for k, v in os.environ.items() if k not in ('ENGRAM_API_KEY', 'OPENAI_API_KEY', 'TYPESAFE_API_KEY', 'ENGRAM_LLM_BASE_URL')}
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        env['PYTHONPATH'] = str(ROOT)
        return subprocess.run([sys.executable, '-m', 'engram', *args], cwd=cwd or ROOT,
                              env=env, text=True, capture_output=True, timeout=20)

    def test_cli_roundtrip_recall_export_and_forget_in_separate_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            persist = str(Path(directory) / 'store')
            command = self.run_cli('remember', '--persist', persist, '--scope', 'demo', '--text', 'Mira likes tea.', '--timestamp', '123', '--offline', '--json')
            self.assertEqual(command.returncode, 0, command.stderr)
            node_id = json.loads(command.stdout)['node_id']
            command = self.run_cli('query', '--persist', persist, '--scope', 'demo', '--question', 'What does Mira like?', '--offline', '--json')
            self.assertEqual(command.returncode, 0, command.stderr)
            self.assertEqual(json.loads(command.stdout)['evidence'][0]['node']['node_id'], node_id)
            export = str(Path(directory) / 'data.json')
            command = self.run_cli('export', '--persist', persist, '--scope', 'demo', '--output', export, '--json')
            self.assertEqual(command.returncode, 0, command.stderr)
            self.assertEqual(len(json.loads(Path(export).read_text())['nodes']), 1)
            command = self.run_cli('forget', '--persist', persist, '--scope', 'demo', '--id', str(node_id), '--json')
            self.assertTrue(json.loads(command.stdout)['deleted'])
            command = self.run_cli('ask', '--persist', persist, '--scope', 'demo', '--question', 'What does Mira like?', '--offline', '--json')
            self.assertFalse(json.loads(command.stdout)['evidence_ids'])

    def test_invalid_cli_input_is_json_and_does_not_ingest_partial_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = Path(directory) / 'input.json'
            payload.write_text(json.dumps([{'text': 'Good first fact.'}, {'text': 42}]))
            command = self.run_cli('build', '--persist', str(Path(directory) / 'store'), '--input', str(payload), '--json')
            self.assertEqual(command.returncode, 2)
            self.assertIn('error', json.loads(command.stdout))
            with Engram(str(Path(directory) / 'store')) as app:
                self.assertFalse(app.store.nodes)

    def test_cli_does_not_implicitly_read_dotenv_from_an_arbitrary_project(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, '.env').write_text('ENGRAM_LLM_BASE_URL=invalid\nOPENAI_API_KEY=unintended\n')
            command = self.run_cli('ask', '--persist', str(Path(directory) / 'store'), '--question', 'Any memories?', '--offline', '--json', cwd=directory)
            self.assertEqual(command.returncode, 0, command.stdout + command.stderr)
            self.assertEqual(json.loads(command.stdout)['mode'], 'template')


@unittest.skipUnless(importlib.util.find_spec('typesafe_sdk'), 'optional TypeSafe SDK not installed')
class NativeSdkContract(unittest.TestCase):
    def test_real_sdk_serializes_and_parses_native_batch_over_mock_transport(self):
        import httpx2
        from typesafe_sdk import RetryPolicy, TypeSafeClient
        requests = []
        def handle(request):
            body = json.loads(request.content)
            requests.append(body)
            answers = {}
            for name, question in body['questions'].items():
                if question['type'] == 'choice':
                    options = question['criteria']
                    answers[name] = {'type': 'choice', 'choice': 'keep_apart', 'confidence': .8,
                                     'probabilities': {key: 1 / len(options) for key in options}}
                else:
                    answers[name] = {'type': 'noul', 'noul': .9}
            return httpx2.Response(200, json={'model': 'decision-test', 'usage': {'input_tokens': 20, 'output_tokens': 0}, 'answers': answers})
        with TypeSafeClient(api_key='local-test-key', base_url='https://example.invalid/v1',
                            retry=RetryPolicy(max_retries=0), transport=httpx2.MockTransport(handle)) as client:
            values = TypeSafeBackend(client=client).ask_many([({'observation': 'A'}, admission_questions()),
                                                             ({'candidate': 'B'}, consolidation_questions())])
        self.assertEqual(len(requests), 1)
        self.assertEqual(len(values), 2)
        self.assertEqual(values[1]['representation'], 'keep_apart')
        self.assertIn('jobs[1]', requests[0]['questions']['j1__supersedes']['instructions'])


class ReindexAndDeadlineContracts(unittest.TestCase):
    def test_explicit_reindex_rebuilds_even_when_model_id_is_unchanged(self):
        from engram import CallableEncoder, MemoryStore
        with tempfile.TemporaryDirectory() as directory:
            with MemoryStore.load(directory, encoder=CallableEncoder(lambda text: [1, 0], 'pinned-test-model')) as store:
                store.add('Mira likes tea.', 1)
            with MemoryStore.load(directory, encoder=CallableEncoder(lambda text: [0, 1], 'pinned-test-model'), reindex=True) as store:
                self.assertEqual(store.get(1).embedding, [0.0, 1.0])

    def test_vector_scan_reports_deadline_instead_of_unbounded_scan(self):
        import time

        from engram import MemoryStore
        with MemoryStore() as store:
            store.add('Mira likes tea.', 1)
            self.assertFalse(store.vector_search('tea', deadline=time.monotonic() - 1))
            self.assertFalse(store.last_vector_search['complete'])
            self.assertEqual(store.last_vector_search['scanned'], 0)


class FinalRegressionContracts(unittest.TestCase):
    def test_json_argument_errors_have_a_machine_readable_response(self):
        command = subprocess.run([sys.executable, '-m', 'engram', 'query', '--top-k', 'bad', '--json'],
                                 cwd=ROOT, capture_output=True, text=True, timeout=10, check=False)
        self.assertEqual(command.returncode, 2)
        self.assertEqual(json.loads(command.stdout)['error']['type'], 'ArgumentError')

    def test_offline_answers_ignore_invalid_provider_environment(self):
        from unittest.mock import patch
        with patch.dict(os.environ, {'ENGRAM_LLM_BASE_URL': 'invalid', 'OPENAI_API_KEY': 'unintended'}):
            with Engram(config=EngramConfig(offline=True)) as app:
                app.ingest('Mira likes tea.')
                self.assertEqual(app.ask('What does Mira like?')['mode'], 'template')

    def test_known_conflict_cannot_be_overruled_by_an_optimistic_judge(self):
        class Optimistic(HeuristicBackend):
            def ask(self, state, questions):
                values = super().ask(state, questions)
                if 'enough' in questions:
                    values.update(enough=1.0, conflicts_present=0.0, missing_piece=0.0)
                return values
        with Engram(backend=Optimistic()) as app:
            first = app.ingest('The API timeout is 30 seconds.', 1)['node_id']
            second = app.ingest('The API timeout is 60 seconds.', 2)['node_id']
            app.store.link('topic', first, second, 'conflicts_with')
            result = app.query('What is the API timeout?')
            self.assertEqual(set(result['trace']['evidence_ids']), {first, second})
            self.assertNotEqual(result['trace']['stop_reason'], 'enough_evidence')


class ConnectionLifecycleContract(unittest.TestCase):
    def test_legacy_callers_without_explicit_close_release_connections(self):
        import gc
        import sqlite3

        from engram import MemoryStore
        store = MemoryStore()
        connection = store._conn
        del store
        gc.collect()
        with self.assertRaises(sqlite3.ProgrammingError):
            connection.execute('SELECT 1')


if __name__ == '__main__':
    unittest.main()
