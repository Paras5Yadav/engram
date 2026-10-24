import unittest

from engram.evaluate import evaluate


class EvaluationContracts(unittest.TestCase):
    def test_metrics_report_supported_and_unanswerable_questions(self):
        report = evaluate({'observations': [{'label': 'tea', 'text': 'Mira prefers tea.', 'timestamp': 1}],
                           'queries': [{'question': 'What does Mira prefer?', 'expected': ['tea']},
                                       {'question': 'What is the database password?', 'no_match': True}]})
        self.assertEqual(report['metrics']['gold_evidence_recall'], 1)
        self.assertEqual(report['metrics']['no_match_accuracy'], 1)
        self.assertEqual(len(report['dataset_sha256']), 64)
        self.assertEqual(len(report['source_sha256']), 64)
        self.assertEqual(report['backend'], 'heuristic')

    def test_evaluator_reports_false_matches_and_forbidden_evidence(self):
        report = evaluate({'observations': [{'label': 'tea', 'text': 'Mira prefers tea.'}],
                           'queries': [{'question': 'What does Mira prefer?', 'no_match': True, 'forbidden': ['tea']}]})
        self.assertEqual(report['metrics']['no_match_accuracy'], 0)
        self.assertEqual(report['metrics']['forbidden_evidence_count'], 1)

    def test_unknown_gold_label_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate({'observations': [], 'queries': [{'question': 'What?', 'expected': ['missing']}]})
