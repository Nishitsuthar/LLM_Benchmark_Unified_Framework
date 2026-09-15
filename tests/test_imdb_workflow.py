"""Offline workflow checks: no credentials, images, or model requests required."""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run
from unified_pipeline.base import EvidenceResult, ModelResponse
from unified_pipeline.evaluators.partial_credit_evaluator import PartialCreditEvaluator


class ImdbWorkflowTests(unittest.TestCase):
    def test_cli_requires_smoke_or_explicit_full_run(self):
        for arguments in (
            ['--dataset', 'imdb_20', '--model', 'gemini_flash'],
            ['--dataset', 'imdb_20', '--model', 'nemotron', '--task_ids', 'F1'],
        ):
            with patch('sys.argv', ['run.py', *arguments]), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    run.parse_args()
                self.assertEqual(error.exception.code, 2)
        with patch('sys.argv', ['run.py', '--dataset', 'imdb_controlled', '--model', 'nemotron']):
            self.assertEqual(run.parse_args().model, 'nemotron')

    def test_accuracy_errors_raw_capture_and_separate_runs(self):
        questions = run.load_questions('evidence/imdb_20/questions.json')
        expected = PartialCreditEvaluator._read_expected_answer(
            str(Path('ground_truth/imdb_20') / questions[0]['ground_truth_file'])
        )

        class FakeRouter:
            def __init__(self, model):
                self.model_name = run.IMDB_MODELS[model]
                self.last_raw_response = None
                self.calls = 0
                self.api_successes = 0
                self.api_failures = 0
                self.client = type('Client', (), {'with_options': lambda self, **kwargs: self})()

            def call(self, prompt_text, images=None):
                assert images == ['fake-image']
                self.calls += 1
                if self.calls == 10:
                    self.api_failures += 1
                    raise RuntimeError('mock API unavailable')
                self.api_successes += 1
                text = expected if self.calls == 1 else 'deliberately incorrect answer'
                self.last_raw_response = {'choices': [{'message': {'content': text}}]}
                return ModelResponse(text, 'raw reasoning', 'stop', 100, 20, 0.25)

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(run, 'RESULTS_DIR', Path(directory)), patch.object(run, 'ModelRouter', FakeRouter), patch.dict(run.EVIDENCE_BUILDERS, {'pdf_visual': lambda: type('Builder', (), {'build': lambda *args: EvidenceResult('', {'images': ['fake-image']})})()}):
                for model in run.IMDB_MODELS:
                    with patch('sys.argv', ['run.py', '--dataset', 'imdb_20', '--model', model, '--confirm-full-run']), contextlib.redirect_stdout(io.StringIO()) as output:
                        run.main()
                    self.assertIn('ACCURACY', output.getvalue())
                    self.assertNotIn('Avg F1', output.getvalue())
                # A separate smoke run cannot append to a previous full run.
                with patch('sys.argv', ['run.py', '--dataset', 'imdb_20', '--model', 'gemini_flash', '--task_ids', 'F1']), contextlib.redirect_stdout(io.StringIO()):
                    run.main()
            summaries = list(Path(directory).rglob('*_summary.json'))
            self.assertEqual(len(summaries), 3)
            for path in summaries:
                summary = json.loads(path.read_text())
                rows = list(csv.DictReader(io.StringIO(path.with_name(path.name.replace('_summary.json', '_metrics.csv')).read_text())))
                if summary['attempted_questions'] == 1:
                    self.assertEqual(summary['accuracy'], 1)
                    self.assertEqual(len(rows), 1)
                    continue
                self.assertEqual(summary['accuracy'], 1 / 9)
                self.assertEqual(summary['errors'], 1)
                self.assertEqual(len(rows), 10)
                self.assertEqual(rows[1]['eval_status'], 'evaluated_incorrect')
                self.assertEqual(rows[1]['exact_match'], '0')
                self.assertEqual(rows[1]['automatic_score'], '0')
                self.assertEqual(rows[1]['evaluation_status'], 'evaluated_incorrect')
                self.assertEqual(rows[0]['category'], questions[0]['category'])
                self.assertEqual(rows[0]['model_response'], expected)
                self.assertEqual(summary['successful_api_calls'], 9)
                self.assertEqual(summary['failed_api_calls'], 1)
                self.assertTrue(rows[-1]['ground_truth'])
                self.assertIn('mock API unavailable', rows[-1]['error'])
                raw = path.with_name(path.name.replace('_summary.json', '_raw.jsonl')).read_text().splitlines()
                self.assertEqual(len(raw), 9)
                self.assertEqual(json.loads(raw[0])['response']['choices'][0]['message']['content'], expected)


if __name__ == '__main__':
    unittest.main()
