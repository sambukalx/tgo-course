"""Проверки границ: хранилище, чанки и независимый от моделей HTTP-клиент."""

import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from coursework.eval.run import text_f1
from coursework.retrieval import chunks
from coursework.storage import RequestStore
from coursework.engine import CourseAssistant


class ServiceTests(unittest.TestCase):
    def test_storage_metrics_include_failures(self):
        with tempfile.TemporaryDirectory() as folder:
            store = RequestStore(Path(folder) / 'history.sqlite')
            self.assertEqual(store.metrics()['requests'], 0)
            store.add('a', dict(seconds=1, generated_tokens=2, refused=False))
            store.add('b', dict(seconds=3, generated_tokens=4, refused=True), failed=True)
            measured = store.metrics()
            self.assertEqual(measured['requests'], 2)
            self.assertEqual(measured['generated_tokens'], 6)
            self.assertEqual(measured['failures'], 1)
            self.assertAlmostEqual(measured['p95_seconds'], 2.9)
            self.assertEqual(measured['refusal_rate'], .5)

    def test_chunks_preserve_source_and_overlap(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            target = root / 'lectures/01/notes.md'
            target.parent.mkdir(parents=True)
            target.write_text('## Раздел\n\n' + 'Пример текста. ' * 100, encoding='utf-8')
            pieces = chunks(root, max_chars=300, overlap=50)
            self.assertGreater(len(pieces), 1)
            self.assertTrue(all(item['source'] == 'lectures/01/notes.md' for item in pieces))
            self.assertTrue(all(len(item['text']) <= 300 for item in pieces))
            self.assertEqual([item['id'] for item in pieces], list(range(len(pieces))))

    def test_gold_evidence_is_real_and_has_unknown_questions(self):
        root = Path(__file__).resolve().parent.parent
        dataset = json.loads((root / 'coursework/eval/dataset.json').read_text(encoding='utf-8'))
        self.assertGreaterEqual(len(dataset), 30)
        self.assertTrue(any(row['unanswerable'] for row in dataset))
        for row in dataset:
            if not row['unanswerable']:
                self.assertIn(row['evidence'], (root / row['source']).read_text(encoding='utf-8'))

    def test_token_f1_counts_multiplicity(self):
        self.assertAlmostEqual(text_f1('a a b', 'a b b'), 2 / 3)
        self.assertEqual(text_f1('abc', 'xyz'), 0)


class OutputContractTests(unittest.TestCase):
    """Регрессии по реально встреченным ответам; модель здесь не обучается."""

    def answer(self, candidate: str, verdict: str) -> dict:
        assistant = CourseAssistant.__new__(CourseAssistant)
        assistant.lock = threading.Lock()
        assistant.index = SimpleNamespace(signature='fixture', search=lambda *args: [dict(section='Источник', text='Гауссовский шум дает MSE')])
        assistant.generate = Mock(side_effect=[(candidate, 4), (verdict, 2)])
        with patch('coursework.engine.PROMPT_VERSION', 'en-v2'):
            return assistant.ask('Какой шум приводит к MSE?', top_k=1)

    def test_citation_alone_is_not_an_answer_even_when_checker_agrees(self):
        result = self.answer('Номер использованного фрагмента: [1]', 'YES.')
        self.assertTrue(result['refused'])
        self.assertTrue(result['empty_answer'])

    def test_nonexistent_source_is_rejected_even_when_checker_agrees(self):
        result = self.answer('Гауссовский шум [5]', 'YES.')
        self.assertTrue(result['refused'])
        self.assertTrue(result['invalid_citations'])

    def test_first_verdict_label_can_be_followed_by_explanation(self):
        result = self.answer('Гауссовский шум [1]', 'YES.\nThe sources support it.')
        self.assertFalse(result['refused'])
        self.assertTrue(result['supported'])


if __name__ == '__main__':
    unittest.main()
