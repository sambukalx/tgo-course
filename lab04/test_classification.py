"""Маска ответа, единый шаблон и метрики при пропущенных предсказаниях."""

from pathlib import Path
import sys
import unittest

import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab04.classification import MODEL, MODEL_REVISION, ReviewPrompt, collate_records, metrics, parse_rating
from lab04.train import answer_loss


class ClassificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
        cls.builder = ReviewPrompt(cls.tokenizer)

    def test_prompt_is_identical_in_training_and_inference(self):
        for rating in range(1, 6):
            text = "Хорошее кафе, но слишком долго ждал заказ."
            record = self.builder.training_record(text, rating)
            expected = self.tokenizer(self.builder.render(text), add_special_tokens=False)["input_ids"]
            self.assertEqual(record["input_ids"][:-2], expected)
            self.assertTrue(all(value == -100 for value in record["labels"][:-2]))
            self.assertEqual(record["labels"][-2:], record["input_ids"][-2:])

    def test_long_review_fits_without_losing_answer(self):
        record = self.builder.training_record("очень длинный отзыв " * 1000, 1)
        self.assertLessEqual(len(record["input_ids"]), 256)
        self.assertEqual(sum(value != -100 for value in record["labels"]), 2)

    def test_selected_logits_equal_full_masked_loss(self):
        config = Qwen3Config(vocab_size=64, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                             head_dim=8, max_position_embeddings=32)
        model = Qwen3ForCausalLM(config).eval()
        records = [{"input_ids": [11, 12, 13, 3, 4], "labels": [-100, -100, -100, 3, 4]},
                   {"input_ids": [21, 22, 23, 24, 5, 4], "labels": [-100, -100, -100, -100, 5, 4]}]
        batch = collate_records(records, 0)
        with torch.no_grad():
            torch.testing.assert_close(answer_loss(model, batch), model(**batch, use_cache=False).loss)

    def test_invalid_answers_stay_errors(self):
        self.assertIsNone(parse_rating("15"))
        self.assertIsNone(parse_rating("хорошо"))
        self.assertEqual(parse_rating("Оценка: 4."), 4)
        result = metrics([1, 1, 2], [1, None, 2])
        self.assertEqual(result["invalid"], 1)
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertAlmostEqual(result["f1_macro"], (2 / 3 + 1) / 5)
        self.assertEqual(result["mae_stars"], 0)


if __name__ == "__main__":
    unittest.main()
