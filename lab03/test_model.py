"""Существенные проверки: будущее скрыто, top-p сохраняет минимальное ядро."""

from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab03.generation import choose_token, first_repeat
from lab03.model import TinyLM


class ModelTests(unittest.TestCase):
    def test_future_tokens_do_not_change_past_predictions(self):
        torch.manual_seed(7)
        model = TinyLM(12, width=16, layers=2, heads=2, context=8).eval()
        with torch.no_grad():
            original = model(torch.tensor([[1, 2, 3, 4]]))
            changed = model(torch.tensor([[1, 2, 3, 9]]))
        torch.testing.assert_close(original[:, :3], changed[:, :3], atol=1e-7, rtol=1e-6)

    def test_top_p_keeps_only_highest_token_when_it_is_enough(self):
        rng = torch.Generator().manual_seed(42)
        logits = torch.tensor([1.0, 0.0, -10.0])
        self.assertEqual(choose_token(logits, 1.0, 0.7, rng), 0)

    def test_greedy_ignores_seed(self):
        for seed in (1, 2, 10):
            self.assertEqual(choose_token(torch.tensor([0.0, 4.0, -2.0]), 0, None, torch.Generator().manual_seed(seed)), 1)

    def test_overlapping_repeated_fourgrams(self):
        self.assertIsNone(first_repeat([1, 2, 3, 4, 5]))
        self.assertEqual(first_repeat([1, 1, 1, 1, 1]), 1)
        self.assertEqual(first_repeat([1, 2, 3, 4, 1, 2, 3, 4]), 4)


if __name__ == "__main__":
    unittest.main()
