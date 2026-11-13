"""Проверки формул на маленьких корпусах с известным ответом."""

import math
import unittest

from lab01.bigram import BigramModel, UNK, tokenize


class BigramTests(unittest.TestCase):
    def test_tokenizer(self):
        self.assertEqual(tokenize("Ёж, кто-то и ТЕСТ123!"), ["еж", "кто-то", "и", "тест"])

    def test_counts_and_last_word(self):
        model = BigramModel().fit(["а", "б", "а", "в", "а"])
        self.assertEqual(model.prob("а", "б"), 0.5)
        self.assertEqual(model.prob("а", "в"), 0.5)
        self.assertEqual(model.word_counts["а"], 3)
        self.assertEqual(model.context_counts["а"], 2)

    def test_laplace_normalizes_known_and_unknown_contexts(self):
        model = BigramModel(alpha=0.1).fit(["а", "б", "а", "в"])
        for context in ("а", "б", "в", "новое"):
            self.assertAlmostEqual(sum(model.prob(context, w) for w in model.vocab), 1.0)
        self.assertAlmostEqual(model.prob("а", "новое"), 0.1 / (2 + 0.1 * 4))
        self.assertAlmostEqual(model.prob("новое", "а"), 1 / 4)

    def test_logprob_and_perplexity_use_transitions(self):
        model = BigramModel().fit(["а", "б", "а", "в"])
        self.assertAlmostEqual(model.logprob(["а", "б", "а"]), math.log(0.5))
        self.assertAlmostEqual(model.perplexity(["а", "б", "а"]), math.sqrt(2))
        self.assertEqual(model.logprob(["б", "в"]), -math.inf)

    def test_validation_does_not_change_vocabulary(self):
        model = BigramModel(0.01).fit(["а", "б", "а"])
        vocabulary = model.vocab
        self.assertTrue(math.isfinite(model.logprob(["совсем", "новые", "слова"])))
        self.assertEqual(model.vocab, vocabulary)
        self.assertNotIn("новые", model.word_counts)
        self.assertIn(UNK, model.vocab)

    def test_generation_is_reproducible_and_uses_observed_transitions(self):
        model = BigramModel().fit(["а", "б", "а", "в", "а"])
        text = model.generate("а", n_words=30, seed=17)
        self.assertEqual(text, model.generate("а", n_words=30, seed=17))
        words = text.split()
        self.assertEqual(len(words), 30)
        for previous, current in zip(words, words[1:]):
            self.assertGreater(model.prob(previous, current), 0)

    def test_smoothing_handles_unknown_start(self):
        model = BigramModel(0.1).fit(["а", "б", "а"])
        self.assertEqual(len(model.generate("новое", n_words=8).split()), 8)

    def test_invalid_parameters(self):
        for alpha in (-1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                BigramModel(alpha)
        with self.assertRaises(ValueError):
            BigramModel().fit(["а"])
        with self.assertRaises(ValueError):
            BigramModel().prob("а", "б")
        model = BigramModel().fit(["а", "б"])
        for temperature in (0, -1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                model.generate("а", temperature=temperature)
        with self.assertRaises(ValueError):
            model.perplexity(["а"])


if __name__ == "__main__":
    unittest.main()
