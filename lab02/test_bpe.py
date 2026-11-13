"""Проверки частот, непересекающихся слияний и обратного преобразования."""

import unittest

from bpe import BPE, END, merge_pair, tokenize_words


class BPETests(unittest.TestCase):
    def test_words(self):
        self.assertEqual(tokenize_words("Ёлка, кто-то! English 123"), ["елка", "кто-то"])

    def test_frequency_is_weighted(self):
        tokenizer = BPE().fit(["аб"] * 10 + ["вг", "вд", "ве"], num_merges=1, min_frequency=1)
        self.assertEqual(tokenizer.merges[0], ("а", "б"))

    def test_overlaps_are_not_merged_twice(self):
        self.assertEqual(merge_pair(["а", "а", "а", "а", "а", END], ("а", "а")), ("аа", "аа", "а", END))

    def test_end_of_word_is_a_symbol(self):
        tokenizer = BPE().fit(["а"] * 5, num_merges=1)
        self.assertEqual(tokenizer.encode_word("а"), ["а</w>"])

    def test_ranks_and_decode(self):
        tokenizer = BPE([("а", "б"), ("аб", "в"), ("абв", END)])
        self.assertEqual(tokenizer.encode_word("абв"), ["абв</w>"])
        self.assertEqual(tokenizer.decode(tokenizer.encode("АБВ абв")), "абв абв")

    def test_ties_and_repeated_fit(self):
        tokenizer = BPE().fit(["вг", "аб"], num_merges=1, min_frequency=1)
        self.assertEqual(tokenizer.merges, [("а", "б")])
        tokenizer.fit(["д"] * 5, num_merges=1)
        self.assertEqual(tokenizer.merges, [("д", END)])

    def test_rare_words_are_excluded_from_learning(self):
        tokenizer = BPE().fit(["аб"] * 5 + ["вг"], num_merges=1)
        self.assertEqual(tokenizer.encode_word("вг"), ["в", "г", END])
        self.assertEqual(tokenizer.decode(tokenizer.encode("аб вг")), "аб вг")

    def test_invalid_data(self):
        with self.assertRaises(ValueError):
            BPE().fit([], num_merges=1)
        with self.assertRaises(ValueError):
            BPE().fit(["а"], num_merges=1)


if __name__ == "__main__":
    unittest.main()
