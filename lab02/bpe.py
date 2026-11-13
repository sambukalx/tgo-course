"""Свой BPE: частые пары символов сливаются в один токен."""

from collections import Counter
from collections.abc import Iterable, Sequence
import re


END = "</w>"
WORDS = re.compile(r"[а-я]+(?:-[а-я]+)*")


def tokenize_words(text: str) -> list[str]:
    """Выделить русские слова по правилу из задания."""
    return WORDS.findall(text.lower().replace("ё", "е"))


def merge_pair(symbols: Sequence[str], pair: tuple[str, str]) -> tuple[str, ...]:
    """Слить все непересекающиеся вхождения пары слева направо."""
    result = []
    position = 0
    while position < len(symbols):
        if position + 1 < len(symbols) and (symbols[position], symbols[position + 1]) == pair:
            result.append(symbols[position] + symbols[position + 1])
            position += 2
        else:
            result.append(symbols[position])
            position += 1
    return tuple(result)


class BPE:
    """Список слияний и их ранги; никаких готовых токенизаторов."""

    def __init__(self, merges: Iterable[Sequence[str]] = ()) -> None:
        self.merges = [tuple(pair) for pair in merges]
        if any(len(pair) != 2 for pair in self.merges):
            raise ValueError("слияние должно содержать два символа")
        self.ranks = {}
        for rank, pair in enumerate(self.merges):
            self.ranks.setdefault(pair, rank)
        self.alphabet: set[str] = set()

    def fit(self, words: Iterable[str], num_merges: int, min_frequency: int = 5) -> "BPE":
        """Учить слияния на словах не реже порога, учитывая их частоты."""
        if num_merges < 1 or min_frequency < 1:
            raise ValueError("число слияний и порог должны быть положительными")
        counts = Counter(words)
        if not counts:
            raise ValueError("корпус пуст")
        self.alphabet = set("".join(counts)) | {END}
        vocabulary = [(tuple(word) + (END,), count) for word, count in counts.items() if count >= min_frequency]
        if not vocabulary:
            raise ValueError("после порога частоты не осталось слов")
        self.merges = []
        for _ in range(num_merges):
            pairs = Counter()
            for symbols, count in vocabulary:
                for pair in zip(symbols, symbols[1:]):
                    pairs[pair] += count
            if not pairs:
                break
            # При равных частотах выбираю пару по алфавиту. Результат
            # не зависит от случайного порядка хешей и повторяется.
            pair = min(pairs, key=lambda item: (-pairs[item], item))
            self.merges.append(pair)
            vocabulary = [(merge_pair(symbols, pair), count) for symbols, count in vocabulary]
        self.ranks = {}
        for rank, pair in enumerate(self.merges):
            self.ranks.setdefault(pair, rank)
        return self

    def encode_word(self, word: str) -> list[str]:
        """Применять доступное слияние с самым ранним рангом."""
        symbols = list(word) + [END]
        while len(symbols) > 1:
            available = [
                (self.ranks[pair], position)
                for position, pair in enumerate(zip(symbols, symbols[1:]))
                if pair in self.ranks
            ]
            if not available:
                break
            _, position = min(available)
            symbols[position:position + 2] = [symbols[position] + symbols[position + 1]]
        return symbols

    def encode(self, text: str) -> list[str]:
        """Разбить нормализованный текст, сохранив конец каждого слова."""
        cache = {}
        result = []
        for word in tokenize_words(text):
            if word not in cache:
                cache[word] = self.encode_word(word)
            result.extend(cache[word])
        return result

    @staticmethod
    def decode(tokens: Sequence[str]) -> str:
        """Склеить символы и заменить границы слов пробелами."""
        return "".join(tokens).replace(END, " ").strip()

    @property
    def vocabulary(self) -> set[str]:
        """Алфавит плюс все символы, созданные слияниями."""
        return self.alphabet | {left + right for left, right in self.merges}
