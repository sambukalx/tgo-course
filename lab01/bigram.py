"""Биграммная модель: счетчики, сглаживание и выбор следующего слова."""

from collections import Counter, defaultdict
from collections.abc import Sequence
import math
import random
import re


WORD_RE = re.compile(r"[а-я]+(?:-[а-я]+)*")
UNK = "<unk>"


def tokenize(text: str) -> list[str]:
    """Оставить русские слова, сохранив дефисы внутри слова."""
    return WORD_RE.findall(text.lower().replace("ё", "е"))


class BigramModel:
    """P(следующее слово | предыдущее слово) по обучающему тексту.

    В словаре всегда есть один дополнительный исход <unk>. Незнакомые
    слова заменяются им, поэтому проверка не расширяет обученный словарь.
    """

    def __init__(self, alpha: float = 0.0) -> None:
        if not math.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha должно быть конечным и неотрицательным")
        self.alpha = alpha
        self.word_counts: Counter[str] = Counter()
        self.next_counts: dict[str, Counter[str]] = {}
        self.context_counts: Counter[str] = Counter()
        self.vocab: tuple[str, ...] = ()
        self._vocab_set: set[str] = set()

    def fit(self, words: Sequence[str]) -> "BigramModel":
        """Посчитать слова и пары только по переданной обучающей части."""
        if len(words) < 2:
            raise ValueError("для обучения нужны хотя бы два слова")
        if UNK in words:
            raise ValueError("<unk> зарезервирован для неизвестного слова")
        self.word_counts = Counter(words)
        rows: defaultdict[str, Counter[str]] = defaultdict(Counter)
        self.context_counts = Counter()
        for previous, current in zip(words, words[1:]):
            rows[previous][current] += 1
            self.context_counts[previous] += 1
        self.next_counts = dict(rows)
        self.vocab = tuple(sorted(self.word_counts)) + (UNK,)
        self._vocab_set = set(self.vocab)
        return self

    def _known(self, word: str) -> str:
        if not self.vocab:
            raise ValueError("сначала вызовите fit")
        return word if word in self._vocab_set else UNK

    def prob(self, previous: str, current: str) -> float:
        """Вероятность перехода; при alpha=0 незнакомая пара дает ноль."""
        previous, current = self._known(previous), self._known(current)
        row = self.next_counts.get(previous, {})
        # У последнего слова корпуса нет следующего слова. В знаменателе
        # число исходящих переходов, иначе сумма строки была бы меньше 1.
        denominator = self.context_counts[previous] + self.alpha * len(self.vocab)
        if denominator == 0:
            return 0.0
        return (row.get(current, 0) + self.alpha) / denominator

    def logprob(self, text: str | Sequence[str]) -> float:
        """Сумма натуральных логарифмов по переходам, без первого слова."""
        words = tokenize(text) if isinstance(text, str) else text
        terms = []
        for previous, current in zip(words, words[1:]):
            probability = self.prob(previous, current)
            if probability == 0:
                return -math.inf
            terms.append(math.log(probability))
        return math.fsum(terms)

    def perplexity(self, words: Sequence[str]) -> float:
        """exp(средний минус log P) по N=len(words)-1 переходам."""
        if len(words) < 2:
            raise ValueError("для перплексии нужны хотя бы два слова")
        return math.exp(-self.logprob(words) / (len(words) - 1))

    def generate(
        self, start: str, n_words: int = 30, temperature: float = 1.0, seed: int = 42
    ) -> str:
        """Выбирать слова по P^(1/T); длина включает стартовое слово."""
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("температура должна быть конечной и больше нуля")
        if n_words < 1:
            raise ValueError("длина должна быть положительной")
        start = self._known(start)
        rng = random.Random(seed)
        result = [start]
        for _ in range(n_words - 1):
            row = self.next_counts.get(result[-1], {})
            candidates = self.vocab if self.alpha else tuple(row)
            if not candidates:
                raise ValueError("нет продолжений: задайте alpha > 0")
            # Общий знаменатель сократится при перенормировке. Работа
            # в логарифмах не дает степеням переполниться при малой T.
            scores = [math.log(row.get(word, 0) + self.alpha) for word in candidates]
            largest = max(scores)
            weights = [math.exp((score - largest) / temperature) for score in scores]
            result.append(rng.choices(candidates, weights=weights, k=1)[0])
        return " ".join(result)
