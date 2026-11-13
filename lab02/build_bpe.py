"""Отдельно выполнить первую часть лабораторной 2 без ML-библиотек."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab02.bpe import BPE, tokenize_words


HERE = Path(__file__).resolve().parent


def build(text: str, fast: bool = False) -> dict:
    """Обучить BPE и посчитать статистику на всем корпусе."""
    words = tokenize_words(text)
    counts = Counter(words)
    required = 200 if fast else 1500
    tokenizer = BPE().fit(words, num_merges=required, min_frequency=5)
    if len(tokenizer.merges) != required:
        raise ValueError("корпус слишком мал для требуемого числа слияний")
    chars = sum(len(word) * count for word, count in counts.items())
    tokens = sum(len(tokenizer.encode_word(word)) * count for word, count in counts.items())
    return {
        "num_merges": len(tokenizer.merges),
        "vocab_size": len(tokenizer.vocabulary),
        "alphabet": sorted(tokenizer.alphabet),
        "min_frequency": 5,
        "merges": tokenizer.merges,
        "stats": {
            "chars": chars, "words": len(words), "tokens_bpe": tokens,
            "tokens_per_word": tokens / len(words),
            "compression_vs_chars": chars / tokens,
        },
        "examples": {word: tokenizer.encode_word(word) for word in ("наполеон", "сказала", "неудовольствие")},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--output", type=Path, default=HERE / "artifacts/bpe.json")
    args = parser.parse_args()
    corpus = args.corpus or HERE / "corpus.txt"
    if not corpus.is_file() and args.corpus is None:
        corpus = HERE.parent / "lab01/corpus.txt"
    if not corpus.is_file():
        parser.error("нужен corpus.txt или путь к корпусу через --corpus")
    started = time.perf_counter()
    result = build(corpus.read_text(encoding="utf-8"), fast=args.fast)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Слияний:", result["num_merges"], "| Словарь:", result["vocab_size"])
    print("Токенов на слово:", round(result["stats"]["tokens_per_word"], 3))
    print("Сжатие относительно букв:", round(result["stats"]["compression_vs_chars"], 3))
    print("Примеры:", result["examples"])
    print(f"Время: {time.perf_counter() - started:.1f} с | Сохранено: {args.output}")


if __name__ == "__main__":
    main()
