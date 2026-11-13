"""Потоки BPE-токенов и случайные окна без перехода в проверочную часть."""

import json
from pathlib import Path

import torch

from lab02.bpe import BPE, END, tokenize_words


class BPETokenizer:
    """Нумерация символов собственного BPE из второй лабораторной."""

    def __init__(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        self.bpe = BPE(data["merges"])
        self.bpe.alphabet = set(data.get("alphabet", list("абвгдежзийклмнопрстуфхцчшщъыьэюя-") + [END]))
        self.vocabulary = sorted(self.bpe.vocabulary)
        self.ids = {token: i for i, token in enumerate(self.vocabulary)}
        self.cache: dict[str, list[int]] = {}

    def encode_words(self, words: list[str]) -> list[int]:
        result = []
        for word in words:
            if word not in self.cache:
                self.cache[word] = [self.ids[token] for token in self.bpe.encode_word(word)]
            result.extend(self.cache[word])
        return result

    def encode(self, text: str) -> list[int]:
        return self.encode_words(tokenize_words(text))

    def decode(self, ids: list[int]) -> str:
        return self.bpe.decode([self.vocabulary[index] for index in ids])


def windows(stream: torch.Tensor, starts: torch.Tensor, context: int, device: torch.device):
    """Соседние сдвиги одного окна дают входы и настоящие следующие токены."""
    full = stream[starts[:, None] + torch.arange(context + 1)]
    return full[:, :-1].to(device), full[:, 1:].to(device)
