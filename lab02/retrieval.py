"""Импортируемый поиск search(query, k) по абзацам полного прогона."""

from functools import lru_cache
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

from lab02.models import E5, choose_device, e5_vectors


class ParagraphIndex:
    """Один раз посчитать E5-векторы, затем обслуживать новые запросы."""

    def __init__(self, paragraphs: list[dict]) -> None:
        self.paragraphs = paragraphs
        self.device = choose_device()
        self.tokenizer = AutoTokenizer.from_pretrained(E5)
        self.model = AutoModel.from_pretrained(E5).to(self.device).eval()
        vectors = e5_vectors(["passage: " + p["text"] for p in paragraphs], self.tokenizer, self.model, self.device)
        self.vectors = F.normalize(vectors, dim=-1)

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Вернуть идентификатор, текст и косинусную близость каждого результата."""
        if not query.strip() or not 1 <= k <= len(self.paragraphs):
            raise ValueError("нужны непустой запрос и допустимое k")
        vector = e5_vectors(["query: " + query], self.tokenizer, self.model, self.device)
        scores = (F.normalize(vector, dim=-1) @ self.vectors.T)[0]
        values, ids = torch.topk(scores, k=k)
        return [{**self.paragraphs[index], "score": float(score)} for index, score in zip(ids.tolist(), values.tolist())]


@lru_cache(maxsize=1)
def _default_index() -> ParagraphIndex:
    path = Path(__file__).resolve().parent / "artifacts/paragraphs.json"
    return ParagraphIndex(json.loads(path.read_text(encoding="utf-8")))


def search(query: str, k: int = 5) -> list[dict]:
    """Найти абзацы по новому текстовому запросу; модель загружается при первом вызове."""
    return _default_index().search(query, k)
