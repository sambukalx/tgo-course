"""Собственное разбиение Markdown и поиск E5 в локальном Qdrant."""

import hashlib
import json
from pathlib import Path
import re

from qdrant_client import QdrantClient, models
import torch
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

EMBEDDING_MODEL = 'intfloat/multilingual-e5-small'
EMBEDDING_REVISION = '614241f622f53c4eeff9890bdc4f31cfecc418b3'
COLLECTION = 'tgo_notes'


def chunks(root: Path, max_chars: int = 1100, overlap: int = 150) -> list[dict]:
    """Сохранить путь и заголовок; большие разделы нарезать с перекрытием."""
    result = []
    for path in sorted(root.glob('lectures/*/*.md')):
        source = path.relative_to(root).as_posix()
        text = path.read_text(encoding='utf-8')
        sections = re.split(r'(?m)^(?=## )', text)
        for section in sections:
            if not section.strip():
                continue
            heading = section.splitlines()[0].lstrip('# ').strip()
            start = 0
            while start < len(section):
                end = min(len(section), start + max_chars)
                if end < len(section):
                    boundary = section.rfind('\n', start + max_chars // 2, end)
                    if boundary > start:
                        end = boundary
                piece = section[start:end].strip()
                if len(piece) > 80:
                    result.append(dict(id=len(result), source=source, section=heading, text=piece))
                if end == len(section):
                    break
                start = max(start + 1, end - overlap)
    if not result:
        raise ValueError('не найдены lectures/*/*.md')
    return result


class DocumentIndex:
    """Хранить векторы и текст в Qdrant, пересобирать при изменении корпуса."""

    def __init__(self, root: Path, state: Path) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL, revision=EMBEDDING_REVISION)
        self.model = AutoModel.from_pretrained(EMBEDDING_MODEL, revision=EMBEDDING_REVISION).eval()
        self.client = QdrantClient(path=str(state / 'qdrant'))
        documents = chunks(root)
        signature = hashlib.sha256(json.dumps([EMBEDDING_REVISION, documents], ensure_ascii=False).encode()).hexdigest()
        marker = state / 'index.json'
        old = json.loads(marker.read_text(encoding='utf-8')) if marker.is_file() else {}
        if old.get('sha256') != signature or not self.client.collection_exists(COLLECTION):
            if self.client.collection_exists(COLLECTION):
                self.client.delete_collection(COLLECTION)
            self.client.create_collection(COLLECTION, vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE))
            for start in range(0, len(documents), 16):
                part = documents[start:start + 16]
                vectors = self.encode(['passage: ' + item['section'] + '\n' + item['text'] for item in part])
                self.client.upsert(COLLECTION, points=[models.PointStruct(id=item['id'], vector=vector, payload=item)
                                                      for item, vector in zip(part, vectors)])
            marker.write_text(json.dumps(dict(sha256=signature, chunks=len(documents)), indent=2), encoding='utf-8')
        self.signature = signature
        self.size = len(documents)

    @torch.inference_mode()
    def encode(self, texts: list[str]) -> list[list[float]]:
        """Среднее по непустым токенам, L2-нормализация; тот же метод, что в lab02."""
        batch = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
        hidden = self.model(**batch).last_hidden_state
        mask = batch['attention_mask'].unsqueeze(-1)
        means = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        return F.normalize(means, dim=-1).tolist()

    def search(self, question: str, k: int) -> list[dict]:
        """Вернуть k фрагментов со счетом косинусной близости."""
        vector = self.encode(['query: ' + question])[0]
        hits = self.client.query_points(COLLECTION, query=vector, limit=k, with_payload=True).points
        return [{**hit.payload, 'score': hit.score} for hit in hits]

    def close(self) -> None:
        """Освободить блокировку файлов локального Qdrant."""
        self.client.close()
