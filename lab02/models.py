"""Токенизация и три способа получить вектор текста."""

from functools import lru_cache
import os

import torch
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer


QWEN = "Qwen/Qwen3-0.6B-Base"
E5 = "intfloat/multilingual-e5-small"


def choose_device() -> torch.device:
    """Выбрать доступное устройство; LAB_DEVICE позволяет задать его явно."""
    requested = os.environ.get("LAB_DEVICE")
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@lru_cache(maxsize=1)
def qwen_tokenizer():
    """Загрузить один общий токенизатор для подсчета и эмбеддингов."""
    tokenizer = AutoTokenizer.from_pretrained(QWEN)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def count_tokens(text: str) -> int:
    """Посчитать токены Qwen без служебных токенов и шаблона чата."""
    return len(qwen_tokenizer()(text, add_special_tokens=False)["input_ids"])


def load_qwen(device: torch.device):
    """Загрузить основу Qwen: для векторов и внимания выход в словарь не нужен."""
    model = AutoModel.from_pretrained(QWEN, dtype=torch.float32, attn_implementation="eager")
    return model.to(device).eval()


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Усреднить только настоящие токены, исключив padding."""
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)


@torch.inference_mode()
def qwen_vectors(texts: list[str], tokenizer, model, device: torch.device, batch_size: int = 4):
    """Вернуть средние входные и контекстные векторы одних и тех же токенов."""
    static, contextual = [], []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(texts[start:start + batch_size], add_special_tokens=False,
                          padding=True, return_tensors="pt").to(device)
        embedded = model.get_input_embeddings()(batch["input_ids"])
        static.append(mean_pool(embedded, batch["attention_mask"]).cpu())
        output = model(**batch, output_hidden_states=True, output_attentions=False, use_cache=False)
        contextual.append(mean_pool(output.hidden_states[-1], batch["attention_mask"]).cpu())
        if start % 40 == 0 or start + batch_size >= len(texts):
            print(f"Qwen: {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
    return torch.cat(static), torch.cat(contextual)


@torch.inference_mode()
def e5_vectors(texts: list[str], tokenizer, model, device: torch.device, batch_size: int = 16) -> torch.Tensor:
    """Получить E5-векторы; префиксы query/passage добавляются вызывающим кодом."""
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(texts[start:start + batch_size], padding=True, truncation=True,
                          max_length=512, return_tensors="pt").to(device)
        hidden = model(**batch).last_hidden_state
        vectors.append(mean_pool(hidden, batch["attention_mask"]).cpu())
    return torch.cat(vectors)


def search(query_vectors: torch.Tensor, paragraph_vectors: torch.Tensor, k: int = 5) -> list[list[int]]:
    """Найти k абзацев по убыванию косинусной близости для каждого запроса."""
    similarities = F.normalize(query_vectors, dim=-1) @ F.normalize(paragraph_vectors, dim=-1).T
    return similarities.topk(k=k, dim=-1).indices.tolist()
