"""Общий шаблон задачи, данные, разбор ответа и метрики пяти классов."""

from collections import Counter
from contextlib import AbstractContextManager
import os
from pathlib import Path
import random
import re
import threading

from datasets import load_dataset, load_from_disk
import psutil
import torch


MODEL = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
DATASET = "mteb/GeoreviewClassification"
DATASET_REVISION = "fb29d6137c54550329411023aee45a8ff29f0c62"
SYSTEM = (
    "Определи оценку автора отзыва об организации. "
    "Шкала: 1 - очень плохо, 2 - плохо, 3 - средне, 4 - хорошо, 5 - отлично. "
    "Ответь только одной цифрой от 1 до 5."
)
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def choose_device() -> torch.device:
    """Использовать CUDA, Intel XPU, MPS или CPU без привязки к одной машине."""
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


def load_data(path: str | Path | None = None):
    """Прочитать локальный save_to_disk или открытый набор из HF."""
    selected = path or os.environ.get("LAB_DATA")
    return load_from_disk(str(selected)) if selected else load_dataset(DATASET, revision=DATASET_REVISION)


def select_balanced(split, per_class: int = 1000, seed: int = 42):
    """Случайно отобрать одинаковое число примеров каждого класса без повторов."""
    indices = list(range(len(split)))
    random.Random(seed).shuffle(indices)
    counts, selected = Counter(), []
    labels = split["label"]
    for index in indices:
        label = int(labels[index])
        if not 0 <= label <= 4:
            raise ValueError("ожидались исходные метки 0-4")
        if counts[label] < per_class:
            selected.append(index)
            counts[label] += 1
        if len(selected) == 5 * per_class:
            break
    if len(selected) != 5 * per_class:
        raise ValueError("в выборке недостаточно примеров одного из классов")
    return split.select(selected)


def few_shot_examples(validation, seed: int = 42) -> list[dict]:
    """Взять по одному примеру из validation, не используя тестовые ответы."""
    sample = select_balanced(validation, per_class=1, seed=seed)
    return sorted([dict(row) for row in sample], key=lambda row: row["label"])


class ReviewPrompt:
    """Один шаблон для обучения, zero-shot, few-shot и адаптера."""

    def __init__(self, tokenizer, max_len: int = 256) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len
        empty = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": self._user("")}]
        encoded = tokenizer.apply_chat_template(empty, tokenize=True, return_dict=True,
                                                add_generation_prompt=True, enable_thinking=False)
        overhead = len(encoded["input_ids"])
        self.review_budget = max_len - overhead - 2 - 4
        if self.review_budget < 1:
            raise ValueError("системная инструкция не помещается в контекст обучения")

    @staticmethod
    def _user(text: str) -> str:
        return f"Отзыв:\n{text}\nОценка:"

    def shorten(self, text: str) -> str:
        """Сохранить начало отзыва и границу Unicode-символа при усечении."""
        encoded = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        if len(encoded["input_ids"]) <= self.review_budget:
            return text
        end = encoded["offset_mapping"][self.review_budget - 1][1]
        return text[:end]

    def messages(self, text: str, examples: list[dict] | None = None) -> list[dict]:
        result = [{"role": "system", "content": SYSTEM}]
        for example in examples or []:
            result.extend([
                {"role": "user", "content": self._user(self.shorten(example["text"]))},
                {"role": "assistant", "content": str(int(example["label"]) + 1)},
            ])
        result.append({"role": "user", "content": self._user(self.shorten(text))})
        return result

    def render(self, text: str, examples: list[dict] | None = None) -> str:
        return self.tokenizer.apply_chat_template(self.messages(text, examples), tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=False)

    def training_record(self, text: str, rating: int) -> dict:
        """Метки промпта -100; два оцениваемых токена - цифра и конец ответа."""
        prefix = self.tokenizer(self.render(text), add_special_tokens=False)["input_ids"]
        answer = self.tokenizer(str(rating), add_special_tokens=False)["input_ids"] + [self.tokenizer.eos_token_id]
        if len(answer) != 2 or not 1 <= rating <= 5:
            raise ValueError("ожидались цифра 1-5 и токен конца ответа")
        if len(prefix) + len(answer) > self.max_len:
            raise ValueError("пример не поместился в max_len")
        return {"input_ids": prefix + answer, "labels": [-100] * len(prefix) + answer}


def collate_records(records: list[dict], pad_token: int) -> dict[str, torch.Tensor]:
    """Левый padding выравнивает ответы в конце каждого примера."""
    length = max(len(record["input_ids"]) for record in records)
    inputs = torch.full((len(records), length), pad_token, dtype=torch.long)
    labels = torch.full_like(inputs, -100)
    mask = torch.zeros_like(inputs)
    for i, record in enumerate(records):
        n = len(record["input_ids"])
        inputs[i, -n:] = torch.tensor(record["input_ids"])
        labels[i, -n:] = torch.tensor(record["labels"])
        mask[i, -n:] = 1
    positions = (mask.cumsum(dim=-1) - 1).clamp_min(0)
    return {"input_ids": inputs, "labels": labels, "attention_mask": mask, "position_ids": positions}


def parse_rating(text: str) -> int | None:
    """Принять отдельную цифру шкалы; 15 и ответ без цифры считаются ошибкой."""
    match = re.search(r"(?<!\d)([1-5])(?!\d)", text)
    return int(match.group(1)) if match else None


def metrics(labels: list[int], predictions: list[int | None]) -> dict:
    """Посчитать макро-F1 ровно по пяти классам, включая отсутствующие."""
    f1 = {}
    for rating in range(1, 6):
        tp = sum(y == rating and p == rating for y, p in zip(labels, predictions))
        fp = sum(y != rating and p == rating for y, p in zip(labels, predictions))
        fn = sum(y == rating and p != rating for y, p in zip(labels, predictions))
        denominator = 2 * tp + fp + fn
        f1[str(rating)] = 2 * tp / denominator if denominator else 0.0
    valid = [(y, p) for y, p in zip(labels, predictions) if p is not None]
    return {
        "accuracy": sum(y == p for y, p in zip(labels, predictions)) / len(labels),
        "f1_macro": sum(f1.values()) / 5, "f1_per_class": f1,
        "mae_stars": sum(abs(y - p) for y, p in valid) / len(valid) if valid else None,
        "invalid": len(labels) - len(valid), "invalid_fraction": 1 - len(valid) / len(labels),
    }


class PeakMemory(AbstractContextManager):
    """Пик RSS процесса с опросом раз в 50 мс; на CUDA - счетчик PyTorch."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.peak = 0
        self.stop = threading.Event()
        self.process = psutil.Process()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.stop.is_set():
            self.peak = max(self.peak, self.process.memory_info().rss)
            self.stop.wait(0.05)

    def __enter__(self):
        if self.device.type in ("cuda", "xpu"):
            getattr(torch, self.device.type).reset_peak_memory_stats(self.device)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.stop.set()
        self.thread.join()
        return False

    @property
    def gb(self) -> float:
        peak = getattr(torch, self.device.type).max_memory_allocated(self.device) if self.device.type in ("cuda", "xpu") else self.peak
        return peak / 1024**3
