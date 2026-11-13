"""Своя генерация: температура, минимальное ядро top-p и жадный выбор."""

import math

import torch
from torch.nn import functional as F


def choose_token(logits: torch.Tensor, temperature: float, top_p: float | None, rng: torch.Generator) -> int:
    """Выбрать токен из логитов последней позиции."""
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("температура должна быть конечной и неотрицательной")
    if top_p is not None and (not math.isfinite(top_p) or not 0 < top_p <= 1):
        raise ValueError("top_p должно находиться в (0, 1]")
    if temperature == 0:
        return int(logits.argmax())
    probabilities = F.softmax(logits.float().cpu() / temperature, dim=-1)
    if top_p is None:
        return int(torch.multinomial(probabilities, 1, generator=rng).item())
    sorted_probabilities, indices = probabilities.sort(descending=True)
    cumulative = sorted_probabilities.cumsum(dim=0)
    # Оставляем ровно минимальное число токенов, достигающее порога.
    count = min(int(torch.searchsorted(cumulative, top_p)) + 1, len(indices))
    selected = torch.multinomial(sorted_probabilities[:count], 1, generator=rng)
    return int(indices[selected].item())


def first_repeat(ids: list[int], size: int = 4) -> int | None:
    """Позиция первого повторившегося n-грамма среди новых токенов."""
    seen = set()
    for start in range(len(ids) - size + 1):
        part = tuple(ids[start:start + size])
        if part in seen:
            return start
        seen.add(part)
    return None


class TextGenerator:
    """Обертка хранит модель и токенизатор, а generate принимает только текст."""

    def __init__(self, model, tokenizer, device: torch.device, seed: int = 42) -> None:
        self.model, self.tokenizer, self.device, self.seed = model, tokenizer, device, seed

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: int, temperature: float = 1.0, top_p: float | None = None):
        """Вернуть полный текст с затравкой и отдельно номера новых токенов."""
        prompt_ids = self.tokenizer.encode(prompt)
        if not prompt_ids or max_new_tokens < 0:
            raise ValueError("нужны непустая затравка и неотрицательная длина")
        was_training = self.model.training
        self.model.eval()
        rng = torch.Generator().manual_seed(self.seed)
        all_ids, new_ids = prompt_ids.copy(), []
        try:
            for _ in range(max_new_tokens):
                context = torch.tensor([all_ids[-self.model.context:]], device=self.device)
                logits = self.model(context)[0, -1]
                token = choose_token(logits, temperature, top_p, rng)
                all_ids.append(token)
                new_ids.append(token)
        finally:
            self.model.train(was_training)
        return self.tokenizer.decode(all_ids), new_ids
