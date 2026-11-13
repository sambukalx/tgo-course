"""Небольшой декодер с собственным многоголовым причинным вниманием."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class CausalAttention(nn.Module):
    """QKᵀ / sqrt(d), маска будущего, softmax и взвешенная сумма V."""

    def __init__(self, width: int, heads: int, context: int) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("ширина должна делиться на число голов")
        self.heads, self.head_width = heads, width // heads
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, width, bias=False)
        self.v = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(context, context, dtype=torch.bool)), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        query = self.q(x).view(batch, length, self.heads, self.head_width).transpose(1, 2)
        key = self.k(x).view(batch, length, self.heads, self.head_width).transpose(1, 2)
        value = self.v(x).view(batch, length, self.heads, self.head_width).transpose(1, 2)
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_width)
        scores = scores.masked_fill(~self.mask[:length, :length], -torch.inf)
        weights = F.softmax(scores, dim=-1)
        mixed = (weights @ value).transpose(1, 2).contiguous().view(batch, length, width)
        return self.output(mixed)


class DecoderBlock(nn.Module):
    """Нормализация перед каждым блоком и две остаточные связи."""

    def __init__(self, width: int, heads: int, context: int) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(width)
        self.attention = CausalAttention(width, heads, context)
        self.norm_ffn = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm_attention(x))
        return x + self.ffn(self.norm_ffn(x))


class TinyLM(nn.Module):
    """Токенные и позиционные эмбеддинги, стопка блоков и выход в словарь."""

    def __init__(self, vocab_size: int, width: int = 128, layers: int = 4, heads: int = 4, context: int = 128) -> None:
        super().__init__()
        self.context = context
        self.tokens = nn.Embedding(vocab_size, width)
        self.positions = nn.Embedding(context, width)
        self.blocks = nn.ModuleList([DecoderBlock(width, heads, context) for _ in range(layers)])
        self.final_norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab_size, bias=False)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.shape[1] > self.context:
            raise ValueError("вход длиннее обученного контекста")
        positions = torch.arange(ids.shape[1], device=ids.device)
        x = self.tokens(ids) + self.positions(positions)
        for block in self.blocks:
            x = block(x)
        return self.head(self.final_norm(x))
