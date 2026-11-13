"""Обучить собственный декодер и сравнить четыре режима генерации."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from urllib.request import urlopen

import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from lab02.bpe import tokenize_words
from lab03.data import BPETokenizer, windows
from lab03.generation import TextGenerator, first_repeat
from lab03.model import TinyLM

ARTIFACTS = HERE / "artifacts"
BPE = HERE.parent / "lab02" / "artifacts" / "bpe.json"
PROMPT = "князь андрей"


def save_json(name: str, value) -> None:
    (ARTIFACTS / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_corpus() -> str:
    """Использовать скачанный корпус или получить его с указанного в задании адреса."""
    candidates = [HERE / "corpus.txt", HERE.parent / "lab02/corpus.txt", HERE.parent / "lab01/corpus.txt"]
    for candidate in candidates:
        if candidate.is_file():
            raw = candidate.read_bytes()
            break
    else:
        address = "https://raw.githubusercontent.com/nevmenandr/word2vec-russian-novels/master/books_before/WarAndPeace.txt"
        with urlopen(address, timeout=60) as response:
            raw = response.read()
        candidates[0].write_bytes(raw)
    if hashlib.sha256(raw).hexdigest() != "5f088a66e1f3992153555fdf0361652f759697e429bd698fb2900fdf26df9836":
        raise ValueError("версия корпуса отличается от использованной во второй работе")
    return raw.decode("utf-8")


def loss_on(model, batch) -> torch.Tensor:
    x, y = batch
    logits = model(x)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))


@torch.inference_mode()
def evaluate(model, batch) -> float:
    was_training = model.training
    model.eval()
    result = float(loss_on(model, batch))
    model.train(was_training)
    return result


def write_curve(rows: list[dict]) -> None:
    with (ARTIFACTS / "train_curve.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["step", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(rows)


def plot_curve(rows: list[dict]) -> None:
    """Сохранить график тех же значений, которые записаны в CSV."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(8, 3.4))
    steps = [row["step"] for row in rows]
    axis.plot(steps, [row["train_loss"] for row in rows], label="Обучение", color="#46627a")
    axis.plot(steps, [row["val_loss"] for row in rows], label="Проверка", color="#bd793b")
    axis.set_xlabel("Шаг")
    axis.set_ylabel("Кросс-энтропия")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(ARTIFACTS / "train_curve.png", dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="200 шагов для CI")
    parser.add_argument("--bpe", type=Path, default=BPE)
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    ARTIFACTS.mkdir(exist_ok=True)
    seed = 42
    torch.manual_seed(seed)
    torch.set_num_threads(args.threads)
    requested = os.environ.get("LAB_DEVICE")
    device = torch.device(requested or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
    bpe_path = args.bpe if args.bpe.is_absolute() else HERE / args.bpe
    tokenizer = BPETokenizer(bpe_path)
    words = tokenize_words(read_corpus())
    cut = int(0.9 * len(words))
    train = torch.tensor(tokenizer.encode_words(words[:cut]), dtype=torch.long)
    val = torch.tensor(tokenizer.encode_words(words[cut:]), dtype=torch.long)
    steps, batch_size, interval = (200, 16, 25) if args.fast else (2000, 32, 100)
    context = 128
    train_rng = torch.Generator().manual_seed(seed)
    val_rng = torch.Generator().manual_seed(seed + 1)
    val_starts = torch.randint(len(val) - context, (20,), generator=val_rng)
    val_batch = windows(val, val_starts, context, device)

    def train_batch():
        starts = torch.randint(len(train) - context, (batch_size,), generator=train_rng)
        return windows(train, starts, context, device)

    model = TinyLM(len(tokenizer.vocabulary)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    generator = TextGenerator(model, tokenizer, device, seed=seed)
    sample_steps = {0, 50, 100, steps} if args.fast else {0, 100, 500, steps}
    samples = []
    started = time.perf_counter()
    curve = [{"step": 0, "train_loss": evaluate(model, train_batch()), "val_loss": evaluate(model, val_batch)}]
    samples.append({"step": 0, "text": generator.generate(PROMPT, 60, temperature=0.8)[0]})
    print(f"Параметров: {sum(p.numel() for p in model.parameters()):,}; устройство {device}; val(0)={curve[0]['val_loss']:.4f}", flush=True)
    segment_losses = []
    model.train()
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_on(model, train_batch())
        if not torch.isfinite(loss):
            raise RuntimeError("лосс стал нечисловым")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        segment_losses.append(float(loss.detach()))
        if step % interval == 0 or step == steps:
            curve.append({"step": step, "train_loss": sum(segment_losses) / len(segment_losses), "val_loss": evaluate(model, val_batch)})
            segment_losses.clear()
            write_curve(curve)
            print(f"Шаг {step}/{steps}: train={curve[-1]['train_loss']:.4f}, val={curve[-1]['val_loss']:.4f}", flush=True)
        if step in sample_steps:
            samples.append({"step": step, "text": generator.generate(PROMPT, 60, temperature=0.8)[0]})
    train_minutes = (time.perf_counter() - started) / 60
    config = {
        "vocab_size": len(tokenizer.vocabulary), "d_model": 128, "n_layers": 4, "n_heads": 4,
        "d_ff": 512, "context": context, "n_params": sum(p.numel() for p in model.parameters()),
        "steps": steps, "batch": batch_size, "lr": 1e-3, "seed": seed,
        "train_tokens": len(train), "val_tokens": len(val), "bpe_merges": len(tokenizer.bpe.merges),
        "train_minutes": train_minutes, "device": str(device), "threads": args.threads,
    }
    save_json("tiny_config.json", config)
    save_json("samples.json", {"prompt": PROMPT, "temperature": 0.8, "samples": samples})
    modes = [("greedy", 0.0, None), ("t0.7", 0.7, None), ("t1.5", 1.5, None), ("top_p0.9", 1.0, 0.9)]
    count = 40 if args.fast else 80
    runs = []
    for name, temperature, top_p in modes:
        text, ids = generator.generate(PROMPT, count, temperature=temperature, top_p=top_p)
        row = {"name": name, "temperature": temperature, "top_p": top_p,
               "text": text, "ids": ids, "n_tokens": len(ids)}
        if name == "greedy":
            row["loop_start"] = first_repeat(ids)
        runs.append(row)
    save_json("generation.json", {"model": "tiny", "prompt": PROMPT, "max_new_tokens": count, "runs": runs})
    plot_curve(curve)
    print(f"Готово: обучение {train_minutes:.2f} мин, val_loss упал на {curve[0]['val_loss'] - curve[-1]['val_loss']:.3f}", flush=True)


if __name__ == "__main__":
    main()
