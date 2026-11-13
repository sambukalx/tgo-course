"""Одна эпоха LoRA по сбалансированным 5000 отзывам, без обучения на промпте."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import torch
from torch.nn import functional as F
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from lab04.classification import DATASET, MODEL, MODEL_REVISION, TARGET_MODULES, PeakMemory, ReviewPrompt, choose_device, collate_records, load_data, select_balanced


def answer_loss(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Вычислить тот же masked CE, выводя в словарь только две нужные позиции."""
    length = batch["input_ids"].shape[1]
    # Логит на позиции перед цифрой предсказывает цифру; следующий - EOS.
    positions = torch.tensor([length - 3, length - 2], device=batch["input_ids"].device)
    inputs = {name: tensor for name, tensor in batch.items() if name != "labels"}
    logits = model(**inputs, use_cache=False, logits_to_keep=positions).logits
    targets = batch["labels"][:, -2:]
    if (targets < 0).any():
        raise ValueError("ответы должны быть выровнены справа")
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1))


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def write_log(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["step", "loss"])
        writer.writeheader()
        writer.writerows(rows)


def train_lora(dataset=None, output_directory: Path = HERE / "artifacts", per_class: int = 1000,
               micro_batch: int = 4, threads: int = 8, seed: int = 42, resume: bool = False) -> dict:
    """Обучить все семь LoRA-модулей, сохранить адаптер, журнал и параметры."""
    if micro_batch < 1 or 8 % micro_batch:
        raise ValueError("микробатч должен делить эффективный батч 8")
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(threads, os.cpu_count() or 1))
    torch.manual_seed(seed)
    random.seed(seed)
    device = choose_device()
    dataset = dataset if dataset is not None else load_data()
    selected = select_balanced(dataset["train"], per_class=per_class, seed=seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    prompt = ReviewPrompt(tokenizer)
    records = [prompt.training_record(row["text"], int(row["label"]) + 1) for row in selected]
    data_sha = hashlib.sha256(json.dumps(records, separators=(",", ":")).encode()).hexdigest()
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    use_amp = device.type in ("cuda", "xpu")
    accelerator = getattr(torch, device.type) if use_amp else None
    amp_dtype = torch.bfloat16 if use_amp and accelerator.is_bf16_supported() else torch.float16
    base_dtype = amp_dtype if use_amp else torch.float32
    base = AutoModelForCausalLM.from_pretrained(MODEL, revision=MODEL_REVISION, dtype=base_dtype, attn_implementation="sdpa")
    base.config.use_cache = False
    model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                                           target_modules=TARGET_MODULES, bias="none", task_type="CAUSAL_LM"))
    model.to(device).train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    n_trainable = sum(parameter.numel() for parameter in parameters)
    n_total = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=2e-4)
    scaler = torch.amp.GradScaler(device.type if use_amp else "cuda", enabled=use_amp and amp_dtype == torch.float16)
    checkpoint = output_directory / "training_state.pt"
    completed, earlier_seconds, log = 0, 0.0, []
    if resume:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["data_sha"] != data_sha or saved["device"] != str(device):
            raise ValueError("контрольная точка относится к другим данным или устройству")
        set_peft_model_state_dict(model, saved["adapter"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        torch.set_rng_state(saved["rng"])
        if use_amp:
            accelerator.set_rng_state_all(saved["device_rng"])
        completed, earlier_seconds, log = saved["step"], saved["elapsed_seconds"], saved["log"]
    steps = math.ceil(len(records) / 8)
    print(f"Обучение: {len(records)} примеров, {steps} шагов, микробатч {micro_batch}, устройство {device}", flush=True)
    print(f"Обучаемых параметров: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.3f}%)", flush=True)
    print(f"Маска первого примера: {len(records[0]['labels'])-2} токенов промпта, 2 токена ответа", flush=True)
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    with PeakMemory(device) as memory:
        for step, start in enumerate(range(0, len(order), 8), 1):
            if step <= completed:
                continue
            indices = order[start:start + 8]
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for offset in range(0, len(indices), micro_batch):
                part = indices[offset:offset + micro_batch]
                batch = {name: tensor.to(device) for name, tensor in collate_records([records[i] for i in part], tokenizer.pad_token_id).items()}
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    loss = answer_loss(model, batch)
                if not torch.isfinite(loss):
                    raise RuntimeError("лосс стал нечисловым")
                scaler.scale(loss * len(part) / len(indices)).backward()
                step_loss += float(loss.detach()) * len(part) / len(indices)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            scaler.step(optimizer)
            scaler.update()
            log.append({"step": step, "loss": step_loss})
            elapsed = earlier_seconds + time.perf_counter() - started
            if step == 1 or step % 25 == 0 or step == steps:
                write_log(output_directory / "train_log.csv", log)
                print(f"Шаг {step}/{steps}: loss={step_loss:.4f}, время={elapsed/60:.1f} мин", flush=True)
            if step == 1 or step % 5 == 0 or step == steps:
                write_json(output_directory / "progress.json", {"step": step, "steps": steps, "loss": step_loss,
                                                               "elapsed_seconds": elapsed, "status": "training"})
            if step % 100 == 0:
                state = {"step": step, "data_sha": data_sha, "device": str(device), "log": log,
                         "adapter": get_peft_model_state_dict(model), "optimizer": optimizer.state_dict(),
                         "scaler": scaler.state_dict(), "rng": torch.get_rng_state(), "elapsed_seconds": elapsed}
                if use_amp:
                    state["device_rng"] = accelerator.get_rng_state_all()
                temporary = checkpoint.with_suffix(".tmp.pt")
                torch.save(state, temporary)
                temporary.replace(checkpoint)
        if use_amp:
            accelerator.synchronize(device)
    train_minutes = (earlier_seconds + time.perf_counter() - started) / 60
    model.eval().save_pretrained(output_directory / "adapter", safe_serialization=True, save_embedding_layers=False)
    config = {
        "model": MODEL, "model_revision": MODEL_REVISION, "dataset": DATASET, "lora_r": 8, "lora_alpha": 16,
        "lora_dropout": 0.05, "target_modules": TARGET_MODULES, "trainable_params": n_trainable,
        "total_params": n_total, "trainable_pct": 100*n_trainable/n_total,
        "train_examples": len(records), "epochs": 1, "steps": steps, "batch": 8,
        "micro_batch": micro_batch, "lr": 2e-4, "max_len": 256, "review_budget": prompt.review_budget,
        "train_minutes": train_minutes, "peak_memory_gb": memory.gb, "device": str(device),
        "dtype": str(next(model.parameters()).dtype), "amp_dtype": str(amp_dtype) if use_amp else None,
        "seed": seed, "data_sha256": data_sha,
        "started_at_utc": started_utc, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_directory / "train_config.json", config)
    write_json(output_directory / "progress.json", {"step": steps, "steps": steps, "loss": log[-1]["loss"],
                                                   "elapsed_seconds": train_minutes * 60, "status": "completed"})
    print(f"Адаптер сохранен; обучение {train_minutes:.2f} мин, пик памяти {memory.gb:.2f} ГБ", flush=True)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, default=HERE / "artifacts")
    parser.add_argument("--per-class", type=int, default=1000)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train_lora(load_data(args.data), args.output, args.per_class, args.micro_batch, args.threads, resume=args.resume)


if __name__ == "__main__":
    main()
