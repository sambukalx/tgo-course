"""Сравнить zero-shot, пять примеров в промпте и обученный LoRA-адаптер."""

import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import sys
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from lab04.classification import DATASET, DATASET_REVISION, MODEL, MODEL_REVISION, PeakMemory, ReviewPrompt, choose_device, few_shot_examples, load_data, metrics, parse_rating

ARTIFACTS = HERE / "artifacts"
ADAPTER = ARTIFACTS / "adapter"


@torch.inference_mode()
def evaluate_mode(model, tokenizer, builder, rows, device, examples=None, batch_size: int = 8):
    """Оценить одну конфигурацию на неизменном порядке тестовых примеров."""
    predictions, raw_answers, prompt_lengths = [], [], []
    started = time.perf_counter()
    with PeakMemory(device) as memory:
        for start in range(0, len(rows), batch_size):
            part = rows[start:start + batch_size]
            prompts = [builder.render(row["text"], examples) for row in part]
            batch = tokenizer(prompts, add_special_tokens=False, padding=True, return_tensors="pt").to(device)
            prompt_lengths.extend(batch["attention_mask"].sum(dim=-1).tolist())
            generated = model.generate(**batch, do_sample=False, max_new_tokens=3, use_cache=True,
                                       pad_token_id=tokenizer.eos_token_id,
                                       temperature=None, top_p=None, top_k=None)
            new_ids = generated[:, batch["input_ids"].shape[1]:]
            answers = tokenizer.batch_decode(new_ids, skip_special_tokens=True)
            raw_answers.extend(answers)
            predictions.extend(parse_rating(answer) for answer in answers)
            if start % 80 == 0 or start + batch_size >= len(rows):
                print(f"Оценка: {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    labels = [int(row["label"]) + 1 for row in rows]
    result = metrics(labels, predictions)
    result["sec_per_example"] = (time.perf_counter() - started) / len(rows)
    result["peak_memory_gb"] = memory.gb
    result["prompt_tokens"] = sum(prompt_lengths) / len(prompt_lengths)
    return result, predictions, raw_answers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="24 тестовых примера")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--adapter", type=Path, default=ADAPTER)
    parser.add_argument("--output", type=Path, default=ARTIFACTS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    ARTIFACTS.mkdir(exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    torch.set_num_threads(args.threads)
    device = choose_device()
    dataset = load_data(args.data)
    count = 24 if args.fast else 1000
    rows = [dict(row) for row in dataset["test"].select(range(count))]
    examples = few_shot_examples(dataset["validation"])
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    builder = ReviewPrompt(tokenizer)
    dtype = torch.bfloat16 if device.type in ("cuda", "xpu") and getattr(torch, device.type).is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=MODEL_REVISION,
                                                dtype=dtype, attn_implementation="sdpa").to(device).eval()
    labels = [int(row["label"]) + 1 for row in rows]
    counts = Counter(labels)
    result = {"model": MODEL, "model_revision": MODEL_REVISION, "dataset": DATASET, "dataset_revision": DATASET_REVISION,
              "n_test": count, "fast": args.fast, "class_counts": {str(c): counts[c] for c in range(1, 6)},
              "device": str(device), "dtype": str(dtype), "seed": 42, "review_budget": builder.review_budget,
              "batch_size": args.batch_size, "threads": args.threads, "max_new_tokens": 3, "modes": {}}
    predictions = [{"idx": i, "label": labels[i], "zero_shot": "", "few_shot": "", "adapter": "",
                    "raw_zero_shot": "", "raw_few_shot": "", "raw_adapter": ""} for i in range(count)]
    modes = [("zero_shot", None), ("few_shot", examples)]
    if (args.adapter / "adapter_model.safetensors").is_file():
        modes.append(("adapter", None))
    else:
        result["adapter_missing"] = True
        print("Адаптер не найден: будут только два режима, работа пока не завершена.", flush=True)
    for name, shots in modes:
        if name == "adapter":
            model = PeftModel.from_pretrained(model, args.adapter).to(device).eval()
        print("Режим:", name, flush=True)
        measured, ratings, raw = evaluate_mode(model, tokenizer, builder, rows, device, shots, args.batch_size)
        result["modes"][name] = measured
        for i, (rating, answer) in enumerate(zip(ratings, raw)):
            predictions[i][name] = rating if rating is not None else ""
            predictions[i]["raw_" + name] = answer
        (args.output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
        with (args.output / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
            writer.writeheader()
            writer.writerows(predictions)
        print(f"{name}: macro-F1={measured['f1_macro']:.4f}, accuracy={measured['accuracy']:.4f}, invalid={measured['invalid']}", flush=True)


if __name__ == "__main__":
    main()
