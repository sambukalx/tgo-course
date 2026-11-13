"""Связь местоимения с существительным и доля внимания на инструкцию."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


SENTENCE = "Врач осмотрел пациента и сказал, что его анализы готовы."
INSTRUCTION = "Отвечай только по-русски и в конце каждого ответа ставь слово КОНЕЦ."


def word_positions(offsets: list, text: str, word: str) -> list[int]:
    """Найти все токены, пересекающие символьный интервал слова."""
    start = text.index(word)
    end = start + len(word)
    return [i for i, (left, right) in enumerate(offsets) if left < end and right > start]


@torch.inference_mode()
def pronoun_attention(tokenizer, model, device: torch.device, output_directory: Path) -> dict:
    """Выбрать голову по сумме весов на всем слове «пациента»."""
    encoded = tokenizer(SENTENCE, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    pronoun = word_positions(offsets, SENTENCE, "его")
    antecedent = word_positions(offsets, SENTENCE, "пациента")
    ids = torch.tensor([encoded["input_ids"]], device=device)
    output = model(input_ids=ids, output_attentions=True, use_cache=False)
    rows = torch.stack([layer[0, :, pronoun[-1], :].float().cpu() for layer in output.attentions])
    shares = rows[:, :, antecedent].sum(dim=-1)
    layer, head = divmod(int(shares.argmax()), shares.shape[1])
    weights = rows[layer, head].tolist()
    labels = [SENTENCE[left:right] for left, right in offsets]
    result = {
        "sentence": SENTENCE, "tokens": labels,
        "pronoun_pos": pronoun, "antecedent_pos": antecedent,
        "best_layer": layer, "best_head": head, "weights": weights,
        "share_on_antecedent": float(shares[layer, head]),
    }
    figure, axis = plt.subplots(figsize=(11, 3.8))
    colors = ["#cf7934" if i in antecedent else "#4c6277" for i in range(len(labels))]
    axis.bar(range(len(labels)), weights, color=colors)
    axis.set_xticks(range(len(labels)), labels, rotation=55, ha="right")
    axis.set_ylabel("Вес внимания")
    axis.set_title(f"«его» → «пациента»: слой {layer}, голова {head}")
    figure.tight_layout()
    figure.savefig(output_directory / "attention_map.png", dpi=150)
    plt.close(figure)
    return result


@torch.inference_mode()
def dilution(tokenizer, model, device: torch.device, words: list[str], fast: bool, destination: Path) -> list[dict]:
    """Менять длину наполнителя, сохраняя один и тот же префикс инструкции."""
    prefix = tokenizer(INSTRUCTION + "\n", add_special_tokens=False, return_offsets_mapping=True)
    # Отдельный токен перевода строки не относится к тексту инструкции.
    positions = [i for i, (left, right) in enumerate(prefix["offset_mapping"])
                 if i > 0 and left < len(INSTRUCTION) and right > 0]
    filler = tokenizer(" ".join(words[20000:30000]), add_special_tokens=False)["input_ids"]
    lengths = (64, 256, 768) if fast else (128, 512, 1024)
    results = []
    for length in lengths:
        ids = prefix["input_ids"] + filler[:length - len(prefix["input_ids"]) ]
        assert len(ids) == length
        output = model(input_ids=torch.tensor([ids], device=device), output_attentions=True, use_cache=False)
        row = torch.stack([layer[0, :, -1, :].float().cpu() for layer in output.attentions]).mean(dim=(0, 1))
        result = {
            "context_len": length, "instruction_len": len(positions),
            "share_on_instruction": float(row[positions].sum()),
            "sink_share": float(row[0]), "uniform_share": len(positions) / length,
        }
        results.append(result)
        print(f"Контекст {length}: инструкция {result['share_on_instruction']:.5f}, сток {result['sink_share']:.5f}", flush=True)
        del output
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    return results
