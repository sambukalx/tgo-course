"""BPE, бюджет контекста, поиск по векторам и внимание настоящей Qwen."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
from urllib.request import urlopen

import torch
from transformers import AutoModel, AutoTokenizer

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from lab02.attention import dilution, pronoun_attention
from lab02.bpe import tokenize_words
from lab02.build_bpe import build
from lab02.models import E5, QWEN, choose_device, e5_vectors, load_qwen, qwen_tokenizer, qwen_vectors, search

ARTIFACTS = HERE / "artifacts"
CORPUS_URL = "https://raw.githubusercontent.com/nevmenandr/word2vec-russian-novels/master/books_before/WarAndPeace.txt"
CORPUS_SHA256 = "5f088a66e1f3992153555fdf0361652f759697e429bd698fb2900fdf26df9836"
ENGLISH = (
    "The prince had been talking for almost an hour, and the guests were growing tired. "
    "Anna looked at the clock, then at the window, then at the clock again. Nobody dared "
    "to leave before the host, and the host did not seem to notice the time at all. "
    "A servant brought tea, and the conversation turned, as it always did in that house, "
    "to the war and to the emperor."
)


def save_json(name: str, value) -> None:
    """Записать артефакт без округления чисел."""
    (ARTIFACTS / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_corpus() -> str:
    """Скачать фиксированный корпус один раз и проверить содержимое."""
    path = HERE / "corpus.txt"
    if not path.is_file():
        with urlopen(CORPUS_URL, timeout=60) as response:
            raw = response.read()
        if hashlib.sha256(raw).hexdigest() != CORPUS_SHA256:
            raise ValueError("изменилась версия корпуса")
        path.write_bytes(raw)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CORPUS_SHA256:
        raise ValueError("корпус поврежден или отличается от исходного")
    return raw.decode("utf-8")


def select_paragraphs(text: str, limit: int) -> list[dict]:
    """Взять первые абзацы на 40-120 русских слов, сохранив исходный текст."""
    paragraphs = []
    for paragraph in text.splitlines():
        if 40 <= len(tokenize_words(paragraph)) <= 120:
            paragraphs.append({"id": len(paragraphs), "text": paragraph})
            if len(paragraphs) == limit:
                break
    return paragraphs


def token_statistics(tokenizer, words: list[str]) -> dict:
    """Сравнить ровно заданные русский и английский отрывки."""
    russian = " ".join(words[500:600])
    result = {"tokenizer": QWEN, "vocab_size": len(tokenizer), "context_tokens": 4096}
    for name, text, n_words in (("ru", russian, 100), ("en", ENGLISH, len(ENGLISH.split()))):
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        ratio = len(ids) / n_words
        result[name] = {"words": n_words, "tokens": len(ids), "tokens_per_word": ratio,
                        "words_in_context": int(4096 / ratio)}
        if name == "ru":
            result["example_ru"] = [tokenizer.decode([token]) for token in ids[:20]]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="быстрый режим для CI")
    args = parser.parse_args()
    ARTIFACTS.mkdir(exist_ok=True)
    started = time.perf_counter()
    torch.manual_seed(42)
    torch.set_num_threads(8)
    device = choose_device()
    text = load_corpus()
    words = tokenize_words(text)
    bpe = build(text, fast=args.fast)
    save_json("bpe.json", bpe)
    print(f"BPE: {bpe['num_merges']} слияний; устройство {device}", flush=True)

    tokenizer = qwen_tokenizer()
    save_json("tokens_per_word.json", token_statistics(tokenizer, words))
    paragraphs = select_paragraphs(text, 60 if args.fast else 500)
    save_json("paragraphs.json", paragraphs)
    queries = json.loads((HERE / "queries.json").read_text(encoding="utf-8"))
    texts = [p["text"] for p in paragraphs] + [q["query"] for q in queries]
    n = len(paragraphs)

    qwen = load_qwen(device)
    static, contextual = qwen_vectors(texts, tokenizer, qwen, device)
    nearest = {
        "static": search(static[n:], static[:n]),
        "contextual": search(contextual[n:], contextual[:n]),
    }
    save_json("attention_map.json", pronoun_attention(tokenizer, qwen, device, ARTIFACTS))
    dilution(tokenizer, qwen, device, words, args.fast, ARTIFACTS / "dilution.csv")
    qwen_revision = getattr(qwen.config, "_commit_hash", None)
    del qwen, static, contextual
    gc.collect()

    e5_tokenizer = AutoTokenizer.from_pretrained(E5)
    e5_model = AutoModel.from_pretrained(E5).to(device).eval()
    passages = ["passage: " + paragraph["text"] for paragraph in paragraphs]
    requests = ["query: " + query["query"] for query in queries]
    vectors = e5_vectors(passages + requests, e5_tokenizer, e5_model, device)
    nearest["e5"] = search(vectors[n:], vectors[:n])
    search_results = []
    for i, query in enumerate(queries):
        answer = next(p["id"] for p in paragraphs if query["answer_phrase"] in p["text"])
        search_results.append({**query, "answer_id": answer,
                               "top": {method: rows[i] for method, rows in nearest.items()}})
    recalls = {method: sum(q["answer_id"] in q["top"][method] for q in search_results) / len(queries)
               for method in nearest}
    save_json("search.json", {"k": 5, "n_paragraphs": n, "queries": search_results, "recall_at_k": recalls})
    save_json("run_info.json", {"fast": args.fast, "device": str(device), "seed": 42,
                               "seconds": time.perf_counter() - started, "torch": torch.__version__,
                               "qwen_revision": qwen_revision, "e5_revision": getattr(e5_model.config, "_commit_hash", None)})
    print("Recall@5:", recalls, flush=True)
    print(f"Готово за {time.perf_counter() - started:.1f} с", flush=True)


if __name__ == "__main__":
    main()
