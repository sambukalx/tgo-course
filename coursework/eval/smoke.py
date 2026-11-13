"""Проверка развернутого сервиса через HTTP; не заменяет оценку на 32 вопросах."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from coursework.cli import request


def check_service(url: str, output: Path) -> dict:
    """Проверить CPU, размеры моделей, журналирование, задержку и один отказ."""
    health = request(url + '/health')
    assert health['status'] == 'ready' and health['device'] == 'cpu'
    assert health['model_params'] <= 4_000_000_000
    assert health['embedding_params'] <= 600_000_000
    before = request(url + '/metrics')
    examples = []
    for question in (
        'Что означает weight decay в AdamW?',
        'Из какого распределения шума получается средняя квадратичная ошибка?',
        'В каком кафе преподаватель будет обедать завтра?',
    ):
        started = time.perf_counter()
        response = request(url + '/ask', dict(question=question, top_k=3, verify=True))
        elapsed = time.perf_counter() - started
        assert elapsed < 120, f'ответ занял {elapsed:.1f} с'
        examples.append(dict(question=question, http_seconds=elapsed, response=response))
        print(question, '=>', response['answer'], f'({elapsed:.2f} с)', flush=True)
    after = request(url + '/metrics')
    assert after['requests'] - before['requests'] == len(examples)
    assert examples[-1]['response']['refused'], 'в корпусе нет сведений о кафе'
    result = dict(checked_at_utc=datetime.now(timezone.utc).isoformat(), url=url, health=health,
                  before_metrics=before, examples=examples, after_metrics=after)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8', newline='\n')
    return result


def main() -> None:
    """Из корня: python -m coursework.eval.smoke --url http://127.0.0.1:8000."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'results/deployment.json')
    args = parser.parse_args()
    result = check_service(args.url, args.output)
    print(json.dumps(result['after_metrics'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
