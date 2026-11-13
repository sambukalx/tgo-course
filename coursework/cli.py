"""CLI работает только через HTTP и не требует torch или доступа к базе."""

import argparse
import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def request(url: str, payload: dict | None = None) -> dict:
    """Передать UTF-8 JSON и ограничить ожидание сервера."""
    body = json.dumps(payload, ensure_ascii=False).encode('utf-8') if payload is not None else None
    req = Request(url, data=body, headers={'Content-Type': 'application/json'})
    with urlopen(req, timeout=125) as response:
        return json.load(response)


def main() -> None:
    """Спросить один вопрос либо показать текущие метрики сервиса."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('question', nargs='?')
    parser.add_argument('--url', default=os.environ.get('COURSE_API_URL', 'http://127.0.0.1:8000'))
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--no-verify', action='store_true')
    parser.add_argument('--metrics', action='store_true')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    if not args.metrics and not args.question:
        parser.error('укажите вопрос или --metrics')
    try:
        result = request(args.url + '/metrics') if args.metrics else request(
            args.url + '/ask', dict(question=args.question, top_k=args.top_k, verify=not args.no_verify))
    except (HTTPError, URLError, TimeoutError) as error:
        print('Запрос не выполнен:', error, file=sys.stderr)
        raise SystemExit(1)
    if args.json or args.metrics:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print(result['answer'])
    print(f"Время: {result['seconds']:.2f} с; токенов: {result['generated_tokens']}; проверяющая модель: {result['verification']}")
    if result.get('empty_answer'):
        print('Отказ: модель вернула пустой ответ или одну ссылку.')
    if result.get('invalid_citations'):
        print('Отказ: модель сослалась на отсутствующий номер источника.')
    if result['unsupported_claims']:
        print('Не подтверждено:', ' '.join(result['unsupported_claims']))
    for i, source in enumerate(result['sources'], 1):
        print(f"[{i}] {source['source']} — {source['section']} (score={source['score']:.3f})")


if __name__ == '__main__':
    main()
