"""Оценка двух конфигураций через HTTP с записью настоящих прогонов в MLflow."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import statistics

import mlflow

from coursework.cli import request

HERE = Path(__file__).resolve().parent


def text_f1(reference: str, answer: str) -> float:
    """Token-F1 без претензии на полноценную семантическую оценку русского языка."""
    words = lambda text: Counter(re.findall(r'\w+', text.casefold()))
    gold, predicted = words(reference), words(answer)
    common = sum((gold & predicted).values())
    return 2 * common / (sum(gold.values()) + sum(predicted.values())) if common else 0.


def run_experiments(url: str, output: Path, tracking_uri: str | None = None) -> list[dict]:
    """Менять только top_k: одинаковые вопросы, модели, генерация и проверка."""
    output.mkdir(parents=True, exist_ok=True)
    dataset_bytes = (HERE / 'dataset.json').read_bytes()
    rows = json.loads(dataset_bytes)
    digest = hashlib.sha256(dataset_bytes).hexdigest()
    state = HERE.parent / 'state'
    state.mkdir(exist_ok=True)
    uri = tracking_uri or os.environ.get('MLFLOW_TRACKING_URI') or 'sqlite:///' + (state / 'mlflow.sqlite').resolve().as_posix()
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment('tgo-document-assistant')
    health = request(url + '/health')
    summaries = []
    for top_k in (1, 3):
        model_name = health.get('model', 'Qwen/Qwen3-0.6B')
        prompt_version = health.get('prompt_version', 'ru-v1')
        name = f'{model_name.split("/")[-1].lower()}-{prompt_version}-k{top_k}-verified'
        predictions = []
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params(dict(top_k=top_k, verify=True, dataset_sha256=digest,
                                   corpus_sha256=health['corpus_sha256'], model=model_name, prompt_version=prompt_version,
                                   implementation_sha256=health.get('implementation_sha256', 'not_recorded_in_prototype'),
                                   embedding='intfloat/multilingual-e5-small', temperature=0,
                                   max_new_tokens=120, test_examples=len(rows), device='cpu'))
            for i, row in enumerate(rows):
                response = request(url + '/ask', dict(question=row['question'], top_k=top_k, verify=True))
                correct = bool(response['refused']) if row['unanswerable'] else (
                    not response['refused'] and all(re.search(pattern, response['answer'], re.I) for pattern in row['keywords']))
                retrieved = any(source['source'] == row['source'] and source['section'] == row['section']
                                for source in response['sources']) if not row['unanswerable'] else None
                predictions.append(dict(id=row['id'], question=row['question'], reference=row['answer'],
                                        unanswerable=row['unanswerable'], correct=bool(correct),
                                        token_f1=text_f1(row['answer'], response['answer']), retrieved_gold=retrieved,
                                        response=response))
                print(f"{name} {i+1}/{len(rows)} {row['id']}: {'OK' if correct else 'ошибка'}, "
                      f"{response['seconds']:.2f}с, отказ={response['refused']}", flush=True)
                # Промежуточный файл сохраняет уже выполненные запросы при прерывании.
                (output / f'{name}.json').write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding='utf-8', newline='\n')
            known = [row for row in predictions if not row['unanswerable']]
            unknown = [row for row in predictions if row['unanswerable']]
            accepted = [row for row in predictions if not row['response']['refused']]
            times = sorted(row['response']['seconds'] for row in predictions)
            metrics = dict(accuracy=sum(row['correct'] for row in predictions) / len(predictions),
                           answerable_accuracy=sum(row['correct'] for row in known) / len(known),
                           correct_refusal_rate=sum(row['correct'] for row in unknown) / len(unknown),
                           retrieval_recall=sum(row['retrieved_gold'] for row in known) / len(known),
                           mean_token_f1=statistics.mean(row['token_f1'] for row in known),
                           accepted_error_rate=sum(not row['correct'] for row in accepted) / max(1, len(accepted)),
                           refusal_rate=sum(row['response']['refused'] for row in predictions) / len(predictions),
                           mean_seconds=statistics.mean(times), max_seconds=max(times),
                           p95_seconds=times[min(len(times)-1, int(.95 * len(times)))],
                           generated_tokens=sum(row['response']['generated_tokens'] for row in predictions))
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(output / f'{name}.json'), artifact_path='predictions')
            summaries.append(dict(name=name, run_id=run.info.run_id, params=dict(top_k=top_k, verify=True),
                                  metrics=metrics, dataset_sha256=digest, corpus_sha256=health['corpus_sha256']))
            (output / 'experiments.json').write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding='utf-8', newline='\n')
            print(json.dumps(summaries[-1], ensure_ascii=False, indent=2), flush=True)
    # Читаемый переносимый экспорт не зависит от локальных путей базы MLflow.
    client = mlflow.MlflowClient()
    exported = []
    for summary in summaries:
        saved = client.get_run(summary['run_id'])
        exported.append(dict(run_id=saved.info.run_id, status=saved.info.status, start_time=saved.info.start_time,
                             end_time=saved.info.end_time, params=saved.data.params, metrics=saved.data.metrics,
                             tags=saved.data.tags))
    (output / 'mlflow-export.json').write_text(json.dumps(exported, ensure_ascii=False, indent=2), encoding='utf-8', newline='\n')
    return summaries


def main() -> None:
    """Запуск из корня: python -m coursework.eval.run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--output', type=Path, default=HERE / 'results')
    parser.add_argument('--tracking-uri')
    args = parser.parse_args()
    run_experiments(args.url, args.output, args.tracking_uri)


if __name__ == '__main__':
    main()
