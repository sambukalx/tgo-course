"""Считать долю неподтвержденных ответов по явной сверке с источниками."""

import argparse
import hashlib
import json
from pathlib import Path

from mlflow import MlflowClient

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def evaluate_review(path: Path, tracking_uri: str | None = None) -> list[dict]:
    """Проверить хеш ответов, полноту разметки и буквальные цитаты доказательств."""
    annotations = json.loads(path.read_text(encoding='utf-8'))
    summaries = json.loads((HERE / 'results/final/experiments.json').read_text(encoding='utf-8'))
    client = MlflowClient(tracking_uri=tracking_uri) if tracking_uri else None
    result = []
    for item in annotations['configurations']:
        transcript = (ROOT / item['transcript']).resolve()
        assert transcript.is_relative_to(ROOT)
        raw = transcript.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == item['sha256'], 'ответы изменились после сверки'
        responses = {row['id']: row for row in json.loads(raw)}
        labels = item['labels']
        assert {row['id'] for row in labels} == set(responses)
        assert len(labels) == len(responses), 'повторяющиеся метки'
        accepted, unsupported = 0, 0
        for label in labels:
            row = responses[label['id']]
            claims = label.get('unsupported_claims', [])
            if row['response']['refused']:
                assert not claims
                continue
            accepted += 1
            unsupported += bool(claims)
            for claim in claims:
                assert claim['claim'] in row['response']['answer']
                source = (ROOT / claim['source']).resolve()
                assert source.is_relative_to(ROOT / 'lectures')
                assert claim['evidence_quote'] in source.read_text(encoding='utf-8')
        metrics = dict(source_review_hallucination_rate=unsupported / max(1, accepted),
                       source_review_unsupported_answers=unsupported,
                       source_review_accepted_answers=accepted)
        result.append(dict(name=item['name'], metrics=metrics, method=annotations['method']))
        summary = next(run for run in summaries if run['name'] == item['name'])
        summary['source_review'] = metrics
        if client:
            for name, value in metrics.items():
                client.log_metric(summary['run_id'], name, value)
            client.set_tag(summary['run_id'], 'source_review_method', annotations['method'])
            client.log_artifact(summary['run_id'], str(path), artifact_path='source-review')
    target = HERE / 'results/final/source-review.json'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8', newline='\n')
    (HERE / 'results/final/experiments.json').write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding='utf-8', newline='\n')
    if client:
        exported = []
        for summary in summaries:
            run = client.get_run(summary['run_id'])
            exported.append(dict(run_id=run.info.run_id, status=run.info.status, start_time=run.info.start_time,
                                 end_time=run.info.end_time, params=run.data.params, metrics=run.data.metrics, tags=run.data.tags))
        (HERE / 'results/final/mlflow-export.json').write_text(json.dumps(exported, ensure_ascii=False, indent=2), encoding='utf-8', newline='\n')
    return result


def main() -> None:
    """Пересчет разметки не запускает языковую модель и не меняет ее ответы."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, default=HERE / 'results/final/source-review-annotations.json')
    parser.add_argument('--tracking-uri')
    args = parser.parse_args()
    print(json.dumps(evaluate_review(args.annotations, args.tracking_uri), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
