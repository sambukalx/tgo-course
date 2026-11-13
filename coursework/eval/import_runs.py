"""Перенести экспорт реальных опытов в MLflow контейнера, сохранив происхождение."""

import argparse
import json
from pathlib import Path

from mlflow import MlflowClient


def import_exports(uri: str, folders: list[Path]) -> list[dict]:
    """Не повторять импорт source_run_id; исходные времена взяты из экспорта."""
    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name('tgo-document-assistant')
    experiment_id = experiment.experiment_id if experiment else client.create_experiment('tgo-document-assistant')
    present = {run.data.tags.get('source_run_id') for run in client.search_runs([experiment_id])}
    mapping = []
    for folder in folders:
        path = folder / 'mlflow-export.json'
        if not path.is_file():
            continue
        for source in json.loads(path.read_text(encoding='utf-8')):
            if source['run_id'] in present:
                continue
            tags = dict(source.get('tags', {}))
            tags.update(source_run_id=source['run_id'], imported_from=str(path.relative_to(Path(__file__).resolve().parents[2])),
                        import_note='Копия настоящего завершенного опыта; инференс при импорте не выполняется')
            run = client.create_run(experiment_id, start_time=source['start_time'], tags=tags)
            for key, value in source['params'].items():
                client.log_param(run.info.run_id, key, value)
            for key, value in source['metrics'].items():
                client.log_metric(run.info.run_id, key, value, timestamp=source['end_time'])
            client.log_artifact(run.info.run_id, str(path), artifact_path='provenance')
            name = source.get('tags', {}).get('mlflow.runName', '')
            predictions = folder / (name + '.json')
            if predictions.is_file():
                client.log_artifact(run.info.run_id, str(predictions), artifact_path='predictions')
            client.set_terminated(run.info.run_id, status=source['status'], end_time=source['end_time'])
            present.add(source['run_id'])
            mapping.append(dict(source_run_id=source['run_id'], imported_run_id=run.info.run_id))
    return mapping


def main() -> None:
    """Импорт нужен только для просмотра готовых опытов в чистом Docker MLflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--uri', default='http://127.0.0.1:5000')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent / 'results'
    mapping = import_exports(args.uri, [root, root / 'main', root / 'final'])
    (root / 'docker-import.json').write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding='utf-8', newline='\n')
    print(json.dumps(mapping, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
