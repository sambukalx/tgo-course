"""SQLite хранит историю запросов; метрики считаются по тем же записям."""

import json
from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path
import sqlite3
from datetime import datetime, timezone


class RequestStore:
    """Короткие транзакции и отдельное соединение для каждого вызова."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as database:
            database.execute("""CREATE TABLE IF NOT EXISTS requests (
                id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, question TEXT NOT NULL,
                response TEXT NOT NULL, seconds REAL NOT NULL, generated_tokens INTEGER NOT NULL,
                refused INTEGER NOT NULL, failed INTEGER NOT NULL DEFAULT 0)""")

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Commit/rollback и обязательный close: SQLite context сам файл не закрывает."""
        database = sqlite3.connect(self.path, timeout=20)
        try:
            with database:
                yield database
        finally:
            database.close()

    def add(self, question: str, response: dict, failed: bool = False) -> int:
        """Сохранить результат и вернуть номер запроса."""
        with self.connect() as database:
            cursor = database.execute(
                "INSERT INTO requests(created_at,question,response,seconds,generated_tokens,refused,failed) VALUES(?,?,?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(), question,
                 json.dumps(response, ensure_ascii=False), response['seconds'],
                 response['generated_tokens'], int(response['refused']), int(failed)),
            )
            return int(cursor.lastrowid)

    def metrics(self) -> dict:
        """Число запросов, среднее, p95, доля отказов и расход токенов."""
        with self.connect() as database:
            rows = database.execute('SELECT seconds,generated_tokens,refused,failed FROM requests').fetchall()
        if not rows:
            return dict(requests=0, mean_seconds=0, p95_seconds=0, refusal_rate=0, generated_tokens=0, failures=0)
        times = sorted(row[0] for row in rows)
        position = (len(times) - 1) * .95
        low = int(position)
        high = min(low + 1, len(times) - 1)
        return dict(requests=len(rows), mean_seconds=sum(times) / len(rows),
                    p95_seconds=times[low] + (times[high] - times[low]) * (position - low),
                    refusal_rate=sum(row[2] for row in rows) / len(rows),
                    generated_tokens=sum(row[1] for row in rows), failures=sum(row[3] for row in rows))
