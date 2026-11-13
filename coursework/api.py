"""HTTP-инференс: /ask, /health, /metrics; интерфейс не импортирует модели."""

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
import logging
import os
from pathlib import Path
import time
import psutil

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from coursework.engine import CourseAssistant, MODEL, REVISION, PROMPT_VERSION
from coursework.storage import RequestStore

ROOT = Path(os.environ.get('COURSE_ROOT', Path(__file__).resolve().parent.parent))
STATE = Path(os.environ.get('COURSE_STATE', ROOT / 'coursework/state'))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Загрузить модели и индекс до приема запросов, закрыть при завершении."""
    STATE.mkdir(parents=True, exist_ok=True)
    app.state.store = RequestStore(STATE / 'requests.sqlite')
    app.state.assistant = CourseAssistant(ROOT, STATE, int(os.environ.get('COURSE_THREADS', '8')))
    try:
        yield
    finally:
        app.state.assistant.close()


app = FastAPI(title='Ассистент по ТГО', version='1.0', lifespan=lifespan)


class Question(BaseModel):
    """Ограничить размер входа и разрешенные параметры эксперимента."""

    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=5)
    verify: bool = True


@app.get('/health')
def health() -> dict:
    """Готовность означает, что startup уже загрузил модели и индекс."""
    assistant = app.state.assistant
    return dict(status='ready', model=MODEL, model_revision=REVISION, prompt_version=PROMPT_VERSION,
                model_params=assistant.parameter_count, chunks=assistant.index.size,
                corpus_sha256=assistant.index.signature, device=str(next(assistant.model.parameters()).device),
                embedding_params=sum(parameter.numel() for parameter in assistant.index.model.parameters()),
                implementation_sha256=assistant.implementation_sha256,
                rss_gb=psutil.Process().memory_info().rss / 1024**3)


@app.post('/ask')
def ask(body: Question) -> dict:
    """Выполнить запрос и записать ответ, время, токены и отказ в SQLite."""
    if not body.question.strip():
        raise HTTPException(422, 'Пустой вопрос')
    started = time.perf_counter()
    try:
        result = app.state.assistant.ask(body.question, body.top_k, body.verify)
    except Exception:
        logging.exception('Ошибка обработки запроса')
        result = dict(answer='Ошибка инференса', seconds=time.perf_counter() - started,
                      generated_tokens=0, refused=True)
        app.state.store.add(body.question, result, failed=True)
        raise HTTPException(500, 'Ошибка инференса; подробности в журнале сервера')
    result['request_id'] = app.state.store.add(body.question, result)
    return result


@app.get('/metrics')
def metrics() -> dict:
    """Метрики всех сохраненных запросов, включая ошибки инференса."""
    return app.state.store.metrics()
