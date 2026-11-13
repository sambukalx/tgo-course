"""Выполнить все ячейки lab01.ipynb и сохранить настоящие выводы.

В первой работе достаточно стандартной библиотеки. Если Jupyter еще
не установлен, этот скрипт выполняет Python-ячейки последовательно.
В ноутбуке используются обычный Python и print, без магических команд.
"""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import time
import traceback
from urllib.request import urlopen


HERE = Path(__file__).resolve().parent
SOURCE = "https://raw.githubusercontent.com/nevmenandr/word2vec-russian-novels/master/books_before/WarAndPeace.txt"
SOURCE_BLOB = "8ab369096e7686214d5870a05a138fbc57493d94"


def check_corpus(data: bytes) -> None:
    """Сверить содержимое с Git-объектом корпуса из задания курса."""
    actual = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    if actual != SOURCE_BLOB:
        raise ValueError("корпус отличается от проверенной версии; проверьте источник")


def execute_notebook(path: Path) -> None:
    """Запустить код ячеек в общей области имен, записав stdout и ошибки."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__lab01_notebook__"}
    count = 0
    # Очищаем прежние выводы, чтобы после ошибки не остались результаты
    # еще не выполненных ячеек от предыдущего запуска.
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    old_directory = Path.cwd()
    try:
        os.chdir(HERE)
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            count += 1
            cell["execution_count"] = count
            stdout, stderr = io.StringIO(), io.StringIO()
            try:
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    exec(compile("".join(cell["source"]), f"{path.name}:cell-{count}", "exec"), namespace)
            except Exception as error:
                for name, stream in (("stdout", stdout), ("stderr", stderr)):
                    if stream.getvalue():
                        cell["outputs"].append({"output_type": "stream", "name": name, "text": stream.getvalue().splitlines(keepends=True)})
                cell["outputs"].append({"output_type": "error", "ename": type(error).__name__, "evalue": str(error), "traceback": traceback.format_exc().splitlines()})
                raise
            for name, stream in (("stdout", stdout), ("stderr", stderr)):
                if stream.getvalue():
                    cell["outputs"].append({"output_type": "stream", "name": name, "text": stream.getvalue().splitlines(keepends=True)})
                    print(stream.getvalue(), end="")
    finally:
        os.chdir(old_directory)
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def main() -> None:
    """При необходимости скачать корпус, проверить его и выполнить работу."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="скачать корпус, если его еще нет")
    args = parser.parse_args()
    corpus = HERE / "corpus.txt"
    if not corpus.is_file():
        if not args.download:
            parser.error("нет lab01/corpus.txt: скачайте его по ссылке в README или добавьте --download")
        print("Скачивание корпуса на локальный компьютер...")
        with urlopen(SOURCE, timeout=60) as response:
            data = response.read(10_000_001)
        if len(data) > 10_000_000:
            raise ValueError("корпус неожиданно больше 10 МБ")
        check_corpus(data)
        corpus.write_bytes(data)
    check_corpus(corpus.read_bytes())
    started = time.perf_counter()
    execute_notebook(HERE / "lab01.ipynb")
    print(f"Ноутбук выполнен за {time.perf_counter() - started:.2f} с")


if __name__ == "__main__":
    main()
