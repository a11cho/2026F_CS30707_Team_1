"""저장 중 중단되어도 완료된 파일을 읽을 수 있도록 교체합니다."""

import csv
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile


def atomic_write(destination, write):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
        path = Path(stream.name)
    try:
        write(path)
        with path.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(path, destination)
    finally:
        path.unlink(missing_ok=True)


def write_json(path, data):
    atomic_write(path, lambda tmp: tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"))


def write_csv(path, rows):
    if not rows:
        return

    def write(tmp):
        with tmp.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    atomic_write(path, write)
