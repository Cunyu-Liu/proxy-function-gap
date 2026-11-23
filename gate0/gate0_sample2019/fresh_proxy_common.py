import csv
import gzip
import io
import json
from pathlib import Path


class ClosingTextIOWrapper(io.TextIOWrapper):
    def __init__(self, gzip_handle, raw):
        self._raw = raw
        super().__init__(gzip_handle, encoding="utf-8", newline="")

    def close(self):
        try:
            super().close()
        finally:
            self._raw.close()


def open_deterministic_tsv_gz(path, fieldnames):
    raw = open(path, "xb")
    gzip_handle = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
    handle = ClosingTextIOWrapper(gzip_handle, raw)
    writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    return handle, writer


def read_tsv(path):
    with gzip.open(path, "rt", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def write_row(writer, row):
    writer.writerow({key: "" if value is None else str(value) for key, value in row.items() if key in writer.fieldnames})


def intersections(role_sets):
    names = sorted(role_sets)
    return [
        {"left": left, "right": right, "count": len(role_sets[left] & role_sets[right])}
        for left, right in __import__("itertools").combinations(names, 2)
    ]
