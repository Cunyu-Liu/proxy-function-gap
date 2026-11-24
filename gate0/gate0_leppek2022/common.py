import csv
import gzip
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_config(path):
    return json.loads(Path(path).read_text())


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def raw_row_hash(row):
    payload = {key: value for key, value in row.items() if not key.startswith("__")}
    return sha256_text(canonical_json(payload))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def finite_float(value):
    if value is None or str(value).strip() == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def row_value(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, set)):
        return "|".join(str(item) for item in value)
    return str(value)


def open_tsv_gz(path, fieldnames):
    handle = gzip.open(path, "wt", newline="", compresslevel=5)
    writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    return handle, writer


def write_row(writer, row):
    writer.writerow({key: row_value(row.get(key)) for key in writer.fieldnames})


def write_tsv(path, fieldnames, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row_value(row.get(key)) for key in fieldnames})


def read_tsv(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def normalize_rna(value):
    sequence = value.strip().upper().replace("U", "T")
    if not sequence or set(sequence) - set("ACGT"):
        raise ValueError("RNA sequence is empty or has non-ACGU characters")
    return sequence


def levenshtein(left, right, cutoff=None):
    if left == right:
        return 0
    if len(left) > len(right):
        left, right = right, left
    if cutoff is not None and len(right) - len(left) > cutoff:
        return cutoff + 1
    previous = list(range(len(left) + 1))
    for right_index, right_char in enumerate(right, 1):
        current = [right_index]
        row_minimum = right_index
        for left_index, left_char in enumerate(left, 1):
            value = min(
                current[-1] + 1,
                previous[left_index] + 1,
                previous[left_index - 1] + (left_char != right_char),
            )
            current.append(value)
            row_minimum = min(row_minimum, value)
        if cutoff is not None and row_minimum > cutoff:
            return cutoff + 1
        previous = current
    return previous[-1]


def levenshtein_full(left, right):
    return levenshtein(left, right, cutoff=None)


def levenshtein_bit(left, right):
    if not left:
        return len(right)
    if not right:
        return len(left)
    masks = {}
    for index, char in enumerate(left):
        masks[char] = masks.get(char, 0) | (1 << index)
    positive = ~0
    negative = 0
    score = len(left)
    last = 1 << (len(left) - 1)
    for char in right:
        equal = masks.get(char, 0)
        xv = equal | negative
        xh = (((equal & positive) + positive) ^ positive) | equal
        ph = negative | ~(xh | positive)
        mh = positive & xh
        if ph & last:
            score += 1
        if mh & last:
            score -= 1
        ph = (ph << 1) | 1
        mh <<= 1
        positive = mh | ~(xv | ph)
        negative = ph & xv
    return score


def quantile(sorted_values, probability):
    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    return sorted_values[lower] * (upper - position) + sorted_values[upper] * (position - lower)


def distribution(values):
    numbers = sorted(value for value in values if value is not None and math.isfinite(value))
    if not numbers:
        return {"n": 0}
    mean = math.fsum(numbers) / len(numbers)
    variance = math.fsum((value - mean) ** 2 for value in numbers) / (len(numbers) - 1) if len(numbers) > 1 else None
    return {
        "n": len(numbers),
        "mean": mean,
        "sd": math.sqrt(variance) if variance is not None else None,
        "min": numbers[0],
        "q1": quantile(numbers, 0.25),
        "median": quantile(numbers, 0.5),
        "q3": quantile(numbers, 0.75),
        "max": numbers[-1],
    }


def ensure_new_directory(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    path.mkdir(parents=True)


def checksum_directory(path, exclude=None):
    path = Path(path)
    exclude = set(exclude or [])
    result = {}
    for item in sorted(path.rglob("*")):
        if item.is_file() and item.relative_to(path).as_posix() not in exclude:
            result[item.relative_to(path).as_posix()] = {
                "bytes": item.stat().st_size,
                "sha256": sha256_file(item),
            }
    return result
