import csv
import gzip
import hashlib
import json
import math
from pathlib import Path


def load_config(path):
    return json.loads(Path(path).read_text())


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value):
    return hashlib.sha256(value.encode()).hexdigest()


def raw_row_hash(row):
    payload = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_text(payload)


def normalize_sequence(raw_sequence, file_rule):
    if file_rule["length"] is None:
        if not 25 <= len(raw_sequence) <= 100:
            raise ValueError("Variable-length UTR outside 25..100")
        return raw_sequence, file_rule["format"]
    if len(raw_sequence) != file_rule["length"]:
        raise ValueError("Unexpected fixed raw-sequence length")
    if file_rule["suffix"] and not raw_sequence.endswith(file_rule["suffix"]):
        raise ValueError("Fixed raw-sequence suffix mismatch")
    if file_rule["length"] == 50 and file_rule["suffix"]:
        raise ValueError("Invalid UTR50 rule")
    return raw_sequence[:50], file_rule["format"]


def finite_float(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def open_tsv_gz(path, fieldnames):
    handle = gzip.open(path, "wt", newline="", compresslevel=5)
    writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    return handle, writer


def row_value(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, set)):
        return "|".join(str(item) for item in value)
    return str(value)


def write_row(writer, row):
    writer.writerow({key: row_value(row.get(key)) for key in writer.fieldnames})


def hamming_distance(left, right):
    if len(left) != len(right):
        return None
    return sum(a != b for a, b in zip(left, right))


def source_metadata(filename):
    gsm = filename.split("_")[0]
    if filename == "GSM3130443_designed_library.csv.gz":
        return gsm, "designed_library", "designed_single_assay", "designed_no_replicate", "single_assay", "eGFP", "unmod"
    if filename.startswith("GSM4084997"):
        return gsm, "varying_length_library", "varying_single_assay", "varying_no_replicate", "single_assay", "eGFP", "unmod"
    parts = filename[:-7].split("_")
    replicate = parts[-1]
    chemistry = "m1pseudo" if "m1pseudo" in filename else "pseudo" if "pseudo" in filename else "unmod"
    cds = "mCherry" if "mcherry" in filename else "eGFP"
    family = "random_mcherry_library" if "mcherry" in filename else "random_egfp_library"
    return gsm, "random_library", family, filename[:-7], "experimental_replicate", cds, chemistry


def make_raw_row_id(gsm, source_row_number):
    return f"{gsm}:row-{source_row_number:07d}"
