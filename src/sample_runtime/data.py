import csv
import gzip
import hashlib

from .provenance import DESIGNED_DATA, EXPECTED_COUNTS, SCALER, STEP_LIBRARIES

STEP_FIELDS = (
    "library", "id", "generation", "sparse_step", "utr", "sequence_id", "total_reads",
    "measured_rl", "original_raw_output", "retrained_raw_output", "original_pred_mrl", "retrained_pred_mrl",
)
TARGET_FIELDS = (
    "library", "id", "generation", "sparse_step", "target_mrl_info1", "utr", "sequence_id", "total_reads",
    "measured_rl", "published_pred_info2", "original_raw_output", "retrained_raw_output",
    "original_pred_mrl", "retrained_pred_mrl",
)


def sequence_id(utr):
    return hashlib.sha256(utr.encode("ascii")).hexdigest()


def load_design_rows():
    step_rows = []
    target_rows = []
    with gzip.open(DESIGNED_DATA, "rt", newline="") as handle:
        for source in csv.DictReader(handle):
            if source["designed"] != "True":
                continue
            if len(source["utr"]) != 50 or set(source["utr"]) - set("ACGT"):
                raise ValueError(f"Invalid UTR: {source[utr]!r}")
            common = {
                "library": source["library"],
                "id": source["id"],
                "generation": "",
                "sparse_step": "",
                "utr": source["utr"],
                "sequence_id": sequence_id(source["utr"]),
                "total_reads": source["total"],
                "measured_rl": source["rl"],
            }
            if source["library"] in STEP_LIBRARIES:
                common["generation"] = str(int(source["info1"]))
                common["sparse_step"] = str(int(source["info2"]))
                step_rows.append(common)
            elif source["library"].startswith("target_"):
                common["target_mrl_info1"] = source["info1"]
                common["published_pred_info2"] = source["info2"]
                target_rows.append(common)
    if len(step_rows) != EXPECTED_COUNTS["step"] or len(target_rows) != EXPECTED_COUNTS["target"]:
        raise ValueError(f"Unexpected row counts: step={len(step_rows)}, target={len(target_rows)}")
    step_rows.sort(key=lambda row: connection_key(row))
    target_rows.sort(key=lambda row: connection_key(row))
    validate_connections(step_rows, target_rows)
    return step_rows, target_rows


def connection_key(row):
    return row["library"], row["id"], row["sparse_step"], row["utr"]


def validate_connections(step_rows, target_rows):
    for name, rows in (("step", step_rows), ("target", target_rows)):
        keys = [connection_key(row) for row in rows]
        if len(keys) != EXPECTED_COUNTS[name] or len(set(keys)) != len(keys):
            raise ValueError(f"{name} connection keys are not one-to-one")
        if any(not row["id"] or not row["measured_rl"] for row in rows):
            raise ValueError(f"{name} rows contain missing ids or measured rl")
    step_pairs = {(row["library"], row["id"], row["generation"], row["sparse_step"], row["utr"]) for row in step_rows}
    if len(step_pairs) != EXPECTED_COUNTS["step"]:
        raise ValueError("Step generation and sparse-step join is not one-to-one")


def attach_predictions(rows, predictions):
    if len(rows) != len(predictions["original"]) or len(rows) != len(predictions["retrained"]):
        raise ValueError("Prediction row count mismatch")
    result = []
    for index, source in enumerate(rows):
        row = dict(source)
        original_raw = float(predictions["original"][index])
        retrained_raw = float(predictions["retrained"][index])
        row["original_raw_output"] = repr(original_raw)
        row["retrained_raw_output"] = repr(retrained_raw)
        row["original_pred_mrl"] = repr(SCALER["mean"] + SCALER["scale"] * original_raw)
        row["retrained_pred_mrl"] = repr(SCALER["mean"] + SCALER["scale"] * retrained_raw)
        result.append(row)
    return result


def write_rows(path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
