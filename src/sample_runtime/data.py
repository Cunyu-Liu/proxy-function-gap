import csv
import gzip
import hashlib

from .provenance import DESIGNED_DATA, EXPECTED_COUNTS, GATE0_SEQUENCE_INDEX, SCALER, STEP_LIBRARIES

STEP_FIELDS = (
    "raw_row_id", "sequence_id", "sequence_hash", "source_row_number", "raw_index",
    "library", "id", "generation", "sparse_step", "utr", "total_reads", "measured_rl",
    "original_raw_output", "retrained_raw_output", "original_pred_mrl", "retrained_pred_mrl",
)
TARGET_FIELDS = (
    "raw_row_id", "sequence_id", "sequence_hash", "source_row_number", "raw_index",
    "library", "id", "generation", "sparse_step", "target_mrl_info1", "utr", "total_reads",
    "measured_rl", "published_pred_info2", "original_raw_output", "retrained_raw_output",
    "original_pred_mrl", "retrained_pred_mrl",
)


def sequence_hash(utr):
    return hashlib.sha256(utr.encode("ascii")).hexdigest()


def gate0_sequence_id(utr):
    return "sample2019_" + sequence_hash(utr)


def gate0_raw_row_id(source_row_number):
    if source_row_number < 1:
        raise ValueError("Gate 0 source rows are one-based")
    return f"GSM3130443:row-{source_row_number:07d}"


def load_design_rows():
    step_rows = []
    target_rows = []
    with gzip.open(DESIGNED_DATA, "rt", newline="") as handle:
        for source_row_number, source in enumerate(csv.DictReader(handle), 1):
            if source["designed"] != "True":
                continue
            if len(source["utr"]) != 50 or set(source["utr"]) - set("ACGT"):
                raise ValueError(f"Invalid UTR: {source[utr]!r}")
            common = {
                "raw_row_id": gate0_raw_row_id(source_row_number),
                "sequence_id": gate0_sequence_id(source["utr"]),
                "sequence_hash": sequence_hash(source["utr"]),
                "source_row_number": str(source_row_number),
                "raw_index": source[""],
                "library": source["library"],
                "id": source["id"],
                "generation": "",
                "sparse_step": "",
                "target_mrl_info1": "",
                "published_pred_info2": "",
                "utr": source["utr"],
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
    all_rows = step_rows + target_rows
    if len({row["raw_row_id"] for row in all_rows}) != len(all_rows):
        raise ValueError("Gate 0 raw_row_id values are not unique")
    for name, rows in (("step", step_rows), ("target", target_rows)):
        keys = [connection_key(row) for row in rows]
        if len(keys) != EXPECTED_COUNTS[name] or len(set(keys)) != len(keys):
            raise ValueError(f"{name} connection keys are not one-to-one")
        if any(not row["id"] or not row["measured_rl"] for row in rows):
            raise ValueError(f"{name} rows contain missing ids or measured rl")
        for row in rows:
            if row["sequence_id"] != gate0_sequence_id(row["utr"]) or row["sequence_hash"] != sequence_hash(row["utr"]):
                raise ValueError(f"Gate 0 sequence ID mismatch at {connection_key(row)}")
    step_pairs = {(row["library"], row["id"], row["generation"], row["sparse_step"], row["utr"]) for row in step_rows}
    if len(step_pairs) != EXPECTED_COUNTS["step"]:
        raise ValueError("Step generation and sparse-step join is not one-to-one")


def validate_gate0_rows(step_rows, target_rows):
    wanted = {row["raw_row_id"] for row in step_rows + target_rows}
    gate0 = {}
    with gzip.open(GATE0_SEQUENCE_INDEX, "rt", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["raw_row_id"] in wanted:
                gate0[row["raw_row_id"]] = row
    if set(gate0) != wanted:
        raise ValueError(f"Gate 0 raw_row_id coverage mismatch: {len(gate0)} != {len(wanted)}")
    for kind, rows in (("step", step_rows), ("target", target_rows)):
        for row in rows:
            expected = gate0[row["raw_row_id"]]
            assert expected["sequence_id"] == row["sequence_id"]
            assert expected["sequence_hash"] == row["sequence_hash"]
            assert expected["normalized_sequence"] == row["utr"]
            assert expected["source_row_number"] == row["source_row_number"]
            assert expected["raw_index"] == row["raw_index"]
            assert expected["source_library"] == row["library"]
            assert expected["source_id"] == row["id"]
            assert expected["measured_rl"] == row["measured_rl"]
            if kind == "step":
                assert expected["info1"] == row["generation"]
                assert expected["info2"] == row["sparse_step"]
            else:
                assert expected["info1"] == row["target_mrl_info1"]
                assert expected["info2"] == row["published_pred_info2"]
    return {"gate0_sequence_index": str(GATE0_SEQUENCE_INDEX), "step_rows_asserted": len(step_rows), "target_rows_asserted": len(target_rows), "raw_row_id_rule": "GSM3130443:row-<one-based-row-number-padded-to-7>", "sequence_id_rule": "sample2019_ + sha256(normalized_utr)", "per_row_assertions": True}


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
