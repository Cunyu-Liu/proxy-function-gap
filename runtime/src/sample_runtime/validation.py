import math

import h5py
import numpy as np

from .data import connection_key, validate_connections
from .provenance import EXPECTED_COUNTS, ORIGINAL_WEIGHTS, RETRAINED_WEIGHTS, SCALER, TARGET_VALIDATION_AFFINE


def validate_target_info2(target_rows, tolerance=5e-4):
    published = np.array([float(row["published_pred_info2"]) for row in target_rows])
    raw = np.array([float(row["original_raw_output"]) for row in target_rows])
    reconstructed = TARGET_VALIDATION_AFFINE["mean"] + TARGET_VALIDATION_AFFINE["scale"] * raw
    errors = np.abs(published - reconstructed)
    result = {
        "n": len(target_rows),
        "max_abs": float(errors.max()),
        "mean_abs": float(errors.mean()),
        "p99_abs": float(np.quantile(errors, 0.99)),
        "tolerance": tolerance,
        "info2_role": "published prediction used only for numerical validation",
    }
    if not np.isfinite(errors).all() or errors.max() > tolerance:
        raise ValueError(f"Target info2 validation failed: {result}")
    return result


def _arrays(path):
    output = {}
    with h5py.File(path, "r") as handle:
        weights = handle["model_weights"]
        for layer in ("convolution1d_1", "convolution1d_2", "dense_1", "dense_2"):
            for kind in ("kernel", "bias"):
                suffixes = (f"/{kind}:0", "_W:0" if kind == "kernel" else "_b:0")
                found = []
                def visit(name, obj):
                    if isinstance(obj, h5py.Dataset) and name.startswith(layer + "/") and any(s in name for s in suffixes):
                        found.append(obj[...])
                weights.visititems(visit)
                if len(found) != 1:
                    raise ValueError(f"Independent reader expected one {layer}/{kind}")
                output[layer + kind] = found[0]
    return output


def _encode(utr):
    mapping = {"A": [1, 0, 0, 0], "C": [0, 1, 0, 0], "G": [0, 0, 1, 0], "T": [0, 0, 0, 1]}
    return np.array([mapping[base] for base in utr + "ATGG"], dtype=np.float64)


def _conv(inputs, weight, bias):
    if weight.ndim == 4:
        weight = weight[:, 0, :, :]
    output = np.empty((inputs.shape[0] - weight.shape[0] + 1, weight.shape[2]), dtype=np.float64)
    for position in range(output.shape[0]):
        output[position] = np.einsum("ki,kio->o", inputs[position:position + weight.shape[0]], weight) + bias
    return np.maximum(output, 0)


def _forward(arrays, utr):
    x = _conv(_encode(utr), arrays["convolution1d_1kernel"], arrays["convolution1d_1bias"])
    x = _conv(x, arrays["convolution1d_2kernel"], arrays["convolution1d_2bias"])
    x = x.reshape(-1).dot(arrays["dense_1kernel"]) + arrays["dense_1bias"]
    x = np.maximum(x, 0)
    return float(x.dot(arrays["dense_2kernel"])[0] + arrays["dense_2bias"][0])


def independent_sample_check(step_rows, target_rows, samples_per_group=8, tolerance=5e-4):
    arrays = {"original": _arrays(ORIGINAL_WEIGHTS), "retrained": _arrays(RETRAINED_WEIGHTS)}
    comparisons = []
    for group, rows in (("step", step_rows), ("target", target_rows)):
        indices = np.linspace(0, len(rows) - 1, num=samples_per_group, dtype=int)
        for index in indices:
            row = rows[int(index)]
            item = {"group": group, "key": list(connection_key(row))}
            for model_name in ("original", "retrained"):
                expected = _forward(arrays[model_name], row["utr"])
                actual = float(row[f"{model_name}_raw_output"])
                item[f"{model_name}_abs"] = abs(expected - actual)
            comparisons.append(item)
    maximum = max(max(item["original_abs"], item["retrained_abs"]) for item in comparisons)
    result = {"samples_per_group": samples_per_group, "n_comparisons": len(comparisons), "max_abs": maximum, "tolerance": tolerance, "comparisons": comparisons}
    if not math.isfinite(maximum) or maximum > tolerance:
        raise ValueError(f"Independent NumPy sample validation failed: max_abs={maximum}")
    return result


def validate_prediction_tables(step_rows, target_rows):
    validate_connections(step_rows, target_rows)
    for name, rows in (("step", step_rows), ("target", target_rows)):
        if len(rows) != EXPECTED_COUNTS[name]:
            raise ValueError(f"{name} row conservation failed")
        for row in rows:
            for model_name in ("original", "retrained"):
                raw = float(row[f"{model_name}_raw_output"])
                predicted = float(row[f"{model_name}_pred_mrl"])
                expected = SCALER["mean"] + SCALER["scale"] * raw
                if not math.isfinite(raw) or not math.isfinite(predicted) or abs(predicted - expected) > 1e-10:
                    raise ValueError(f"Invalid {model_name} prediction at {connection_key(row)}")
    return {"step_rows": len(step_rows), "target_rows": len(target_rows), "step_key": "(library,id,sparse_step,utr)", "target_key": "(library,id,empty sparse_step,utr)", "one_to_one": True, "row_conservation": True, "scaler_ddof": 0}
