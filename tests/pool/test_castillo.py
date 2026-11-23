import csv
import gzip
import json

import numpy as np

from proxy_function_gap.castillo import build_castillo_context, evaluate_castillo_pool, load_castillo_pool
from proxy_function_gap.selection_eval import SelectionRecord, selection_config_hash


def test_castillo_pool_builder_and_evaluator(tmp_path):
    raw = tmp_path / "raw.csv.gz"
    sequences = ["A" * 50, "C" * 50, "G" * 50, "T" * 50, "AC" * 25, "GT" * 25]
    with gzip.open(raw, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["UTR", "rl"])
        writer.writeheader()
        for index, sequence in enumerate(sequences):
            writer.writerow({"UTR": sequence, "rl": index + 1})
    sample = tmp_path / "sample.tsv.gz"
    with gzip.open(sample, "wt", newline="") as handle:
        handle.write("utr50\n")
    train = tmp_path / "train.npz"
    np.savez(train, utr50=np.asarray(["A" * 50]))
    output = tmp_path / "context"
    manifest = build_castillo_context(raw, "test", [sample], train, output, pool_count=1, pool_size=2)
    pool_id = next(iter(manifest["pool_hashes"]))
    pool = load_castillo_pool(output, pool_id, train)
    assert len(pool) == 2 and pool.measured is None
    assert all("measured_rl" not in row for row in [json.loads(line) for line in gzip.open(output / "learner" / f"{pool_id}.jsonl.gz", "rt")])
    record = SelectionRecord(pool_id=pool_id, strategy="bon", alpha=0, proposal_cap=1, seed=101, selected_index=0, selected_id=pool.sequence_ids[0], regret=None, proxy_item_forwards=1, ensemble_member_item_forwards=3, member_item_forwards={11: 1, 23: 1, 37: 1}, proposals=1, abstain=False)
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"selection_config_sha256": selection_config_hash([record]), "execution_metadata": {"outer_labels_read": False}, "records": [record.to_dict()]}))
    result = evaluate_castillo_pool(output, pool_id, selection, tmp_path / "evaluation.json")
    assert result["labels_returned_to_learner"] is False
    evaluated = json.loads((tmp_path / "evaluation.json").read_text())["records"][0]
    assert evaluated["regret"] >= 0


def test_right_half_hamming_one_is_blocked():
    from collections import defaultdict
    from proxy_function_gap.castillo import _near_existing, _add_half_index
    first = "CAAGAGGCAGGGCGACGGGCGGGATGAACCGGGAGGGCTGGGGATCAACA"
    second = "CAAGAGGCAGGGCGACGGGCGGCATGAACCGGGAGGGCTGGGGATCAACA"
    index = defaultdict(list)
    _add_half_index(first, index)
    assert _near_existing(second, index)
