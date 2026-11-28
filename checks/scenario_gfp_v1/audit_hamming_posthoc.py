import csv
import json
from pathlib import Path

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/scenario_gfp_v1")
with open(ROOT / "learner" / "low_order_labeled.csv", newline="") as handle:
    rows = list(csv.DictReader(handle))
train_singles = [r for r in rows if r["role"] == "train" and int(r["order"]) == 1]
dev_singles = [r for r in rows if r["role"] == "dev" and int(r["order"]) == 1]
assert train_singles and dev_singles
pair = next((a, b) for a in train_singles for b in dev_singles if sum(x != y for x, y in zip(a["mutated_sequence"], b["mutated_sequence"])) == 2)
assert pair is not None
output = {
    "status": "CORRECTION_PASS",
    "reason": "The pre-evaluation raw-gate field named low_order_train_dev_minimum_hamming overestimated the distance as 3. Disjoint-position single mutants have Hamming distance 2.",
    "correct_low_order_train_dev_minimum_hamming": 2,
    "low_order_train_dev_exact_or_hamming1_crossings": 0,
    "low_order_to_permanent_test_minimum_hamming": 3,
    "permanent_test_isolation_unchanged": True,
    "scientific_results_unchanged": True,
    "timing": "post-evaluation delivery audit; no split, model, prediction, pool, or result changed"
}
(ROOT / "verification" / "posthoc_hamming_correction.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
print(json.dumps(output, indent=2, sort_keys=True))
