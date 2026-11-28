#!/usr/bin/env python3
import argparse
import ast
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    code = Path(args.code)
    config = json.loads(Path(args.config).read_text())
    forbidden = ("test_labels.csv", "evaluator/test_pools.json")
    checks = {}
    for name in ("train_dev.py", "predict_test.py"):
        tree = ast.parse((code / name).read_text())
        strings = [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)]
        checks[f"{name}_no_evaluator_access"] = not any(token in value for value in strings for token in forbidden)
    train_strings = [node.value for node in ast.walk(ast.parse((code / "train_dev.py").read_text())) if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    checks["training_uses_train_and_dev"] = "roles/train.csv" in train_strings and "roles/dev.csv" in train_strings
    checks["training_cannot_read_test_predictions"] = not any("test_predictions.npz" in value for value in train_strings)
    predict_strings = [node.value for node in ast.walk(ast.parse((code / "predict_test.py").read_text())) if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    checks["prediction_uses_unlabeled_only"] = "roles/test_unlabeled.csv" in predict_strings and "DMS_score" not in predict_strings
    evaluator_text = (code / "evaluate_test.py").read_text()
    checks["single_opening_guard"] = "test_opened.json" in evaluator_text and "Permanent test was already opened" in evaluator_text
    checks["frozen_primary_and_budgets"] = config["primary_budget"] == 16 and config["budgets"] == [1, 4, 16, 64]
    checks["five_frozen_seeds"] = config["seeds"] == [11, 23, 37, 53, 71]
    report = {"scope": "pre-freeze static learner/evaluator interface audit", "checks": checks, "pass": all(checks.values())}
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
