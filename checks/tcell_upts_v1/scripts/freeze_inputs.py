from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from upts import load_json, save_json, sha256

B = Path("/teams/HoulabData_1788502323/liucunyu/proxy_function_gap")
REPO = Path("/home/proxy-function-gap")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_json(args.config)
    out = Path(config["artifact_root"])
    hashes_path = out / "input_hashes.json"
    if hashes_path.exists():
        raise SystemExit("input_hashes.json already exists; freeze is one-shot")

    aux = Path(config["auxiliary"]["arrays_root"])
    targets = []
    for ctx in ("n50_hek293t_r1", "defined_hepg2_r1"):
        for name in ("labels.npy", "cnn_predictions.npy", "member_predictions.npy", "split.npy", "prepare_manifest.json"):
            targets.append(aux / ctx / name)
    for d in config["auxiliary"]["directions"]:
        direction = f"{d['source']}_to_{d['destination']}"
        for name in ("dev_pools.npy", "dev_proposals.npy", "test_pools.npy", "test_proposals.npy", "selection.json", "gate.json"):
            targets.append(aux / direction / name)
    targets.append(REPO / "checks/tcell_residual_bstrict_v1/configs/bstrict_frozen.json")
    for name in ("analysis/results.json", "PROTOCOL.md", "frozen_config.json"):
        targets.append(B / "paper1_budget_matched_v1" / name)
    for ctx in ("defined_tcell_r1", "defined_tcell_r2"):
        scores_dir = B / "paper1_v3_strict_h1_retry1" / ctx / "cnn_scores"
        targets.extend(sorted(scores_dir.glob("*.json")))
        targets.append(B / "paper1_v3_strict_h1_retry1" / ctx / "context_manifest.json")
    targets.append(B / "data/derived/castillo_fixed_pools_v3_strict_h1_retry1/frozen_config.json")
    targets.append(B / "data/derived/castillo_fixed_pools_v3_strict_h1_retry1/build_manifest.json")

    hashes = {}
    for path in targets:
        if not path.exists():
            raise SystemExit(f"missing frozen input: {path}")
        hashes[str(path)] = sha256(path)
    save_json(hashes_path, hashes)

    budget = load_json(B / "paper1_budget_matched_v1/analysis/results.json")
    summaries = {(r["context"], r["cap"]): r for r in budget["context_summaries"]}
    thresholds = {}
    for ctx in ("defined_tcell_r1", "defined_tcell_r2"):
        baseline = float(summaries[(ctx, 16)]["bon_regret"])
        thresholds[ctx] = {
            "frozen_bon16_mean_regret": baseline,
            "min_absolute_reduction": max(0.1, 0.05 * baseline),
        }
    save_json(out / "frozen_thresholds.json", {
        "source": str(B / "paper1_budget_matched_v1/analysis/results.json"),
        "source_sha256": hashes[str(B / "paper1_budget_matched_v1/analysis/results.json")],
        "rule": "max(0.1 MRL, 5% of frozen baseline mean BoN16 regret), section 8.3",
        "thresholds": thresholds,
    })
    print(load_json(out / "frozen_thresholds.json"))
    print({"frozen_inputs": len(hashes), "config_sha256": sha256(args.config), "hashes_file": str(hashes_path)})


if __name__ == "__main__":
    main()