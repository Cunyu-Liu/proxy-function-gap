import argparse
import json

import numpy as np

from .inference import EnsembleInferenceConfig, run_ensemble_inference
from .pool_manifest import combine_pools, load_pool_manifest
from .production_pool import build_production_pool
from .production_selection import run_production_grid
from .q_head import QHeadConfig, fit_q_head, save_q_head_artifacts
from .selection_eval import SelectionRecord, run_budget_grid, selection_config_hash, write_selection_records
from .v_head import VHeadConfig, fit_v_head, save_v_head_artifacts


def _record(value):
    return SelectionRecord(
        pool_id=value["pool_id"],
        strategy=value["strategy"],
        alpha=value["alpha"],
        proposal_cap=value["proposal_cap"],
        seed=value["seed"],
        selected_index=value["selected_index"],
        selected_id=value["selected_id"],
        regret=value["regret"],
        proxy_item_forwards=value["proxy_item_forwards"],
        ensemble_member_item_forwards=value["ensemble_member_item_forwards"],
        member_item_forwards=value["member_item_forwards"],
        proposals=value["proposals"],
        abstain=value["abstain"],
        verifier_item_forwards=value.get("verifier_item_forwards", 0),
        v_model_item_forwards=value.get("v_model_item_forwards", 0),
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    ensemble = commands.add_parser("ensemble-inference")
    ensemble.add_argument("--config", required=True)
    preflight = commands.add_parser("pool-preflight")
    preflight.add_argument("--manifest", required=True)
    production = commands.add_parser("build-production-pool")
    production.add_argument("--source-pool-manifest", required=True)
    production.add_argument("--pool-id", required=True)
    production.add_argument("--prediction-manifest", required=True)
    production.add_argument("--output-dir", required=True)
    production_grid = commands.add_parser("production-grid")
    production_grid.add_argument("--manifest", required=True)
    production_grid.add_argument("--inference-config", required=True)
    production_grid.add_argument("--v-artifacts", required=True)
    production_grid.add_argument("--output", required=True)
    grid = commands.add_parser("synthetic-grid")
    grid.add_argument("--manifest", required=True)
    grid.add_argument("--output", required=True)
    fit_v = commands.add_parser("fit-v")
    fit_v.add_argument("--dev", nargs="+", required=True)
    fit_v.add_argument("--calibration", nargs="+", required=True)
    fit_v.add_argument("--output-dir", required=True)
    fit_q = commands.add_parser("fit-q")
    fit_q.add_argument("--selection", nargs="+", required=True)
    fit_q.add_argument("--dev", nargs="+", required=True)
    fit_q.add_argument("--calibration", nargs="+", required=True)
    fit_q.add_argument("--output-dir", required=True)
    fit_q.add_argument("--strategy", default="fp-rej")
    fit_q.add_argument("--alpha", type=float, default=5)
    fit_q.add_argument("--proposal-cap", type=int, default=4)
    fit_q.add_argument("--seed", type=int, default=101)
    args = parser.parse_args(argv)
    if args.command == "ensemble-inference":
        print(json.dumps(run_ensemble_inference(EnsembleInferenceConfig.from_json(args.config)), indent=2, sort_keys=True))
        return 0
    if args.command == "pool-preflight":
        pool = load_pool_manifest(args.manifest)
        print(json.dumps({"pool_id": pool.pool_id, "role": pool.pool_role, "rows": len(pool), "manifest_sha256": pool.manifest_sha256, "synthetic": pool.synthetic, "outer_labels_read": False}, indent=2))
        return 0
    if args.command == "build-production-pool":
        result = build_production_pool(args.source_pool_manifest, args.pool_id, args.prediction_manifest, args.output_dir)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "production-grid":
        pool = load_pool_manifest(args.manifest)
        inference = EnsembleInferenceConfig.from_json(args.inference_config)
        config_hash = run_production_grid(pool, inference, args.v_artifacts, args.output)
        print(json.dumps({"selection_config_sha256": config_hash, "pool_id": pool.pool_id, "actual_cuda_inference": True, "scientific_result": False}))
        return 0
    if args.command == "synthetic-grid":
        pool = load_pool_manifest(args.manifest)
        if not pool.synthetic:
            raise PermissionError("synthetic-grid is restricted to synthetic pools")
        member_scorers = tuple(
            lambda candidate, member=member: pool.member_predictions[member, candidate.index]
            for member in range(3)
        )
        verifier_values = np.clip(1 - np.abs(pool.proxy_mean - pool.measured), 0, 1)
        records = run_budget_grid(pool, member_scorers, verifier_values)
        config_hash = write_selection_records(records, args.output)
        print(json.dumps({"selection_config_sha256": config_hash, "records": len(records), "synthetic": True, "scientific_result": False}))
        return 0
    if args.command == "fit-v":
        dev = combine_pools([load_pool_manifest(path) for path in args.dev], "combined_dev")
        calibration = combine_pools([load_pool_manifest(path) for path in args.calibration], "combined_calibration")
        result = fit_v_head(dev, calibration, VHeadConfig())
        print(json.dumps(save_v_head_artifacts(result, args.output_dir), indent=2, sort_keys=True))
        return 0
    records = []
    selection_sources = []
    for path in args.selection:
        selection = json.loads(open(path).read())
        current = [_record(value) for value in selection["records"]]
        if selection_config_hash(current) != selection["selection_config_sha256"]:
            raise ValueError("selection manifest hash mismatch")
        if selection.get("execution_metadata", {}).get("outer_labels_read"):
            raise PermissionError("Q cannot use outer selection results")
        records.extend(current)
        from .trainer.data import sha256_file
        selection_sources.append({"path": path, "sha256": sha256_file(path)})
    identities = [(item.pool_id, item.strategy, item.alpha, item.proposal_cap, item.seed) for item in records]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate selection records across input files")
    frozen_selection_hash = selection_config_hash(records)
    def examples(paths):
        output = []
        for path in paths:
            pool = load_pool_manifest(path)
            matching = [
                record for record in records
                if record.pool_id == pool.pool_id
                and record.strategy == args.strategy
                and record.alpha == args.alpha
                and record.proposal_cap == args.proposal_cap
                and record.seed == args.seed
            ]
            if len(matching) != 1:
                raise ValueError(f"expected one frozen selection for {pool.pool_id}")
            output.append((pool, matching[0]))
        return output
    result = fit_q_head(examples(args.dev), examples(args.calibration), QHeadConfig(), frozen_selection_hash)
    result["selection_sources"] = selection_sources
    print(json.dumps(save_q_head_artifacts(result, args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    main()
