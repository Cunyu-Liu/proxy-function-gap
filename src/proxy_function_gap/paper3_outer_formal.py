import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .inference import EnsembleInferenceConfig
from .outer_evaluator import _labels
from .outer_learner import load_outer_pool
from .paper3_headroom import FrozenScaler, raw_representation
from .paper3_headroom_v4 import STRATEGIES
from .paper3_outer import (
    OuterProtocolConfig, cluster_bootstrap_mean, learner_curve,
    require_development_isolation, score_frozen_curve, split_group_indices, subset_pool,
)
from .pool_manifest import combine_pools, load_pool_manifest
from .production_selection import FreshMemberScorer
from .regret_acquisition import QueryLabelVault, UnlabeledPool
from .trainer.checkpoint import atomic_json_write
from .trainer.data import sha256_file

SOURCE_POOL_MANIFEST = Path("/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/data/derived/fresh_proxy_fixed_pools_v2/build_a/pool_manifest.json")
INFERENCE_CONFIG = Path("/home/proxy-function-gap-paper3-outer/configs/ensemble_inference_v2.json")
PAPER3_OUTER_HARD_LOCK = True


def require_release(path, config):
    if PAPER3_OUTER_HARD_LOCK:
        raise PermissionError("Paper3 outer is hard locked because Paper2 P0 calibration gate failed")
    release = json.loads(Path(path).read_text())
    if (release.get("paper2_p0_pass") is not True
            or release.get("paper3_outer_authorized") is not True
            or release.get("protocol_config_sha256") != config.sha256()):
        raise PermissionError("explicit Paper2 P0 release for this exact Paper3 protocol is required")
    if release.get("outer_labels_may_be_scored") is not True:
        raise PermissionError("release does not authorize one frozen outer evaluation")
    return release


def score_outer_proxy(pool, inference_config):
    members = [FreshMemberScorer(member, pool, inference_config) for member in inference_config.members]
    values = np.asarray([[member.score_index(index) for member in members] for index in range(len(pool))])
    return replace(pool, proxy_mean=values.mean(axis=1), proxy_std=values.std(axis=1, ddof=1),
                   proxy_range=values.max(axis=1) - values.min(axis=1))


def run_released_outer(config, release_path, output_dir):
    release = require_release(release_path, config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    source = config.v4()
    inference = EnsembleInferenceConfig.from_json(INFERENCE_CONFIG)
    development = combine_pools([load_pool_manifest(source.pool_root / item / "pool_manifest.json")
                                 for item in source.development_pool_ids], "combined_development")
    development_raw, _ = raw_representation(config.representation, development, source)
    scaler = FrozenScaler.fit(development_raw, development.measured)
    development_x = scaler.features(development_raw)
    development_y = scaler.labels(development.measured)
    runtime = config.runtime(output_dir)
    source_manifest = json.loads(SOURCE_POOL_MANIFEST.read_text())
    vault_reference = json.loads((SOURCE_POOL_MANIFEST.parent / "evaluator" / "vault_references.json").read_text())
    completed = []
    for pool_id in (*config.primary_outer_pool_ids, *config.secondary_outer_pool_ids):
        pool = score_outer_proxy(load_outer_pool(SOURCE_POOL_MANIFEST, pool_id), inference)
        query_indices, evaluation_indices = split_group_indices(pool, config.split_seed)
        query = subset_pool(pool, query_indices, pool_id + "_query")
        evaluation = subset_pool(pool, evaluation_indices, pool_id + "_evaluation")
        require_development_isolation(development, query, evaluation)
        role = pool.pool_role
        vault = vault_reference[role]
        if vault["holder"] != "Evaluator" or sha256_file(vault["path"]) != vault["sha256"]:
            raise PermissionError("outer vault holder or hash mismatch")
        labels_by_sequence = _labels(vault["path"])
        query_labels = np.asarray([labels_by_sequence[item] for item in query.sequence_ids])
        evaluation_labels = np.asarray([labels_by_sequence[item] for item in evaluation.sequence_ids])
        query_pool = UnlabeledPool(query.row_ids, scaler.features(raw_representation(config.representation, query, source)[0]))
        evaluation_pool = UnlabeledPool(evaluation.row_ids, scaler.features(raw_representation(config.representation, evaluation, source)[0]))
        query_vault = QueryLabelVault(query_pool, scaler.labels(query_labels))
        curves = []
        for random_seed in source.random_seeds:
            curves.append(learner_curve(development_x, development_y, query_pool, query_vault,
                                        evaluation_pool, query.proxy_mean, query.utr50,
                                        "random", config, runtime, random_seed))
        for offset, strategy in enumerate(STRATEGIES[1:], start=1):
            curves.append(learner_curve(development_x, development_y, query_pool, query_vault,
                                        evaluation_pool, query.proxy_mean, query.utr50,
                                        strategy, config, runtime, source.acquisition_seed + offset))
        pool_dir = output_dir / pool_id
        common = {"config_sha256": config.sha256(), "pool_id": pool_id, "pool_role": role,
                  "query_ids": list(query.row_ids), "evaluation_ids": list(evaluation.row_ids),
                  "release_protocol_config_sha256": release["protocol_config_sha256"]}
        atomic_json_write({**common, "curves": curves, "evaluation_labels_returned_to_learner": False},
                          pool_dir / "learner_frozen.json")
        scored = [score_frozen_curve(curve, evaluation.row_ids, evaluation_labels) for curve in curves]
        atomic_json_write({**common, "curves": scored, "labels_returned_to_learner": False},
                          pool_dir / "evaluator_final.json")
        completed.append(pool_id)
        atomic_json_write({"config_sha256": config.sha256(), "completed_pool_ids": completed,
                           "total_pool_count": 19, "status": "IN_PROGRESS"}, output_dir / "summary.json")
    primary_differences = []
    for pool_id in config.primary_outer_pool_ids:
        payload = json.loads((output_dir / pool_id / "evaluator_final.json").read_text())
        ei_curve = next(curve for curve in payload["curves"] if curve["strategy"] == "expected_improvement")
        ei = next(record["regret"] for record in ei_curve["records"] if record["budget"] == config.primary_budget)
        random_curves = [curve for curve in payload["curves"] if curve["strategy"] == "random"]
        random = float(np.mean([next(record["regret"] for record in curve["records"] if record["budget"] == config.primary_budget)
                                for curve in random_curves]))
        primary_differences.append(ei - random)
    primary = cluster_bootstrap_mean(primary_differences, config.bootstrap_resamples,
                                     config.bootstrap_seed, config.bootstrap_confidence_level)
    atomic_json_write({"schema_version": "paper3-outer-primary-result-v1",
                       "comparison": "expected_improvement_minus_random_at_16",
                       "pool_ids": list(config.primary_outer_pool_ids),
                       "paired_differences": primary_differences, "cluster_bootstrap": primary,
                       "primary_hypothesis_supported": primary["success_upper_below_zero"]},
                      output_dir / "primary_analysis.json")
    atomic_json_write({"config_sha256": config.sha256(), "completed_pool_ids": completed,
                       "total_pool_count": 19, "status": "COMPLETED_ONE_FROZEN_OUTER_REPLAY",
                       "primary_hypothesis_supported": primary["success_upper_below_zero"],
                       "labels_returned_to_learner": False}, output_dir / "summary.json")
    return {"status": "COMPLETED_ONE_FROZEN_OUTER_REPLAY", "pool_count": len(completed)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run Paper3 outer only with an exact P0 release; not used during preparation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = run_released_outer(OuterProtocolConfig.from_json(args.config), args.release, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
