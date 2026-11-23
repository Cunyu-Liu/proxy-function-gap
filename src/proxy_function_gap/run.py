import argparse
import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence

from .budget import Aggregation, Budget, ScoreCache
from .metrics import regret
from .selection import Strategy, ZeroWeightPolicy, select_trajectories
from .synthetic import SyntheticScenario, make_known_error_case


def _config_hash(config) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def execute(config):
    case = make_known_error_case(config["synthetic_seed"], config["scenario"])
    selections = []
    for strategy_name in config["strategies"]:
        result = select_trajectories(
            pool=case.pool,
            strategy=Strategy(strategy_name),
            proposal_cap=config["proposal_cap"],
            trajectories=config["trajectories"],
            top_k=config["top_k"],
            budget_limits=Budget(),
            rng=random.Random(config["selection_seed"]),
            alpha=config["alpha"],
            aggregation=Aggregation(config["aggregation"]),
            zero_weight_policy=ZeroWeightPolicy(config["zero_weight_policy"]),
            proxy=case.proxy,
            verifier=case.verifier,
            qhat=case.pool_qhat,
            abstain_below_q=config["abstain_below_q"],
            cache=ScoreCache() if config["cache"] else None,
        )
        item = result.to_dict()
        item["synthetic_truth"] = [
            case.synthetic_truth[index] for index in result.selected_indices
        ]
        item["synthetic_regret"] = [
            regret(case.synthetic_truth, index, case.pool.feasible)
            for index in result.selected_indices
        ]
        selections.append(item)
    config_hash = _config_hash(config)
    return {
        "schema_version": 2,
        "method_family": "VGSC-inspired fixed-pool",
        "run_id": config_hash[:16],
        "config_hash": config_hash,
        "config": config,
        "synthetic_case": {
            "scenario": case.scenario.value,
            "sequence_ids": list(case.pool.sequence_ids),
            "sequences": list(case.pool.sequences),
            "proposal_probs": list(case.pool.proposal_probs),
            "feasible": list(case.pool.feasible),
            "constraints": asdict(case.constraints),
            "synthetic_truth": list(case.synthetic_truth),
            "proxy_signal": list(case.proxy_signal),
            "structured_bias": list(case.structured_bias),
            "synthetic_proxy_noise": list(case.synthetic_proxy_noise),
            "proxy_values": list(case.proxy_values),
            "verifier_values": list(case.verifier_values),
            "qhat": case.qhat,
        },
        "selections": selections,
        "interpretation": "E0 software correctness only; synthetic values are not biological evidence.",
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run VGSC-inspired fixed-pool E0 synthetic checks")
    parser.add_argument("--synthetic-seed", type=int, default=314159)
    parser.add_argument("--selection-seed", type=int, default=271828)
    parser.add_argument("--proposal-cap", "--budget", dest="proposal_cap", type=int, default=8)
    parser.add_argument("--trajectories", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--aggregation", choices=[item.value for item in Aggregation], default="product")
    parser.add_argument(
        "--zero-weight-policy", choices=[item.value for item in ZeroWeightPolicy], default="abstain"
    )
    parser.add_argument("--scenario", choices=[item.value for item in SyntheticScenario], default="adversarial")
    parser.add_argument("--abstain-below-q", type=float)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=[strategy.value for strategy in Strategy],
        choices=[strategy.value for strategy in Strategy],
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    config = {
        "synthetic_seed": args.synthetic_seed,
        "selection_seed": args.selection_seed,
        "proposal_cap": args.proposal_cap,
        "trajectories": args.trajectories,
        "top_k": args.top_k,
        "alpha": args.alpha,
        "aggregation": args.aggregation,
        "zero_weight_policy": args.zero_weight_policy,
        "scenario": args.scenario,
        "abstain_below_q": args.abstain_below_q,
        "cache": args.cache,
        "strategies": list(args.strategies),
    }
    report = execute(config)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
