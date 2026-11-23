import argparse
import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence

from .metrics import regret
from .selection import Budget, Strategy, select
from .synthetic import make_known_error_case


def _config_hash(config) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def execute(config):
    case = make_known_error_case(config["synthetic_seed"])
    selections = []
    for strategy_name in config["strategies"]:
        result = select(
            pool=case.pool,
            strategy=Strategy(strategy_name),
            budget_limits=Budget(config["budget"]),
            rng=random.Random(config["selection_seed"]),
            alpha=config["alpha"],
            proxy=case.proxy,
            verifier=case.verifier,
            qhat=case.pool_qhat,
        )
        item = result.to_dict()
        item["synthetic_truth"] = case.synthetic_truth[result.selected_index]
        item["synthetic_regret"] = regret(
            case.synthetic_truth, result.selected_index, case.pool.feasible
        )
        selections.append(item)
    config_hash = _config_hash(config)
    return {
        "schema_version": 1,
        "run_id": config_hash[:16],
        "config_hash": config_hash,
        "config": config,
        "synthetic_case": {
            "sequence_ids": list(case.pool.sequence_ids),
            "sequences": list(case.pool.sequences),
            "proposal_probs": list(case.pool.proposal_probs),
            "feasible": list(case.pool.feasible),
            "constraints": asdict(case.constraints),
            "synthetic_truth": list(case.synthetic_truth),
            "structured_bias": list(case.structured_bias),
            "gaussian_noise": list(case.gaussian_noise),
            "proxy_values": list(case.proxy_values),
            "verifier_values": list(case.verifier_values),
            "qhat": case.qhat,
        },
        "selections": selections,
        "interpretation": "E0 software correctness only; synthetic values are not biological evidence.",
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the deterministic fixed-pool E0 synthetic smoke case")
    parser.add_argument("--synthetic-seed", type=int, default=314159)
    parser.add_argument("--selection-seed", type=int, default=271828)
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument(
        "--strategies", nargs="+", default=[strategy.value for strategy in Strategy], choices=[s.value for s in Strategy]
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    config = {
        "synthetic_seed": args.synthetic_seed,
        "selection_seed": args.selection_seed,
        "budget": args.budget,
        "alpha": args.alpha,
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
