import argparse
import json
from pathlib import Path

from .paper2_baselines import evaluate_pool, freeze_dev_config, frozen_methods, load_role_pool, run_pool_baselines, summarize_evaluations
from .paper2_com import COMConfig, run_com_training, verify_com_run


def main(argv=None):
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("com-train")
    train.add_argument("--config", required=True)
    train.add_argument("--execute-training", action="store_true")
    verify = commands.add_parser("com-verify")
    verify.add_argument("--config", required=True)
    verify.add_argument("--output", required=True)
    select = commands.add_parser("select")
    select.add_argument("--config", required=True)
    select.add_argument("--role", required=True)
    select.add_argument("--index", type=int, required=True)
    select.add_argument("--output", required=True)
    select.add_argument("--frozen-config")
    select.add_argument("--permutations", type=int, default=0)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--role", required=True)
    evaluate.add_argument("--index", type=int, required=True)
    evaluate.add_argument("--selection", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--label-path")
    freeze = commands.add_parser("freeze-dev")
    freeze.add_argument("--evaluations", nargs="+", required=True)
    freeze.add_argument("--source-config", required=True)
    freeze.add_argument("--output", required=True)
    summarize = commands.add_parser("summarize")
    summarize.add_argument("--evaluations", nargs="+", required=True)
    summarize.add_argument("--output", required=True)
    summarize.add_argument("--permutations", type=int, default=32)
    args = parser.parse_args(argv)
    if args.command == "com-train":
        if not args.execute_training:
            raise PermissionError("COM training requires --execute-training")
        result = run_com_training(COMConfig.from_json(args.config))
    elif args.command == "com-verify":
        result = verify_com_run(COMConfig.from_json(args.config))
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    elif args.command == "select":
        config = json.loads(Path(args.config).read_text())
        if args.role.startswith("outer_"):
            if not args.frozen_config:
                raise PermissionError("outer baseline selection requires frozen dev hyperparameters")
            frozen = json.loads(Path(args.frozen_config).read_text())
            if frozen["outer_used_for_selection"]:
                raise PermissionError("invalid hyperparameter provenance")
            config["methods"] = frozen_methods(frozen)
        pool = load_role_pool(config, args.role, args.index)
        result = run_pool_baselines(pool, config, args.output, permutations=args.permutations)
    elif args.command == "evaluate":
        config = json.loads(Path(args.config).read_text())
        pool = load_role_pool(config, args.role, args.index)
        result = evaluate_pool(pool, args.selection, args.output, label_path=args.label_path)
    elif args.command == "freeze-dev":
        result = freeze_dev_config(args.evaluations, args.output, args.source_config)
    else:
        result = summarize_evaluations(args.evaluations, args.output, permutations=args.permutations)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
