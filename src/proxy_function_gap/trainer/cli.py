import argparse
import json

from .config import TrainerConfig
from .data import preflight
from .engine import run_parallel_cuda_self_checks, run_parallel_training
from .synthetic import make_synthetic_pack


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fresh Sample-like CUDA trainer with fail-closed preflight")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "cuda-self-check", "train"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", required=True)
        if name == "train":
            command.add_argument("--execute-training", action="store_true")
    synthetic = subparsers.add_parser("make-synthetic-pack")
    synthetic.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    if args.command == "make-synthetic-pack":
        manifest = make_synthetic_pack(args.output_dir)
        print(json.dumps({"data_manifest": str(manifest), "synthetic": True, "scientific_result": False}))
        return 0
    config = TrainerConfig.from_json(args.config)
    if args.command == "preflight":
        print(json.dumps(preflight(config.data_manifest).to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "cuda-self-check":
        print(json.dumps(run_parallel_cuda_self_checks(config), indent=2, sort_keys=True))
        return 0
    if not args.execute_training:
        raise PermissionError("train requires explicit --execute-training")
    print(json.dumps(run_parallel_training(config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
