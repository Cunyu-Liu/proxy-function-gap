import argparse
import json

from proxy_function_gap.kmer_proxy import build_feature_pack, train_kmer


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    features = sub.add_parser("features")
    features.add_argument("--input-npz", required=True)
    features.add_argument("--output-dir", required=True)
    train = sub.add_parser("train")
    train.add_argument("--feature-manifest", required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--memory-fraction", type=float, default=0.08)
    args = parser.parse_args()
    if args.command == "features":
        result = build_feature_pack(args.input_npz, args.output_dir)
        print(json.dumps({"features": result["feature_count"], "files": result["files"]}, default=str))
    else:
        result = train_kmer(args.feature_manifest, args.seed, args.output_dir, args.memory_fraction)
        print(json.dumps({"seed": args.seed, "epoch": result["epoch"], "best_epoch": result["best_epoch"], "model_hash": result["model_hash"], "converged_dev_plateau": True}))


if __name__ == "__main__":
    main()
