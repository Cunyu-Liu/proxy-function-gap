import argparse
import json

from .paper3_replay import ReplayConfig, run_replay


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run Paper3 development/query repair-headroom replay")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    manifest = run_replay(ReplayConfig.from_json(args.config))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
