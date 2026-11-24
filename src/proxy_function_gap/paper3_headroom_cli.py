import argparse
import json

from .paper3_headroom import HeadroomConfig, run_study


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the frozen Paper3 development/query headroom study")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    summary = run_study(HeadroomConfig.from_json(args.config))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
