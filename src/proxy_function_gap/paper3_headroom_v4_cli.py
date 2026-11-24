import argparse
import json

from .paper3_headroom_v4 import HeadroomV4Config, run_study_v4


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the frozen Paper3 V4 analytic regret acquisition study")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    summary = run_study_v4(HeadroomV4Config.from_json(args.config), args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
