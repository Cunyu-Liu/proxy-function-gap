import argparse
import json

from .paper3_outer import OuterProtocolConfig, run_calibration_dry_round


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run a Paper3 calibration-only separated-Q/E dry round")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    summary = run_calibration_dry_round(OuterProtocolConfig.from_json(args.config), args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
