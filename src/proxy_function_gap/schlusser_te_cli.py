import argparse
import json

from .schlusser_te import execute, freeze_protocol, verify_delivery


def main():
    parser = argparse.ArgumentParser(description="Frozen one-time Schlusser HepG2 MRL-to-TE confirmatory evaluation")
    parser.add_argument("command", choices=["freeze", "execute", "verify"])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze_protocol(args.config)
    elif args.command == "execute":
        result = execute(args.config)
    else:
        result = verify_delivery(args.config)
    print(json.dumps(result, sort_keys=True))
    if args.command == "verify" and not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
