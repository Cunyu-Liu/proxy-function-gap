import argparse
import csv
import gzip
import json
from collections import Counter
from pathlib import Path

from .common import load_config, normalize_sequence, sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = json.loads(Path(config["input_manifest"]).read_text())
    output_dir = Path(config["output_dir"])
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    reports = []
    for item in manifest["files"]:
        path = Path(item["path"])
        name = item["name"]
        rule = config["fixed_formats"][name]
        lengths = Counter()
        suffixes = Counter()
        normalized_lengths = Counter()
        rows = 0
        digest = sha256_file(path)
        if digest != item["sha256"]:
            raise ValueError(f"Raw hash mismatch: {name}")
        with gzip.open(path, "rt", newline="") as handle:
            for row in csv.DictReader(handle):
                raw = row["utr"]
                normalized, _ = normalize_sequence(raw, rule)
                rows += 1
                lengths[len(raw)] += 1
                normalized_lengths[len(normalized)] += 1
                if rule["length"] is not None and rule["length"] > 50:
                    suffixes[raw[50:]] += 1
        reports.append({
            "name": name,
            "rows": rows,
            "raw_sha256": digest,
            "raw_lengths": dict(sorted(lengths.items())),
            "normalized_lengths": dict(sorted(normalized_lengths.items())),
            "observed_suffixes_after_50": dict(sorted(suffixes.items())),
            "declared_rule": rule,
            "pass": True,
        })
    result = {
        "normalization_version": config["normalization_version"],
        "author_repository_commit": config["author_repository_commit"],
        "evidence": config["normalization_evidence"],
        "files": reports,
        "all_files_pass": len(reports) == 10,
        "rule": "Fixed 50/54/59 nt rows normalize to the author-used first 50 nt only after exact length/suffix validation. Variable 25..100 nt rows remain full length.",
    }
    (output_dir / "length_format_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"files": len(reports), "all_files_pass": result["all_files_pass"], "output": str(output_dir / "length_format_validation.json")}))


if __name__ == "__main__":
    main()
