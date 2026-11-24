import argparse
import os
import urllib.request
import zipfile
from pathlib import Path

from .common import load_config, sha256_file, utc_now, write_json
from .build import source_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--key", action="append", help="Optional source key; default validates all without downloading")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    keys = args.key or list(config["inputs"])
    records = []
    for key in keys:
        item = config["inputs"][key]
        path = source_path(config, key)
        if not path.exists():
            if not args.download:
                raise FileNotFoundError(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".download")
            if temporary.exists():
                raise FileExistsError(temporary)
            request = urllib.request.Request(item["url"], headers={"User-Agent": "proxy-function-gap-gate0/1.0"})
            if "archive_member" in item:
                archive_temporary = path.with_name(path.name + ".archive.zip")
                if archive_temporary.exists():
                    raise FileExistsError(archive_temporary)
                with urllib.request.urlopen(request, timeout=120) as response, open(archive_temporary, "xb") as handle:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        handle.write(block)
                with zipfile.ZipFile(archive_temporary) as archive, archive.open(item["archive_member"]) as source, open(temporary, "xb") as handle:
                    while True:
                        block = source.read(1024 * 1024)
                        if not block:
                            break
                        handle.write(block)
            else:
                with urllib.request.urlopen(request, timeout=120) as response, open(temporary, "xb") as handle:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        handle.write(block)
            if sha256_file(temporary) != item["sha256"] or temporary.stat().st_size != item["bytes"]:
                raise ValueError(f"Downloaded source failed size/SHA256 validation: {key}; temporary file retained")
            os.replace(temporary, path)
            if "archive_member" in item:
                archive_temporary.unlink()
        digest = sha256_file(path)
        if digest != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise ValueError(f"Existing source failed size/SHA256 validation: {key}")
        records.append({"key": key, "path": str(path), "url": item["url"], "bytes": item["bytes"], "sha256": digest, "license": item["license"]})
    report = {"dataset_id": config["dataset_id"], "validated_at": utc_now(), "status": "PASS", "sources": records}
    destination = Path(config["raw_root"]) / "acquisition_validation.json"
    write_json(destination, report)
    print(destination)


if __name__ == "__main__":
    main()
