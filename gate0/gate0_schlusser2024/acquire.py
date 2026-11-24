import argparse
import concurrent.futures
import hashlib
import http.client
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from .common import file_record, load_config, md5_file, write_json

USER_AGENT = "proxy-function-gap-schlusser2024-gate0/1.0"


def request_bytes(url, byte_range=None, timeout=180, attempts=10):
    headers = {"User-Agent": USER_AGENT}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if byte_range is not None and response.status != 206:
                    raise RuntimeError(f"Range request was not honored: {url}")
                return response.read()
        except (urllib.error.URLError, TimeoutError, RuntimeError, http.client.IncompleteRead, ConnectionError) as error:
            if attempt + 1 == attempts:
                raise
            retry_after = getattr(error, "headers", {}).get("Retry-After") if getattr(error, "headers", None) else None
            time.sleep(min(90, int(retry_after) if retry_after else 5 * (attempt + 1)))
    raise AssertionError("unreachable")


def download_ranged(spec, destination, chunk_bytes, workers):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == spec["bytes"] and md5_file(destination) == spec["md5"]:
        return file_record(destination, {"md5": spec["md5"], "download": "already_verified"})
    temporary = destination.with_suffix(destination.suffix + f".parts_{chunk_bytes}")
    temporary.mkdir(exist_ok=True)
    ranges = [(start, min(start + chunk_bytes, spec["bytes"]) - 1) for start in range(0, spec["bytes"], chunk_bytes)]

    def fetch(item):
        index, bounds = item
        path = temporary / f"{index:06d}.part"
        expected = bounds[1] - bounds[0] + 1
        if not path.exists() or path.stat().st_size != expected:
            payload = request_bytes(spec["url"], bounds)
            if len(payload) != expected:
                raise RuntimeError(f"Short range for {spec['name']}: {len(payload)} != {expected}")
            path.write_bytes(payload)
        return path

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        paths = list(executor.map(fetch, enumerate(ranges)))
    assembly = destination.with_suffix(destination.suffix + ".assembling")
    digest = hashlib.md5()
    with assembly.open("wb") as output:
        for path in paths:
            payload = path.read_bytes()
            output.write(payload)
            digest.update(payload)
    if assembly.stat().st_size != spec["bytes"] or digest.hexdigest() != spec["md5"]:
        raise RuntimeError(f"Checksum mismatch for {spec['name']}")
    os.replace(assembly, destination)
    for path in paths:
        path.unlink()
    temporary.rmdir()
    return file_record(destination, {"md5": spec["md5"], "download": "ranged_complete"})


def acquire(config_path):
    config = load_config(config_path)
    raw = Path(config["raw_root"])
    raw.mkdir(parents=True, exist_ok=True)
    records = []
    for spec in config["zenodo_files"]:
        record = download_ranged(spec, raw / spec["name"], config["chunk_bytes"], config["download_workers"])
        record.update({"name": spec["name"], "study": spec["study"], "url": spec["url"], "license": "CC-BY-4.0", "zenodo_record": "10463061"})
        records.append(record)
        write_json(raw / "download_progress.json", {"complete": records, "pending": [x["name"] for x in config["zenodo_files"] if x["name"] not in {r["name"] for r in records}]})
    for spec in config.get("metadata_downloads", []):
        payload = request_bytes(spec["url"], timeout=90)
        path = raw / spec["name"]
        path.write_bytes(payload)
        records.append(file_record(path, {"url": spec["url"], "role": spec["role"]}))
    for spec in config.get("gitlab_files", []):
        payload = request_bytes(spec["url"], timeout=90)
        path = raw / "upstream" / spec["name"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        records.append(file_record(path, {"url": spec["url"], "role": "MIT licensed upstream code", "commit": config["upstream_commit"]}))
    manifest = {
        "schema_version": "1.0",
        "dataset": "schlusser2024-gate0-v1",
        "upstream_commit": config["upstream_commit"],
        "licenses": config["licenses"],
        "files": records,
        "all_zenodo_md5_verified": all(md5_file(raw / spec["name"]) == spec["md5"] for spec in config["zenodo_files"]),
        "raw_fastq_downloaded": False,
        "raw_fastq_reason": "Published sequence-level tables are used; large FASTQ files remain referenced by ENA metadata and were not independently reprocessed.",
    }
    write_json(raw / "acquisition_manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(acquire(args.config), sort_keys=True))


if __name__ == "__main__":
    main()
