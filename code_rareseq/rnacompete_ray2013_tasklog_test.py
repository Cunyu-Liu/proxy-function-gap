#!/usr/bin/env python3
import concurrent.futures
import json
import os
import tempfile
from types import SimpleNamespace

from rnacompete_ray2013_tasklog import TaskLogger, append_locked, derived_manifest


def main():
    with tempfile.TemporaryDirectory() as root:
        frozen = os.path.join(root, "frozen")
        embeddings = os.path.join(root, "embeddings")
        os.makedirs(frozen)
        os.makedirs(embeddings)
        with open(os.path.join(frozen, "split_manifest.json"), "w") as handle:
            json.dump({"status": "accepted"}, handle)
        with open(os.path.join(root, "labels.npz"), "wb") as handle:
            handle.write(b"labels")
        with open(os.path.join(embeddings, "embedding_manifest.json"), "w") as handle:
            json.dump({"cache": {"npy_sha256": "a", "ids_sha256": "b"}}, handle)
        args = SimpleNamespace(command="train", track="sequence", arch="linear", fold=2, seed=1,
                               frozen_root=frozen, embedding_dir=embeddings,
                               label_cache=os.path.join(root, "labels.npz"), out=root,
                               prog=os.path.abspath(__file__))
        path = os.path.join(root, "task_runs.jsonl")
        logger = TaskLogger(path, args)
        logger.start(cuda_required=True)
        logger.finish("failed", {"required": True, "device": "cuda:0", "probe": "passed"},
                      RuntimeError("synthetic failure"))
        records = [json.loads(line) for line in open(path)]
        assert len(records) == 2 and records[0]["run_id"] == records[1]["run_id"]
        assert records[0]["status"] == "running" and records[0]["end_utc"] is None
        assert records[1]["status"] == "failed" and records[1]["start_utc"] == records[0]["start_utc"]
        assert records[1]["end_utc"] and "synthetic failure" in records[1]["error"]
        assert records[1]["seed"] == 1 and records[1]["fold"] == 2 and records[1]["track"] == "sequence"
        assert records[1]["data_sha256"]["combined_data_sha256"] and records[1]["code_sha256"]

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda i: append_locked(path, {"run_id": f"parallel-{i}"}), range(12)))
        assert len(open(path).readlines()) == 14
        for line in open(path):
            json.loads(line)

        selections = os.path.join(root, "selections")
        os.makedirs(selections)
        with open(os.path.join(selections, "sequence_linear.json"), "w") as handle:
            json.dump({"status": "frozen"}, handle)
        run = os.path.join(root, "runs", "sequence_linear_f2_s1")
        os.makedirs(run)
        with open(os.path.join(run, "eval_config.json"), "w") as handle:
            json.dump({"code_sha256": "code", "frozen_manifest_sha256": "data"}, handle)
        with open(os.path.join(run, ".eval_complete"), "w") as handle:
            handle.write("completed\n")
        manifest = derived_manifest(root)
        assert manifest["derived_from_artifacts"] and not manifest["historical_start_end_fabricated"]
        assert manifest["start_utc"] is None and manifest["end_utc"] is None
        assert all(record["start_utc"] is None and record["end_utc"] is None
                   for record in manifest["records"])
        assert manifest["report_pointer"].endswith("RNAcompete_Ray2013_RNAFM基线执行报告_2026-11-27.md")
    print("RNACOMPETE_TASKLOG_TEST_OK: concurrent JSONL, real new-run times, failure, derived null times")


if __name__ == "__main__":
    main()
