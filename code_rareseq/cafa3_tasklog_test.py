import json
from pathlib import Path
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

from cafa3_derive_task_evidence import derived_record
from cafa3_esmc_baselines import wait_for_storage
from cafa3_tasklog import TaskLogger


class Cafa3TaskLogTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def logger(self):
        return TaskLogger(self.root / "tasks.jsonl", {"code.py": "abc"}, "data-sha",
                          {"available": True, "device": "synthetic-CUDA"})

    def test_concurrent_jsonl_is_complete_and_parseable(self):
        logger = self.logger()
        artifact = self.root / "artifact.txt"
        artifact.write_text("evidence")

        def worker(index):
            event = logger.start("seed", 0, "main", "MFO", parent_run_id="combo-test", seed=index)
            logger.finish(event, "completed", [artifact])

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(worker, range(16)))
        lines = (self.root / "tasks.jsonl").read_text().splitlines()
        records = [json.loads(line) for line in lines]
        self.assertEqual(len(records), 32)
        self.assertEqual(len({record["run_id"] for record in records}), 16)
        terminal = [record for record in records if record["status"] == "completed"]
        self.assertEqual(len(terminal), 16)
        self.assertTrue(all(record["start_utc"] and record["end_utc"] for record in terminal))
        self.assertTrue(all(record["artifacts"][0]["sha256"] for record in terminal))
        self.assertTrue(all(record["data_sha256"] == "data-sha" for record in records))

    def test_storage_wait_creates_no_task_or_completion(self):
        logger = self.logger()
        with mock.patch("cafa3_esmc_baselines.shutil.disk_usage") as usage, \
                mock.patch("cafa3_esmc_baselines.time.sleep") as sleep:
            usage.side_effect = [mock.Mock(free=0), mock.Mock(free=9 * 1024 ** 3)]
            wait_for_storage(self.root / "waiting/MFO", "MFO")
        sleep.assert_called_once_with(300)
        self.assertFalse(logger.path.exists())

    def test_historical_record_is_artifact_derived_without_times(self):
        path = self.root / "runs/fold0/main/MFO"
        path.mkdir(parents=True)
        summary = {"fold": 0, "track": "main", "ontology": "MFO", "selected_tau": 0.5,
                   "t1_selection": "none", "code_sha256": "old-code", "data_manifest_sha256": "old-data",
                   "bce": [{"seed": seed, "best_epoch": 10, "tau": 0.5,
                            "fmax_threshold": 0.2, "smin_threshold": 0.3} for seed in range(3)]}
        (path / "summary.json").write_text(json.dumps(summary))
        names = [".complete", "metrics.tsv.gz", "per_label_metrics.tsv.gz", "validation_history.tsv",
                 "validation_thresholds_knn.tsv"]
        names += [f"bce_seed{seed}.pt" for seed in range(3)]
        names += [f"validation_thresholds_seed{seed}.tsv" for seed in range(3)]
        for name in names:
            (path / name).write_text(name)
        record = derived_record(path, {"cuda_device": "synthetic-CUDA", "torch_version": "test"})
        self.assertEqual(record["status"], "derived_from_artifacts")
        self.assertIsNone(record["start_utc"])
        self.assertIsNone(record["end_utc"])
        self.assertEqual(record["code_sha256"], "old-code")
        self.assertEqual(len(record["artifacts"]["seeds"]), 3)


if __name__ == "__main__":
    unittest.main()
