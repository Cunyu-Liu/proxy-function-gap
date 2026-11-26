import json
import tempfile
import unittest
from pathlib import Path

import sync_experiment_logs as sync


class SyncExperimentLogsTest(unittest.TestCase):
    def write_jsonl(self, path, records):
        path.write_bytes(b"\n".join(records) + b"\n")

    def test_merge_preserves_corruption_identity_and_terminal_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local_path = root / "local.jsonl"
            remote_path = root / "remote.jsonl"
            start = "2026-11-26T00:00:00.000001+00:00"
            completed = {
                "run_id": "E1-same-second",
                "seed": 0,
                "start_utc": start,
                "config": {"out": "/out/a"},
                "status": "completed",
            }
            running = dict(completed, status="running")
            other_output = dict(completed, config={"out": "/out/b"})
            other_seed = dict(completed, seed=1)
            corrupt = b'{"run_id": "broken", "start_utc": "bad"'
            self.write_jsonl(
                local_path,
                [json.dumps(completed).encode(), corrupt],
            )
            self.write_jsonl(
                remote_path,
                [
                    json.dumps(running, separators=(",", ":")).encode(),
                    json.dumps(other_output).encode(),
                    json.dumps(dict(other_output, status="running")).encode(),
                    json.dumps(other_seed).encode(),
                    json.dumps(dict(other_seed, status="running")).encode(),
                    corrupt,
                ],
            )
            local = sync.read_events(local_path, "local")
            remote = sync.read_events(remote_path, "remote")
            result = sync.merge_events(local, remote)
            self.assertEqual(len(result["merged"]), 7)
            self.assertEqual(len(result["added"]), 5)
            self.assertEqual(len(result["duplicates"]), 1)
            attempts = sync.build_attempts(result["merged"])
            self.assertEqual(len(attempts), 3)
            self.assertTrue(all(a["effective_status"] == "completed" for a in attempts))
            self.assertTrue(all(a["pairing"] == "paired" for a in attempts))
            self.assertEqual(
                {tuple(a["output_directories"]) for a in attempts},
                {("/out/a",), ("/out/b",)},
            )
            checks = sync.verify_merge(local, remote, result)
            self.assertTrue(all(checks.values()), checks)
            self.assertEqual(result["merged"][1].raw, corrupt)
            self.assertIsNone(result["merged"][1].record)

    def test_invalidation_is_not_hidden_by_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            base = {
                "run_id": "invalidated-run",
                "seed": 0,
                "start_utc": "2026-11-26T00:00:00+00:00",
                "config": {"out": "/out/a"},
            }
            self.write_jsonl(
                path,
                [json.dumps(dict(base, status=status)).encode()
                 for status in ("running", "completed", "invalidated")],
            )
            events = sync.read_events(path, "local")
            attempt = sync.build_attempts(events)[0]
            self.assertEqual(attempt["effective_status"], "invalidated")
            self.assertTrue(attempt["terminal_conflict"])

    def test_same_start_with_different_outputs_stays_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            base = {
                "run_id": "collision",
                "seed": 0,
                "start_utc": "2026-11-26T00:00:00+00:00",
                "status": "running",
            }
            self.write_jsonl(
                path,
                [
                    json.dumps(dict(base, config={"out": "/out/a"})).encode(),
                    json.dumps(dict(base, config={"out": "/out/b"})).encode(),
                ],
            )
            attempts = sync.build_attempts(sync.read_events(path, "local"))
            self.assertEqual(len(attempts), 2)
            self.assertEqual(
                {tuple(attempt["output_directories"]) for attempt in attempts},
                {("/out/a",), ("/out/b",)},
            )
            self.assertFalse(any(attempt["ambiguous_same_start_outputs"] for attempt in attempts))


if __name__ == "__main__":
    unittest.main()
