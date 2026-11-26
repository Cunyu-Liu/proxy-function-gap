#!/usr/bin/env python3
import json
import os
import tempfile
import unittest

import torch

from e5_holdout_retention import (HOLDOUTS, SEEDS, TRACKS, apply_plan, build_plan,
                                  materialize_checkpoint, sha256_file, verify_plan)


class E5HoldoutRetentionTest(unittest.TestCase):
    def make_matrix(self, root):
        run_root = os.path.join(root, "holdout_convergence")
        summary_dir = os.path.join(root, "summary")
        audit_dir = os.path.join(run_root, "retention_audit")
        os.makedirs(summary_dir)
        for name in ("verification.json", "independent_verification.json"):
            with open(os.path.join(summary_dir, name), "w") as handle:
                json.dump({"status": "complete", "n_pairs_completed": 72, "n_pairs_expected": 72}, handle)
        for seed in SEEDS:
            for track in TRACKS:
                for holdout in HOLDOUTS:
                    run_dir = os.path.join(run_root, f"holdout_convergence_s{seed}", track)
                    for directory in ("checkpoints", "manifests", "history", "pairs", "scores"):
                        os.makedirs(os.path.join(run_dir, directory), exist_ok=True)
                    prefix = f"{holdout}_{track}"
                    paths = {name: os.path.join(run_dir, directory, prefix + suffix)
                             for name, directory, suffix in (("checkpoint", "checkpoints", ".pt"),
                                                            ("manifest", "manifests", ".json"),
                                                            ("history", "history", ".tsv"),
                                                            ("pair", "pairs", ".tsv"),
                                                            ("scores", "scores", ".npz"))}
                    torch.save({"state_dict": {"weight": torch.tensor([float(seed), len(holdout)])}},
                               paths["checkpoint"])
                    for name in ("history", "pair", "scores"):
                        with open(paths[name], "wb") as handle:
                            handle.write(f"{seed}/{track}/{holdout}/{name}\n".encode())
                    manifest = {"heldout": holdout, "track": track, "seed": seed,
                                "artifacts": {"checkpoint_sha256": sha256_file(paths["checkpoint"]),
                                              "history_sha256": sha256_file(paths["history"]),
                                              "pair_sha256": sha256_file(paths["pair"]),
                                              "scores_sha256": sha256_file(paths["scores"])}}
                    with open(paths["manifest"], "w") as handle:
                        json.dump(manifest, handle)
        return run_root, summary_dir, audit_dir

    def test_plan_apply_and_three_independent_verifications(self):
        with tempfile.TemporaryDirectory() as root:
            run_root, summary_dir, audit_dir = self.make_matrix(root)
            plan_path = build_plan(run_root, summary_dir, audit_dir)
            with open(plan_path) as handle:
                plan = json.load(handle)
            self.assertEqual(len(plan["files"]), 72)
            self.assertEqual(sum(row["representative"] for row in plan["files"]), 6)
            self.assertTrue(all(row["holdout"] == "CSTF2T" for row in plan["files"] if row["representative"]))
            apply_plan(plan_path, run_root, audit_dir)
            for row in plan["files"]:
                checkpoint = row["artifacts"]["checkpoint"]["path"]
                if row["representative"]:
                    self.assertTrue(os.path.isfile(checkpoint))
                else:
                    self.assertFalse(os.path.exists(checkpoint))
                    self.assertTrue(os.path.isfile(checkpoint + ".gz"))
                with materialize_checkpoint(checkpoint, row["artifacts"]["checkpoint"]["sha256"]) as restored:
                    self.assertEqual(sha256_file(restored), row["artifacts"]["checkpoint"]["sha256"])
            for pass_index in (1, 2, 3):
                report = verify_plan(plan_path, run_root, audit_dir, pass_index)
                self.assertEqual(report["status"], "passed")
                self.assertEqual(report["tasks_verified"], 72)
                self.assertGreater(report["net_file_bytes_released"], 0)


if __name__ == "__main__":
    unittest.main()
