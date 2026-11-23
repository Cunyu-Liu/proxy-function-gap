import unittest

from gate0_sample2019.fixed_pools import compute_pool_hash, dedup_rows, select_complete_units


def record(sequence_id, unit_id, context="ctx", near=None, trajectory=""):
    return {
        "sequence_id": sequence_id,
        "allocation_unit_id": unit_id,
        "near_group_id": near or unit_id,
        "trajectory_id": trajectory,
        "replicate_count": 1,
        "context_sources": [{"context": context, "source_row_ids": [sequence_id + "_row"]}],
    }


class CompleteUnitSelectionTests(unittest.TestCase):
    def config(self, size):
        return {
            "pool_version": "test-v1",
            "pool_sizes": {"dev": size},
            "selection_rule": "test-selection",
            "q_rule": "uniform",
        }

    def test_exact_size_does_not_split_complete_units(self):
        rows = [record("a1", "a"), record("a2", "a"), record("b", "b"), record("c", "c")]
        chosen, stats = select_complete_units(self.config(2), "dev", rows)
        self.assertEqual({row["sequence_id"] for row in chosen}, {"b", "c"})
        self.assertEqual(stats["selected_allocation_units"], 2)

    def test_impossible_exact_size_fails_closed(self):
        rows = [record("a1", "a"), record("a2", "a"), record("b1", "b"), record("b2", "b")]
        with self.assertRaisesRegex(ValueError, "No exact complete-allocation-unit pool"):
            select_complete_units(self.config(3), "dev", rows)

    def test_cross_role_near_conflict_is_blocked(self):
        blocked = {"sequences": set(), "near": {"a"}, "units": set(), "trajectories": set()}
        chosen, stats = select_complete_units(self.config(1), "dev", [record("a1", "a"), record("b", "b")], blocked)
        self.assertEqual([row["sequence_id"] for row in chosen], ["b"])
        self.assertEqual(stats["cross_role_blocked_allocation_units"], 1)


class SequenceContextDedupTests(unittest.TestCase):
    def test_contexts_share_one_sequence_and_preserve_sources(self):
        config = {"designed_context": "designed"}
        base = {
            "sequence_id": "s",
            "utr50": "A",
            "allocation_unit_id": "u",
            "near_group_id": "n",
            "trajectory_id": "",
            "source_file": "f",
            "source_file_sha256": "h",
            "raw_row_sha256": "r",
            "outer_stratum": "random|x",
            "source_library": "lib",
            "measured_rl": "must-not-propagate",
        }
        rows = [dict(base, raw_row_id="r2", context="z"), dict(base, raw_row_id="r1", context="a")]
        result = dedup_rows(config, "outer_random", rows, set(), set())
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["source_row_ids"], ["r1", "r2"])
        self.assertEqual(result[0]["replicate_count"], 2)
        self.assertNotIn("measured_rl", result[0])
        self.assertEqual([source["context"] for source in result[0]["context_sources"]], ["a", "z"])

    def test_pool_hash_binds_context_sources(self):
        config = {"pool_version": "test-v1", "selection_rule": "rule", "q_rule": "uniform"}
        rows = [record("s", "u"), record("t", "v")]
        first = compute_pool_hash(config, "dev_000", "split", rows)
        rows[0]["context_sources"][0]["source_row_ids"] = ["different"]
        self.assertNotEqual(first, compute_pool_hash(config, "dev_000", "split", rows))


if __name__ == "__main__":
    unittest.main()


class DisjointMultiPoolTests(unittest.TestCase):
    def test_repeated_selection_blocks_sequences_near_units_and_trajectories(self):
        config = {
            "pool_version": "test-v2",
            "pool_sizes": {"dev": 1},
            "selection_rule": "test-selection",
            "q_rule": "uniform",
        }
        rows = [record("a", "ua", trajectory="ta"), record("b", "ub", trajectory="tb")]
        blocked = {key: set() for key in ["sequences", "near", "units", "trajectories"]}
        first, _ = select_complete_units(config, "dev", rows, blocked)
        second, _ = select_complete_units(config, "dev", rows, blocked)
        self.assertNotEqual(first[0]["sequence_id"], second[0]["sequence_id"])
        with self.assertRaisesRegex(ValueError, "No exact complete-allocation-unit pool"):
            select_complete_units(config, "dev", rows, blocked)
