import unittest

from sample_runtime.data import connection_key, load_design_rows
from sample_runtime.provenance import EXPECTED_COUNTS


class DataTests(unittest.TestCase):
    def test_counts_and_connection_keys(self):
        step, target = load_design_rows()
        self.assertEqual(len(step), EXPECTED_COUNTS["step"])
        self.assertEqual(len(target), EXPECTED_COUNTS["target"])
        self.assertEqual(len({connection_key(row) for row in step}), len(step))
        self.assertEqual(len({connection_key(row) for row in target}), len(target))
        self.assertTrue(all(row["sparse_step"] for row in step))
        self.assertTrue(all(row["sparse_step"] == "" for row in target))
        self.assertTrue(all(row["measured_rl"] for row in step + target))


if __name__ == "__main__":
    unittest.main()
