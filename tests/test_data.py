import unittest

from sample_runtime.data import connection_key, gate0_raw_row_id, gate0_sequence_id, load_design_rows, validate_connections, validate_gate0_rows
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
        self.assertTrue(all(row["sequence_id"].startswith("sample2019_") for row in step + target))
        self.assertTrue(all(row["raw_row_id"].startswith("GSM3130443:row-") for row in step + target))

    def test_gate0_row_by_row_assertions(self):
        step, target = load_design_rows()
        result = validate_gate0_rows(step, target)
        self.assertEqual(result["step_rows_asserted"], EXPECTED_COUNTS["step"])
        self.assertEqual(result["target_rows_asserted"], EXPECTED_COUNTS["target"])

    def test_deliberate_id_errors_are_rejected(self):
        with self.assertRaises(ValueError):
            gate0_raw_row_id(0)
        step, target = load_design_rows()
        step[0]["sequence_id"] = "sample2019_invalid"
        with self.assertRaises(ValueError):
            validate_connections(step, target)


if __name__ == "__main__":
    unittest.main()
