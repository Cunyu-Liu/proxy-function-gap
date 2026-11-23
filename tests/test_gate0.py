import unittest

from gate0_sample2019.common import hamming_distance, normalize_sequence


class NormalizationTests(unittest.TestCase):
    def test_fixed_formats(self):
        variable = "A" * 50
        self.assertEqual(normalize_sequence(variable, {"length": 50, "suffix": "", "format": "UTR50"})[0], variable)
        self.assertEqual(normalize_sequence(variable + "ATGG", {"length": 54, "suffix": "ATGG", "format": "UTR50_ATGG"})[0], variable)
        self.assertEqual(normalize_sequence(variable + "ATGGGCGAA", {"length": 59, "suffix": "ATGGGCGAA", "format": "UTR50_ATGGGCGAA"})[0], variable)

    def test_wrong_suffix_fails_closed(self):
        with self.assertRaises(ValueError):
            normalize_sequence("A" * 54, {"length": 54, "suffix": "ATGG", "format": "UTR50_ATGG"})

    def test_variable_sequence_is_not_truncated(self):
        sequence = "A" * 75
        rule = {"length": None, "suffix": None, "format": "VARIABLE_UTR_25_100"}
        self.assertEqual(normalize_sequence(sequence, rule)[0], sequence)

    def test_hamming_requires_same_length(self):
        self.assertEqual(hamming_distance("A" * 50, "A" * 49 + "C"), 1)
        self.assertIsNone(hamming_distance("A" * 50, "A" * 49))


if __name__ == "__main__":
    unittest.main()


class LeakageNegativeTests(unittest.TestCase):
    def test_deliberately_split_near_duplicates_are_rejected(self):
        from gate0_sample2019.grouping import partition_violations
        left = "A" * 50
        right = "A" * 49 + "C"
        violations = partition_violations([(left, right)], {left: "group_A", right: "group_B"})
        self.assertEqual(len(violations), 1)

    def test_distance_one_candidate_half_guarantee(self):
        sequence = "ACGT" * 12 + "AC"
        for position in range(50):
            mutated = sequence[:position] + ("A" if sequence[position] != "A" else "C") + sequence[position + 1:]
            midpoint = len(sequence) // 2
            shares_half = sequence[:midpoint] == mutated[:midpoint] or sequence[midpoint:] == mutated[midpoint:]
            self.assertTrue(shares_half)


class ExactAssignmentNegativeTests(unittest.TestCase):
    def test_same_sequence_in_two_groups_is_rejected(self):
        from gate0_sample2019.grouping import exact_assignment_violations
        records = [
            {"raw_row_id": "raw59", "sequence_id": "same", "near_group_id": "A"},
            {"raw_row_id": "raw54", "sequence_id": "same", "near_group_id": "B"},
        ]
        self.assertEqual(len(exact_assignment_violations(records)), 1)


class TsvReaderRegressionTests(unittest.TestCase):
    def test_tsv_reader_preserves_columns(self):
        import gzip
        import tempfile
        from pathlib import Path
        from gate0_sample2019.verify import rows_gz
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.tsv.gz"
            with gzip.open(path, "wt") as handle:
                handle.write("raw_row_id\tsequence_id\nrow1\tseq1\n")
            self.assertEqual(list(rows_gz(path)), [{"raw_row_id": "row1", "sequence_id": "seq1"}])


class RawRowIdTests(unittest.TestCase):
    def test_row_id_does_not_depend_on_duplicated_raw_index(self):
        from gate0_sample2019.common import make_raw_row_id
        self.assertEqual(make_raw_row_id("GSM1", 1), "GSM1:row-0000001")
        self.assertNotEqual(make_raw_row_id("GSM1", 1), make_raw_row_id("GSM1", 2))
