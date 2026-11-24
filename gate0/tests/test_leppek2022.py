import random
import unittest

from gate0_leppek2022.build import assign_roles, coverage_for
from gate0_leppek2022.common import distribution, levenshtein_bit, levenshtein_full, normalize_rna
from gate0_leppek2022.endpoints import ALL_ENDPOINTS, POOLED_ENDPOINTS


class DistanceTest(unittest.TestCase):
    def test_bit_distance_matches_dynamic_programming(self):
        generator = random.Random(17)
        cases = [("", ""), ("A", ""), ("", "AC"), ("ACGT", "ACGT"), ("ACGT", "AGT"), ("ACGT", "ACGTT")]
        for _ in range(100):
            left = "".join(generator.choice("ACGT") for _ in range(generator.randrange(0, 15)))
            right = "".join(generator.choice("ACGT") for _ in range(generator.randrange(0, 15)))
            cases.append((left, right))
        for left, right in cases:
            with self.subTest(left=left, right=right):
                self.assertEqual(levenshtein_bit(left, right), levenshtein_full(left, right))

    def test_rna_normalization(self):
        self.assertEqual(normalize_rna("acguu"), "ACGTT")
        with self.assertRaises(ValueError):
            normalize_rna("ACGN")


class EndpointContractTest(unittest.TestCase):
    def test_predicted_expression_is_not_measured(self):
        self.assertEqual(POOLED_ENDPOINTS["predicted_expression"]["status"], "predicted")
        self.assertIn("NOT_protein_measurement", POOLED_ENDPOINTS["predicted_expression"]["provenance"])

    def test_endpoint_names_are_unique(self):
        self.assertEqual(len(ALL_ENDPOINTS), len(set(ALL_ENDPOINTS)))

    def test_sentinel_is_missing_but_raw_value_is_retained(self):
        construct = {
            "sequence_id": "s", "source_construct_id": "c", "source_family": "pooled_233",
            "source_file": "f", "source_sheet": "s", "source_row": 2,
        }
        row = {"dG(MFE) EternaFold": "2.0E20"}
        item = next(item for item in coverage_for(construct, row, sentinels={"dG(MFE) EternaFold": [2e20]}) if item["endpoint"] == "mfe_eternafold")
        self.assertEqual(item["status"], "missing")
        self.assertEqual(item["raw_value"], "2.0E20")
        self.assertEqual(item["numeric_value"], "")


class RoleTest(unittest.TestCase):
    def test_cluster_has_exactly_one_role_and_is_deterministic(self):
        components = [
            {"cluster_id": "a", "members": [{"source_family": "pooled_233"}]},
            {"cluster_id": "b", "members": [{"source_family": "pooled_233"}]},
            {"cluster_id": "c", "members": [{"source_family": "cds_24"}]},
        ]
        targets = {"development": 0.6, "candidate": 0.2, "permanent_evaluation": 0.2}
        first, counts = assign_roles(components, "fixed", targets)
        second, _ = assign_roles(components, "fixed", targets)
        self.assertEqual(first, second)
        self.assertEqual(set(first), {"a", "b", "c"})
        self.assertEqual(sum(sum(role.values()) for role in counts.values()), 3)


class SummaryTest(unittest.TestCase):
    def test_distribution(self):
        result = distribution([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(result["n"], 4)
        self.assertEqual(result["median"], 2.5)
        self.assertEqual(result["min"], 1.0)
        self.assertEqual(result["max"], 4.0)


if __name__ == "__main__":
    unittest.main()
