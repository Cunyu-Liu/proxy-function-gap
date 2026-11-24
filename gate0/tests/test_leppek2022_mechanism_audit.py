import json
import unittest
from pathlib import Path

from gate0_leppek2022.mechanism_audit import (
    average_ranks,
    correlation_and_rank_rows,
    pearson,
    source_name_for,
    spearman,
)


class CorrelationTest(unittest.TestCase):
    def test_pearson_and_spearman(self):
        self.assertAlmostEqual(pearson([1, 2, 3], [2, 4, 6]), 1.0)
        self.assertAlmostEqual(pearson([1, 2, 3], [3, 2, 1]), -1.0)
        self.assertAlmostEqual(spearman([1, 2, 2], [3, 2, 1]), -(3 ** 0.5) / 2)
        self.assertEqual(average_ranks([1, 2, 2, 4]), [1.0, 2.5, 2.5, 4.0])
        self.assertEqual(average_ranks([1, 2, 2, 4], descending=True), [4.0, 2.5, 2.5, 1.0])

    def test_constant_endpoint_has_no_pearson_value(self):
        self.assertIsNone(pearson([1, 1], [1, 2]))


class RankReversalTest(unittest.TestCase):
    def test_discordance_and_top_k_are_construct_pair_based(self):
        rows = [
            {"source_construct_id": "A", "x": 1.0, "y": 1.0},
            {"source_construct_id": "B", "x": 2.0, "y": 3.0},
            {"source_construct_id": "C", "x": 3.0, "y": 2.0},
        ]
        correlation, summary, reversals, ranks = correlation_and_rank_rows(rows, "cds_24", "x", "y", 2)
        self.assertEqual(correlation["n"], 3)
        self.assertEqual(summary["n_unordered_pairs"], 3)
        self.assertEqual(summary["concordant_pairs"], 2)
        self.assertEqual(summary["discordant_pairs"], 1)
        self.assertEqual(summary["top_k_overlap_n"], 2)
        self.assertEqual(len(reversals), 3)
        self.assertEqual(len(ranks), 3)


class SourceMappingTest(unittest.TestCase):
    def test_explicit_names_and_default(self):
        self.assertEqual(source_name_for("START reference"), "Nluc start")
        self.assertEqual(source_name_for("12.0"), "12")
        self.assertEqual(source_name_for("Twist_1"), "Twist_1")


class PreregistrationContractTest(unittest.TestCase):
    def test_analysis_constraints_remain_frozen(self):
        path = Path(__file__).resolve().parents[1] / "configs" / "leppek2022_cds_mechanism_audit_v5.json"
        config = json.loads(path.read_text())
        self.assertFalse(config["inferential_statistics"])
        self.assertFalse(config["model_training_or_inference"])
        self.assertFalse(config["outer_results"])
        self.assertFalse(config["predicted_expression_as_measured"])
        self.assertEqual(config["rank_reversal"]["top_k"], 6)
        self.assertNotIn("predicted_expression", config["cds_primary_endpoints"])
        self.assertNotIn("predicted_expression", config["pooled_context_endpoints"])


if __name__ == "__main__":
    unittest.main()
