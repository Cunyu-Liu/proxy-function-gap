import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from gate0_sample2019.protocol import Learner


class LearnerAllowlistTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.view = root / "view.tsv.gz"
        columns = ["raw_row_id", "sequence_id", "measured_rl"]
        with gzip.open(self.view, "wt", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerow({"raw_row_id": "r1", "sequence_id": "s1", "measured_rl": "1.0"})
        self.allowlist = root / "allowlist.json"
        self.allowlist.write_text(json.dumps({
            "feature_roles": {"train": {"path": str(self.view), "columns": ["raw_row_id", "sequence_id"]}},
            "label_roles": {"train": {"path": str(self.view), "columns": columns}},
            "parameter_edit_roles": ["dev", "calibration"],
        }))
        self.learner = Learner(self.allowlist)

    def tearDown(self):
        self.temp.cleanup()

    def test_feature_columns_are_projected(self):
        self.assertEqual(list(self.learner.read_features("train")), [{"raw_row_id": "r1", "sequence_id": "s1"}])

    def test_outer_and_query_labels_fail_closed(self):
        with self.assertRaises(PermissionError):
            list(self.learner.read_labels("outer"))
        with self.assertRaises(PermissionError):
            list(self.learner.read_labels("query"))

    def test_alpha_and_threshold_origins(self):
        self.assertTrue(self.learner.validate_parameter_origin("dev"))
        self.assertTrue(self.learner.validate_parameter_origin("calibration"))
        for role in ["train", "query", "outer"]:
            with self.assertRaises(PermissionError):
                self.learner.validate_parameter_origin(role)


if __name__ == "__main__":
    unittest.main()
