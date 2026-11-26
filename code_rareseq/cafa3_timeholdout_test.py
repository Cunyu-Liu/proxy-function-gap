import hashlib
import io
from pathlib import Path
import tarfile
import tempfile
import unittest

from cafa3_timeholdout import ancestor_distances, canonical_term, fold_for_group, frequency_bucket, parse_obo, read_mapping, read_tsv


class Cafa3TimeHoldoutContractTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.obo = Path(self.tempdir.name) / "go.obo"
        self.obo.write_text("""format-version: 1.2

[Term]
id: GO:0000001
name: root
namespace: biological_process

[Term]
id: GO:0000002
name: child
namespace: biological_process
alt_id: GO:9000002
is_a: GO:0000001 ! root

[Term]
id: GO:0000003
name: grandchild
namespace: biological_process
relationship: part_of GO:0000002 ! child

[Term]
id: GO:0000004
name: wrong aspect
namespace: molecular_function
is_a: GO:0000002 ! child

[Term]
id: GO:0000005
name: obsolete
namespace: biological_process
is_obsolete: true
""", encoding="utf-8")
        self.terms, self.alt_ids, self.collisions = parse_obo(self.obo)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_obo_namespace_alt_id_and_obsolete_filter(self):
        self.assertEqual(canonical_term("GO:9000002", "BPO", self.terms, self.alt_ids), ("GO:0000002", "accepted_alt_id", True))
        self.assertEqual(canonical_term("GO:0000004", "BPO", self.terms, self.alt_ids), (None, "namespace_mismatch", False))
        self.assertEqual(canonical_term("GO:0000005", "BPO", self.terms, self.alt_ids), (None, "obsolete_go", False))
        self.assertEqual(canonical_term("GO:9999999", "BPO", self.terms, self.alt_ids), (None, "unknown_go", False))

    def test_propagation_is_is_a_and_part_of_only(self):
        self.assertEqual(ancestor_distances("GO:0000003", self.terms), {"GO:0000003": 0, "GO:0000002": 1, "GO:0000001": 2})
        self.assertEqual(ancestor_distances("GO:0000004", self.terms), {"GO:0000004": 0})

    def test_tsv_reader_accepts_archive_text_stream(self):
        stream = io.StringIO("P1\tGO:0000001\tP\nP2\tGO:0000002\tF\n")
        self.assertEqual(list(read_tsv(stream)), [["P1", "GO:0000001", "P"], ["P2", "GO:0000002", "F"]])

    def test_target_mapping_from_archive(self):
        archive_path = Path(self.tempdir.name) / "targets.tgz"
        payload = b"T1\tGENE1\nT2\tGENE2\n"
        with tarfile.open(archive_path, "w:gz") as archive:
            info = tarfile.TarInfo("Mapping files/test.map")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        with tarfile.open(archive_path, "r:gz") as archive:
            self.assertEqual(read_mapping(archive, "Mapping files/test.map"), {"T1": "GENE1", "T2": "GENE2"})

    def test_frequency_bucket_boundaries(self):
        expected = {0: "0", 1: "1-4", 4: "1-4", 5: "5-19", 19: "5-19", 20: "20-99", 99: "20-99", 100: "100-499", 499: "100-499", 500: ">=500"}
        self.assertEqual({count: frequency_bucket(count) for count in expected}, expected)

    def test_group_hash_determinism_and_three_fold_support(self):
        groups = [f"exact:{hashlib.md5(str(index).encode()).hexdigest()}" for index in range(100)]
        first = {group: fold_for_group(group) for group in groups}
        second = {group: fold_for_group(group) for group in reversed(groups)}
        self.assertEqual(first, second)
        self.assertEqual(set(first.values()), {0, 1, 2})
        self.assertTrue(all(0 <= fold < 3 for fold in first.values()))


if __name__ == "__main__":
    unittest.main()
