import unittest

from gate0_sample2019.fresh_proxy_common import intersections, open_deterministic_tsv_gz, read_tsv


class FreshProxyHelperTests(unittest.TestCase):
    def test_intersection_detector(self):
        sets = {"train": {"s1"}, "outer": {"s2"}}
        self.assertEqual(intersections(sets)[0]["count"], 0)
        sets["outer"].add("s1")
        self.assertEqual(intersections(sets)[0]["count"], 1)

    def test_deterministic_gzip(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            hashes = []
            for name in ["a", "b"]:
                path = Path(directory) / (name + ".tsv.gz")
                handle, writer = open_deterministic_tsv_gz(path, ["row_id", "value"])
                writer.writerow({"row_id": "r1", "value": "1"})
                handle.close()
                hashes.append(__import__("hashlib").sha256(path.read_bytes()).hexdigest())
                self.assertEqual(list(read_tsv(path)), [{"row_id": "r1", "value": "1"}])
            self.assertEqual(hashes[0], hashes[1])


if __name__ == "__main__":
    unittest.main()
