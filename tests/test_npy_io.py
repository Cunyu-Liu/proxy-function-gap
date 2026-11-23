import hashlib
import unittest

from gate0_sample2019.npy_io import bool_array, decode_float64, decode_unicode, float64_array, one_hot, read_npz, unicode_array, write_npz


class NpyIoTests(unittest.TestCase):
    def test_round_trip_and_determinism(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            hashes = []
            for name in ["a", "b"]:
                path = Path(directory) / (name + ".npz")
                write_npz(path, {
                    "row_id": unicode_array(["r1", "r2"]),
                    "utr50_onehot": one_hot(["ACGT" * 12 + "AC"]),
                    "measured_rl": float64_array([1.25, 2.5]),
                    "synthetic": bool_array(False),
                })
                arrays = read_npz(path)
                self.assertEqual(decode_unicode(*arrays["row_id"]), ["r1", "r2"])
                self.assertEqual(decode_float64(arrays["measured_rl"][2]), [1.25, 2.5])
                self.assertEqual(arrays["utr50_onehot"][1], (1, 50, 4))
                self.assertEqual(arrays["synthetic"][2], b"\x00")
                hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(hashes[0], hashes[1])


if __name__ == "__main__":
    unittest.main()
