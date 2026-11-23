import unittest

import torch

from sample_runtime.model import encode_utrs, load_both_models, predict_both, require_cuda
from sample_runtime.provenance import TARGET_VALIDATION_AFFINE


class ModelTests(unittest.TestCase):
    def test_encoding_appends_atgg(self):
        utr = "A" * 50
        encoded = encode_utrs([utr])
        self.assertEqual(encoded.shape, (1, 54, 4))
        self.assertEqual(encoded[0, 50:54].sum(), 4.0)
        self.assertEqual(encoded[0, 50, 0], 1.0)
        self.assertEqual(encoded[0, 53, 2], 1.0)

    def test_cuda_weight_layout_and_devices(self):
        device = require_cuda("cuda:0")
        models, _ = load_both_models(device)
        utrs = [
            "CCCTGATAACTGAATTGTTGTTTGTTTTATTTGTATTGTTGTTTGTCGTG",
            "CCGCCTTGCGCCCGGCCCGCCGCCGCGGCCCACCCTGGCTGGCCCCTCGT",
            "GTGAAATTATGTGTACTAAACTCATTATGCTCTTTGGTCTTTCCCTTCTG",
        ]
        predictions = predict_both(models, utrs, device, batch_size=3)
        expected_raw = [1.7731566861215562, -1.212981711131623, 1.9146656893469922]
        published = [8.81778335571289, 4.0849199295043945, 9.042067527770996]
        for index, utr in enumerate(utrs):
            self.assertAlmostEqual(predictions["original"][index], expected_raw[index], delta=5e-4)
            reconstructed = TARGET_VALIDATION_AFFINE["mean"] + TARGET_VALIDATION_AFFINE["scale"] * predictions["original"][index]
            self.assertAlmostEqual(reconstructed, published[index], delta=5e-4)
        for model in models.values():
            self.assertFalse(model.training)
            self.assertTrue(all(parameter.is_cuda for parameter in model.parameters()))
            self.assertTrue(all(parameter.dtype == torch.float32 for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
