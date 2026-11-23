import numpy as np
import pytest
import torch

from proxy_function_gap.kmer_proxy import KmerRegressor, kmer_features


def test_kmer_feature_contract():
    values = kmer_features("ATGTTTT")
    assert values.shape == (259,)
    assert np.isclose(values[:256].sum(), 1.0)
    assert values[-2] == 1.0
    assert values[-1] == 4 / 50


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA k-mer model test")
def test_kmer_model_cuda_forward():
    model = KmerRegressor().cuda()
    values = torch.from_numpy(np.asarray([kmer_features("A" * 50)])).cuda()
    assert model(values).shape == (1,)
