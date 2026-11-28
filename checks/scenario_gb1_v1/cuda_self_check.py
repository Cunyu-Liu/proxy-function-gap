#!/usr/bin/env python3
import numpy as np
import torch

from train_dev import fit_pcg


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    generator = torch.Generator(device="cuda").manual_seed(20261127)
    x = torch.randn((256, 12), generator=generator, device="cuda", dtype=torch.float64)
    x[:, 0] = 1.0
    truth = torch.randn(12, generator=generator, device="cuda", dtype=torch.float64)
    y = x @ truth + 0.01 * torch.randn(256, generator=generator, device="cuda", dtype=torch.float64)
    gram = x.T @ x / len(x)
    rhs = x.T @ y / len(x)
    diagonal = torch.full((12,), 0.01, device="cuda", dtype=torch.float64)
    diagonal[0] = 0.0
    regularizer = torch.diag(diagonal)
    y_second = torch.dot(y, y) / len(y)
    weights, history = fit_pcg(gram, rhs, regularizer, y_second, 11, 1e-10, 5000)
    exact = torch.linalg.solve(gram + regularizer, rhs)
    error = float(torch.max(torch.abs(weights - exact)))
    print({"cuda_device": torch.cuda.get_device_name(0), "iterations": history["iterations"], "max_weight_error_vs_direct_solve": error, "relative_residual": history["relative_residual_l2"]})
    if error > 1e-8:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
