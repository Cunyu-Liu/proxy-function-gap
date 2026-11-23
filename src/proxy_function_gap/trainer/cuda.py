import os
import time

import torch


def torch_dtype(name):
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError("dtype must be float32 or float64")


def configure_cuda(gpu_index: int, memory_fraction: float, dtype: str, seed: int):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU training fallback is forbidden")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG=:4096:8 is required")
    if gpu_index >= torch.cuda.device_count():
        raise ValueError("requested CUDA device does not exist")
    if not 0 < memory_fraction <= 1:
        raise ValueError("memory fraction must be in (0, 1]")
    torch.cuda.set_per_process_memory_fraction(memory_fraction, device=gpu_index)
    device = torch.device("cuda", gpu_index)
    torch.cuda.set_device(device)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    actual_dtype = torch_dtype(dtype)
    properties = torch.cuda.get_device_properties(device)
    return device, actual_dtype, {
        "seed": seed,
        "requested_device": f"cuda:{gpu_index}",
        "actual_device": str(device),
        "gpu_name": properties.name,
        "gpu_capability": list(torch.cuda.get_device_capability(device)),
        "gpu_total_memory_bytes": properties.total_memory,
        "per_process_memory_fraction": memory_fraction,
        "dtype": str(actual_dtype),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "deterministic_algorithms": True,
        "tf32_enabled": False,
        "cpu_fallback": False,
    }


def cuda_self_check(gpu_index=0, memory_fraction=0.1, dtype="float32", seed=0):
    device, actual_dtype, metadata = configure_cuda(gpu_index, memory_fraction, dtype, seed)
    started = time.perf_counter()
    left = torch.ones((8, 8), device=device, dtype=actual_dtype)
    right = torch.eye(8, device=device, dtype=actual_dtype)
    result = left @ right
    torch.cuda.synchronize(device)
    if result.device != device or result.dtype != actual_dtype or not torch.equal(result, left):
        raise RuntimeError("CUDA self-check failed")
    metadata.update({
        "self_check": "PASS",
        "result_device": str(result.device),
        "result_dtype": str(result.dtype),
        "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "elapsed_seconds": time.perf_counter() - started,
        "scientific_result": False,
    })
    return metadata
