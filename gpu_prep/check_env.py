"""Print environment information without compiling or reserving a GPU."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import sys


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_environment() -> dict:
    result = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": package_version("torch"),
        "triton": package_version("triton"),
        "pytest": package_version("pytest"),
        "nvcc": shutil.which("nvcc"),
        "nvidia_smi": shutil.which("nvidia-smi"),
        "cuda_available": False,
        "triton_interpret": os.environ.get("TRITON_INTERPRET", "0"),
        "devices": [],
    }
    if result["torch"] is None:
        return result
    import torch
    result["torch_cuda_runtime"] = torch.version.cuda
    result["cuda_available"] = torch.cuda.is_available()
    result["matmul_float32_precision"] = torch.get_float32_matmul_precision()
    result["matmul_allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
    result["matmul_allow_fp16_reduced_precision_reduction"] = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    result["matmul_allow_bf16_reduced_precision_reduction"] = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(i)
            result["devices"].append({
                "index": i,
                "name": properties.name,
                "compute_capability": [properties.major, properties.minor],
                "total_memory_bytes": properties.total_memory,
                "multiprocessor_count": properties.multi_processor_count,
            })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    info = collect_environment()
    print(json.dumps(info, ensure_ascii=False, indent=2))
    if args.require_gpu and not info["cuda_available"]:
        raise SystemExit("This command requires an NVIDIA CUDA device; no CPU timing is substituted.")


if __name__ == "__main__":
    main()
