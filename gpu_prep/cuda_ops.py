"""原始 CUDA C++ 教学 kernels 的延迟编译入口。

约定：CUDA、torch.float32、contiguous、仅前向；第一次调用时编译，import 不编译。
需要带 CUDA 支持的 PyTorch、匹配的 CUDA toolkit/nvcc、C++ 编译器、Ninja。
"""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from types import ModuleType

import torch


@lru_cache(maxsize=1)
def load_extension() -> ModuleType:
    """显式预编译入口；不要把首次编译耗时计入 kernel benchmark。"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA teaching kernels require an available NVIDIA GPU")
    # 留到实际调用再导入：CPU 用户阅读/导入本包时不会触发构建。
    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None:
        raise RuntimeError(
            "CUDA toolkit/nvcc not found. A PyTorch CUDA wheel alone does not "
            "supply the CUDA extension compiler; configure CUDA_HOME."
        )
    source_dir = Path(__file__).resolve().parent / "csrc"
    return load(
        name="together_gpu_prep_cuda_v1",
        sources=[str(source_dir / "bindings.cpp"), str(source_dir / "kernels.cu")],
        extra_include_paths=[str(source_dir)],
        extra_cflags=["-O3", "-std=c++17"],
        # 不使用 --use_fast_math；保留便于参考比较的 FP32 数学路径。
        extra_cuda_cflags=["-O3", "-std=c++17", "-lineinfo"],
        with_cuda=True,
        verbose=os.environ.get("GPU_PREP_VERBOSE_BUILD", "0") == "1",
    )


def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """同形状张量逐元素相加；支持零元素，输出形状与 x 相同。"""
    return load_extension().vector_add(x, y)


def bias_relu(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """x[M,N] + bias[N] 后做 ReLU；支持 M=0 或 N=0。"""
    return load_extension().bias_relu(x, bias)


def row_sum(x: torch.Tensor) -> torch.Tensor:
    """x[M,N] -> [M]，沿列求和；要求 N>0，M 可以为 0。"""
    return load_extension().row_sum(x)


def row_max(x: torch.Tensor) -> torch.Tensor:
    """x[M,N] -> [M]，沿列取 max；要求 N>0，有限值输入。"""
    return load_extension().row_max(x)


def softmax(x: torch.Tensor) -> torch.Tensor:
    """数值稳定的行 softmax；要求 x[M,N]、N>0、有限值输入。"""
    return load_extension().softmax(x)


def transpose_naive(x: torch.Tensor) -> torch.Tensor:
    """朴素物理转置 x[M,N] -> contiguous [N,M]，不是只改 stride。"""
    return load_extension().transpose_naive(x)


def transpose_tiled(x: torch.Tensor) -> torch.Tensor:
    """32x32 shared tile + FP32 padding 的物理转置。"""
    return load_extension().transpose_tiled(x)


def matmul_naive(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a[M,K] @ b[K,N]；每线程一个输出，支持 K=0（全零输出）。"""
    return load_extension().matmul_naive(a, b)


def matmul_tiled(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """16x16 shared-memory tiled FP32 SIMT GEMM；没有 Tensor Core 路径。"""
    return load_extension().matmul_tiled(a, b)


def layer_norm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """沿最后一维做 LayerNorm，weight/bias 均为 [D]，方差使用分母 D。"""
    return load_extension().layer_norm(x, weight, bias, eps)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """沿最后一维做 RMSNorm；没有 mean subtraction 或 bias。"""
    return load_extension().rms_norm(x, weight, eps)


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    query_start: int = 0,
) -> torch.Tensor:
    """Dense attention 教学基线。

    q[B,H,L,D]，k/v[B,H,S,D]，1<=S<=4096，D>0。
    causal=True 时，第 i 个 query 只看 key_position <= query_start+i。
    单 token decode 通常传 query_start=S-1；默认 0 是从位置 0 开始的 query。
    不支持 dropout、padding mask、GQA、backward；不是 FlashAttention。
    """
    return load_extension().attention(q, k, v, causal, query_start)
