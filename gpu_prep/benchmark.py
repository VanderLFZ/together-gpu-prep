"""Correctness-first CUDA wrapper benchmarks, with honest measurement labels.

Run: python -m gpu_prep.benchmark --op matmul --shape 128,256,384
Shape for matmul means M,N,K. All inputs are already resident on the GPU.
The measured callable allocates its output. This is not a pure kernel-only
benchmark and does not time input copies, JIT compilation, or correctness checks.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import time
from typing import Callable

import torch
import torch.nn.functional as F

from .check_env import collect_environment
from .roofline import gemm_workload, vector_add_workload


DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
DEFAULT_SHAPES = {
    "vector_add": (1_048_576,), "bias_relu": (511, 769),
    "row_sum": (511, 769), "row_max": (511, 769), "softmax": (511, 769),
    "transpose": (511, 769), "matmul": (128, 256, 384),
    "layer_norm": (511, 769), "rms_norm": (511, 769),
    "attention": (1, 4, 127, 257, 64),
}


@dataclass
class Case:
    op: str
    shape: tuple[int, ...]
    inputs: tuple[torch.Tensor, ...]
    torch_fn: Callable[[], torch.Tensor]
    reference: torch.Tensor
    logical_bytes: int | None
    flops: int | None
    notes: str


def parse_shape(value: str) -> tuple[int, ...]:
    try:
        shape = tuple(int(s.strip()) for s in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape must be comma-separated integers") from exc
    if not shape or any(n <= 0 for n in shape):
        raise argparse.ArgumentTypeError("benchmark dimensions must be positive; tests cover empty shapes")
    return shape


def _require_dims(shape: tuple[int, ...], count: int, op: str) -> None:
    if len(shape) != count or any(n <= 0 for n in shape):
        raise ValueError(f"{op} requires {count} positive dimensions; received {shape}")


@torch.inference_mode()
def build_case(op: str, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device) -> Case:
    """Make data and a float64 numerical oracle outside every timing region."""
    def rand(*dims: int) -> torch.Tensor:
        return torch.randn(dims, dtype=dtype, device=device)

    size = torch.empty((), dtype=dtype).element_size()
    logical_bytes = None
    flops = None
    notes = "Finite random inputs; float64 oracle is computed outside timing."
    if op == "vector_add":
        _require_dims(shape, 1, op)
        x, y = rand(*shape), rand(*shape)
        args = (x, y)
        fn = lambda: x + y
        ref = (x.double() + y.double()).to(dtype)
        model = vector_add_workload(shape[0], element_bytes=size)
        logical_bytes, flops = model.ideal_bytes, model.flops
    elif op == "matmul":
        _require_dims(shape, 3, op)
        m, n, k = shape
        a, b = rand(m, k), rand(k, n)
        args = (a, b)
        fn = lambda: a @ b
        ref = (a.double() @ b.double()).to(dtype)
        model = gemm_workload(m, n, k, element_bytes=size)
        logical_bytes, flops = model.ideal_bytes, model.flops
        notes += " Shape order is M,N,K; beta=0; logical bytes assume ideal reuse."
    elif op == "attention":
        _require_dims(shape, 5, op)
        batch, heads, length, source, dim = shape
        q, k, v = rand(batch, heads, length, dim), rand(batch, heads, source, dim), rand(batch, heads, source, dim)
        args = (q, k, v)
        fn = lambda: F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        scores = (q.double() @ k.double().transpose(-2, -1)) / math.sqrt(dim)
        ref = (scores.softmax(dim=-1) @ v.double()).to(dtype)
        flops = 4 * batch * heads * length * source * dim
        notes += " Noncausal, no dropout. FLOPs count two GEMMs only. PyTorch SDPA may choose a fused backend."
    else:
        _require_dims(shape, 2, op)
        m, n = shape
        x = rand(m, n)
        args = (x,)
        if op == "bias_relu":
            bias = rand(n)
            args = (x, bias)
            fn = lambda: torch.relu(x + bias)
            # The custom kernels add and apply ReLU in FP32 for low-precision inputs.
            ref = torch.relu(x.double() + bias.double()).to(dtype)
            logical_bytes = (2 * m * n + n) * size
            flops = m * n
            notes += " Bias traffic assumes ideal reuse; max comparisons are omitted from FLOPs."
        elif op == "row_sum":
            fn = lambda: x.sum(dim=-1)
            ref = x.double().sum(dim=-1).to(dtype)
            logical_bytes = (m * n + m) * size
            flops = m * max(n - 1, 0)
        elif op == "row_max":
            fn = lambda: x.amax(dim=-1)
            ref = x.double().amax(dim=-1).to(dtype)
            logical_bytes = (m * n + m) * size
        elif op == "softmax":
            fn = lambda: x.softmax(dim=-1)
            ref = x.double().softmax(dim=-1).to(dtype)
            logical_bytes = 2 * m * n * size
            notes += " Logical traffic is a one-read/one-write lower bound, not measured DRAM traffic."
        elif op == "transpose":
            fn = lambda: x.T.contiguous()
            ref = x.T.contiguous()
            logical_bytes = 2 * m * n * size
            notes += " Physical transpose; the reference includes contiguous(), not just a stride-changing view."
        elif op == "layer_norm":
            weight, bias = rand(n), rand(n)
            args = (x, weight, bias)
            fn = lambda: F.layer_norm(x, (n,), weight, bias, eps=1e-5)
            ref = F.layer_norm(x.double(), (n,), weight.double(), bias.double(), eps=1e-5).to(dtype)
        elif op == "rms_norm":
            weight = rand(n)
            args = (x, weight)
            fn = lambda: F.rms_norm(x, (n,), weight, eps=1e-5)
            xd = x.double()
            ref = (xd * torch.rsqrt(xd.square().mean(dim=-1, keepdim=True) + 1e-5) * weight.double()).to(dtype)
        else:
            raise ValueError(f"unknown op: {op}")
    return Case(op, shape, args, fn, ref, logical_bytes, flops, notes)


def make_callable(case: Case, backend: str, variant: str = "tiled") -> Callable[[], torch.Tensor]:
    if backend == "torch":
        return case.torch_fn
    if backend not in ("cuda", "triton"):
        raise ValueError(f"unknown backend: {backend}")
    module = importlib.import_module(f"gpu_prep.{backend}_ops")
    name = f"{case.op}_{variant}" if case.op in ("transpose", "matmul") else case.op
    operation = getattr(module, name)
    return lambda: operation(*case.inputs)


def summarize_samples(samples_ms: list[float]) -> dict:
    """p95 is an interpolated quantile of repeat averages, not request-tail latency."""
    if not samples_ms or any(not math.isfinite(x) or x <= 0 for x in samples_ms):
        raise ValueError("latency samples must be nonempty, finite, and positive")
    ordered = sorted(samples_ms)
    position = 0.95 * (len(ordered) - 1)
    low, high = math.floor(position), math.ceil(position)
    p95 = ordered[low] + (position - low) * (ordered[high] - ordered[low])
    return {
        "median_ms": statistics.median(ordered), "min_ms": ordered[0],
        "p95_of_repeat_means_ms": p95, "samples_ms": samples_ms,
    }


@torch.inference_mode()
def check_output(fn: Callable[[], torch.Tensor], reference: torch.Tensor) -> dict:
    out = fn()  # A first compilation or extension build can happen here.
    tolerance = {
        torch.float32: (3e-4, 3e-4),
        torch.float16: (8e-3, 8e-3),
        torch.bfloat16: (5e-2, 5e-2),
    }[reference.dtype]
    rtol, atol = tolerance
    torch.testing.assert_close(out, reference, rtol=rtol, atol=atol)
    error = (out.double() - reference.double()).abs()
    denominator = reference.double().abs().clamp_min(atol)
    return {
        "passed": True, "rtol": rtol, "atol": atol,
        "max_absolute_error": error.max().item() if error.numel() else 0.0,
        "max_relative_error_with_atol_floor": (error / denominator).max().item() if error.numel() else 0.0,
    }


@torch.inference_mode()
def measure_cuda(
    fn: Callable[[], torch.Tensor], device: torch.device,
    warmup: int = 25, iterations: int = 100, repeats: int = 7,
) -> dict:
    """Report GPU-event and synchronized host-wall averages separately.

    Contract: fn launches all its work on this device's current stream and returns
    a CUDA tensor. Cross-stream work needs explicit dependencies and a different
    measurement plan. Events may include GPU idle gaps caused by host dispatch.
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("A real NVIDIA CUDA device is required; CPU timings are not substituted.")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if min(warmup, iterations, repeats) <= 0:
        raise ValueError("warmup, iterations, and repeats must be positive")
    event_samples, wall_samples = [], []
    with torch.cuda.device(device):
        for _ in range(warmup):
            last = fn()
        if not isinstance(last, torch.Tensor) or not last.is_cuda or last.device != device:
            raise ValueError("fn must return a tensor on the selected CUDA device")
        torch.cuda.synchronize(device)
        # Create events before timing. Their initialization is forced once.
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        end.record()
        end.synchronize()
        for _ in range(repeats):
            start.record()
            for _ in range(iterations):
                last = fn()
            end.record()
            end.synchronize()
            event_samples.append(start.elapsed_time(end) / iterations)
        for _ in range(repeats):
            torch.cuda.synchronize(device)
            start_wall = time.perf_counter()
            for _ in range(iterations):
                last = fn()
            torch.cuda.synchronize(device)
            wall_samples.append((time.perf_counter() - start_wall) * 1000 / iterations)
    return {
        "gpu_event": summarize_samples(event_samples),
        "synchronized_host_wall": summarize_samples(wall_samples),
        "warmup_calls": warmup, "calls_per_repeat": iterations, "repeats": repeats,
        "measurement_scope": "Resident-input forward wrapper, including metadata checks, output allocation, and launches; JIT and numerical correctness checks excluded.",
        "p95_scope": "Quantile over repeat averages, not p95 latency of individual requests.",
        "cache_policy": "Repeatedly reuse the same inputs; caches may be warm. No explicit cache flush.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", choices=DEFAULT_SHAPES, default="matmul")
    parser.add_argument("--shape", type=parse_shape)
    parser.add_argument("--backends", default="torch,cuda,triton")
    parser.add_argument("--variant", choices=("naive", "tiled"), default="tiled")
    parser.add_argument("--dtype", choices=DTYPES, default="float32")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("No NVIDIA CUDA device is available. Use CPU tests for formulas, not GPU timing.")
    device = torch.device(args.device)
    if device.type != "cuda":
        parser.error("--device must be CUDA")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    backends = [name.strip() for name in args.backends.split(",")]
    if not backends or len(set(backends)) != len(backends) or any(x not in ("torch", "cuda", "triton") for x in backends):
        parser.error("--backends must be distinct names chosen from torch,cuda,triton")
    if "cuda" in backends and args.dtype != "float32":
        parser.error("The CUDA C++ teaching backend supports float32; choose torch,triton for half/bfloat16.")
    if "triton" in backends and os.environ.get("TRITON_INTERPRET") == "1":
        parser.error("Unset TRITON_INTERPRET for GPU timing. Use tools/check_triton_interpreter.py in a separate process for CPU math checks.")
    if min(args.warmup, args.iterations, args.repeats) <= 0:
        parser.error("warmup, iterations, and repeats must be positive")
    torch.manual_seed(args.seed)
    # Compare IEEE FP32 baselines; do not silently compare TF32 with FP32.
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    shape = args.shape or DEFAULT_SHAPES[args.op]
    with torch.cuda.device(device), torch.inference_mode():
        case = build_case(args.op, shape, DTYPES[args.dtype], device)
        functions = {name: make_callable(case, name, args.variant) for name in backends}
        checks = {name: check_output(fn, case.reference) for name, fn in functions.items()}
        # All backends are built and checked before timing any backend.
        order = list(backends)
        random.Random(args.seed).shuffle(order)
        results = {}
        for name in order:
            measured = measure_cuda(functions[name], device, args.warmup, args.iterations, args.repeats)
            seconds = measured["gpu_event"]["median_ms"] / 1000
            measured["correctness"] = checks[name]
            measured["logical_effective_GB_per_s"] = case.logical_bytes / seconds / 1e9 if case.logical_bytes is not None else None
            measured["estimated_TFLOP_per_s"] = case.flops / seconds / 1e12 if case.flops is not None else None
            results[name] = measured
    report = {
        "environment": collect_environment(),
        "case": {"op": case.op, "shape": case.shape, "dtype": args.dtype, "variant": args.variant,
                 "logical_bytes": case.logical_bytes, "estimated_flops": case.flops, "notes": case.notes},
        "measurement_order": order, "seed": args.seed, "results": results,
        "traffic_warning": "Logical bytes are an algorithmic model, not Nsight-measured HBM traffic; cache reuse can make effective bandwidth exceed HBM bandwidth.",
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
