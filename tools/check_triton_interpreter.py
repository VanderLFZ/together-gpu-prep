"""Optional CPU math validation using Triton's own interpreter, never a fallback.

Run in a fresh process, before importing Triton:
    TRITON_INTERPRET=1 python tools/check_triton_interpreter.py

For the validated older Triton 3.1 interpreter, use a compatible NumPy 1.26.x
environment. Do not interpret this report as GPU numerical, race, or performance
validation. Public CUDA-only wrappers are deliberately bypassed here and remain
unchanged. No substitute emulator is implemented by this script.
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import sys

if os.environ.get("TRITON_INTERPRET") != "1":
    raise SystemExit("Start a separate process with TRITON_INTERPRET=1; this tool never selects a fallback automatically.")

import numpy as np
import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpu_prep import triton_ops as ops


@triton.jit
def _memory_probe(X, Y, N: tl.constexpr):
    idx = tl.arange(0, 16)
    value = tl.load(X + idx, idx < N, other=0)
    tl.store(Y + idx, value + 7.0, idx < N)


def attention_reference(q, k, v, causal, query_start, mask):
    scores = (q.double() @ k.double().transpose(-2, -1)) / math.sqrt(q.shape[-1])
    length, source = q.shape[2], k.shape[2]
    visible = torch.ones_like(scores, dtype=torch.bool)
    if causal:
        rows = torch.arange(length) + query_start
        cols = torch.arange(source)
        visible &= cols[None, :] <= rows[:, None]
    if mask is not None:
        visible &= mask
    scores = scores.masked_fill(~visible, -float("inf"))
    maximum = scores.amax(-1, keepdim=True)
    maximum = torch.where(torch.isfinite(maximum), maximum, 0.0)
    numerator = (scores - maximum).exp()
    denominator = numerator.sum(-1, keepdim=True)
    denominator = torch.where(denominator > 0, denominator, 1.0)
    return ((numerator / denominator) @ v.double()).float()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/triton_interpreter.json"))
    args = parser.parse_args()
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "CPU execution of Triton's own interpreter for FP32 indexing, masks, and high-level arithmetic.",
        "not_validated": [
            "GPU numerical execution", "GPU lane/layout lowering", "thread/warp synchronization and races",
            "register/shared-memory resource limits", "GPU performance", "FP16/BF16 execution",
        ],
        "public_cuda_wrappers_changed": False,
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "triton": triton.__version__, "numpy": np.__version__,
            "TRITON_INTERPRET": os.environ.get("TRITON_INTERPRET"),
        },
        "tests": [],
    }

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Detect an incompatible interpreter/numpy memory bridge before attributing
    # incorrect results to the exercise kernels. This probe is independent of them.
    x, out = torch.arange(7, dtype=torch.float32), torch.full((7,), -999.0)
    try:
        _memory_probe[(1,)](x, out, 7)
        torch.testing.assert_close(out, x + 7, atol=0, rtol=0)
        report["interpreter_memory_preflight"] = "passed"
    except Exception as exc:
        report["status"] = "blocked_interpreter_memory_preflight"
        report["interpreter_memory_preflight"] = {
            "error": f"{type(exc).__name__}: {exc}", "actual": out.tolist(), "expected": (x + 7).tolist(),
        }
        save()
        print("BLOCKED: Triton's own minimal CPU load/store probe failed; no exercise result is claimed.")
        return 2

    torch.manual_seed(2027)

    def check(name, kernel, grid, call_args, call_kwargs, actual, expected, atol=5e-5, rtol=3e-4):
        record = {"name": name, "shape": list(actual.shape), "dtype": "float32", "atol": atol, "rtol": rtol}
        try:
            kernel[grid](*call_args, **call_kwargs)
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
            record["status"] = "passed"
            record["max_absolute_error"] = (actual.double() - expected.double()).abs().max().item() if actual.numel() else 0.0
            print("PASS", name)
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            print("FAIL", name, type(exc).__name__)
        report["tests"].append(record)

    for n in (1, 257, 1001):
        x, y = torch.randn(n), torch.randn(n)
        out = torch.full_like(x, float("nan"))
        check(f"vector_add_n{n}", ops._vector_add_kernel, (triton.cdiv(n, 256),),
              (x, y, out, n), {"BLOCK": 256}, out, x + y, atol=0, rtol=0)

    x, bias = torch.randn(3, 33), torch.randn(33)
    out = torch.full_like(x, float("nan"))
    check("bias_relu_tail", ops._bias_relu_kernel, (triton.cdiv(x.numel(), 256),),
          (x, bias, out, x.numel(), 33), {"BLOCK": 256}, out, (x + bias).relu(), atol=0, rtol=0)

    for n in (1, 33, 513):
        x = torch.randn(3, n)
        block = triton.next_power_of_2(n)
        for is_max in (False, True):
            out = torch.full((3,), float("nan"))
            reference = x.amax(-1) if is_max else x.double().sum(-1).float()
            check(f"row_{'max' if is_max else 'sum'}_n{n}", ops._row_reduce_kernel, (3,),
                  (x, out, n, is_max), {"BLOCK": block}, out, reference)
        out = torch.full_like(x, float("nan"))
        check(f"softmax_n{n}", ops._softmax_kernel, (3,), (x, out, n),
              {"BLOCK": block}, out, x.double().softmax(-1).float())

    x = torch.tensor([[1000., 1001., -1000.], [17., 17., 17.]])
    out = torch.full_like(x, float("nan"))
    check("softmax_large_logits", ops._softmax_kernel, (2,), (x, out, 3),
          {"BLOCK": 4}, out, x.double().softmax(-1).float())

    for m, n in ((31, 33), (65, 17)):
        x = torch.randn(m, n)
        for tiled in (False, True):
            out = torch.full((n, m), float("nan"))
            kernel = ops._transpose_tiled_kernel if tiled else ops._transpose_naive_kernel
            grid = (triton.cdiv(m, 32) * triton.cdiv(n, 32),) if tiled else (triton.cdiv(m * n, 256),)
            config = {"TILE": 32} if tiled else {"BLOCK": 256}
            check(f"transpose_{'tiled' if tiled else 'naive'}_{m}x{n}", kernel, grid,
                  (x, out, m, n), config, out, x.T.contiguous(), atol=0, rtol=0)

    for m, n, k in ((7, 3, 19), (31, 17, 33), (5, 9, 0)):
        a, b = torch.randn(m, k), torch.randn(k, n)
        reference = (a.double() @ b.double()).float()
        for tiled in (False, True):
            out = torch.full((m, n), float("nan"))
            kernel = ops._matmul_tiled_kernel if tiled else ops._matmul_naive_kernel
            grid = (triton.cdiv(m, 32) * triton.cdiv(n, 32),) if tiled else (triton.cdiv(m * n, 128),)
            config = {"BM": 32, "BN": 32, "BK": 32} if tiled else {"BLOCK": 128}
            check(f"matmul_{'tiled' if tiled else 'naive'}_{m}x{n}x{k}", kernel, grid,
                  (a, b, out, m, n, k), config, out, reference)

    for shifted in (False, True):
        x = torch.randn(3, 33) + (100.0 if shifted else 0.0)
        weight, bias = torch.randn(33), torch.randn(33)
        xd, wd, bd = x.double(), weight.double(), bias.double()
        for is_rms in (False, True):
            out = torch.full_like(x, float("nan"))
            if is_rms:
                reference = (xd * torch.rsqrt((xd * xd).mean(-1, keepdim=True) + 1e-5) * wd).float()
            else:
                centered = xd - xd.mean(-1, keepdim=True)
                reference = (centered * torch.rsqrt((centered * centered).mean(-1, keepdim=True) + 1e-5) * wd + bd).float()
            check(f"{'rms' if is_rms else 'layer'}_norm_shifted{shifted}", ops._norm_kernel, (3,),
                  (x, weight, bias, out, 33, 1e-5, is_rms), {"BLOCK": 64}, out, reference, atol=1e-4)

    attention_cases = [
        ("noncausal_tail_d16", 17, 37, 16, False, 0, None),
        ("causal_tail_d32", 19, 37, 32, True, 0, None),
        ("decode_offset_d64", 1, 37, 64, True, 36, None),
        ("offset_d128", 7, 65, 128, True, 23, None),
    ]
    mask = torch.ones(19, 65, dtype=torch.bool)
    mask[0] = False
    mask[1] = False
    mask[1, 40] = True
    mask[2, 32:] = False
    attention_cases.append(("shared_mask_empty_rows_and_tiles", 19, 65, 32, False, 0, mask))
    batched = mask.expand(2, 2, 19, 65).clone()
    batched[1, 1, 3] = False
    attention_cases.append(("batched_mask_with_causal_offset", 19, 65, 32, True, 50, batched))
    for name, length, source, dim, causal, query_start, mask in attention_cases:
        q = torch.randn(2, 2, length, dim)
        k, v = torch.randn(2, 2, source, dim), torch.randn(2, 2, source, dim)
        out = torch.full_like(q, float("nan"))
        mask_stride = length * source if mask is not None and mask.ndim == 4 else 0
        mask_pointer = q if mask is None else mask
        reference = attention_reference(q, k, v, causal, query_start, mask)
        check(f"attention_{name}", ops._attention_kernel, (triton.cdiv(length, 16) * 4,),
              (q, k, v, mask_pointer, out, length, source, dim, 1.0 / math.sqrt(dim),
               causal, query_start, mask is not None, mask_stride),
              {"BM": 16, "BN": 32}, out, reference)

    passed = sum(test["status"] == "passed" for test in report["tests"])
    report["passed"] = passed
    report["failed"] = len(report["tests"]) - passed
    report["status"] = "passed_interpreter_math_only" if report["failed"] == 0 else "failed_interpreter_checks"
    save()
    print(f"{passed}/{len(report['tests'])} interpreter math checks passed; no GPU execution or performance was measured.")
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
