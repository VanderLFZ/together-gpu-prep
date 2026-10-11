"""Run a common correctness demonstration on the real GPU selected by the user."""
from __future__ import annotations

import argparse
import importlib
import os

import torch

from gpu_prep.benchmark import DEFAULT_SHAPES, build_case, check_output, make_callable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cuda", "triton", "all"), default="all")
    parser.add_argument("--op", choices=(*DEFAULT_SHAPES, "all"), default="all")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.backend in ("triton", "all") and os.environ.get("TRITON_INTERPRET") == "1":
        parser.error("Unset TRITON_INTERPRET for the GPU demo; use tools/check_triton_interpreter.py separately.")
    if not torch.cuda.is_available():
        parser.error("This demonstration requires a real NVIDIA GPU. Run pytest on CPU for the PyTorch exercises.")
    device = torch.device(args.device)
    if device.type != "cuda":
        parser.error("--device must be a CUDA device")
    torch.manual_seed(7)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    names = ("cuda", "triton") if args.backend == "all" else (args.backend,)
    operations = tuple(DEFAULT_SHAPES) if args.op == "all" else (args.op,)
    with torch.cuda.device(device), torch.inference_mode():
        for op in operations:
            case = build_case(op, DEFAULT_SHAPES[op], torch.float32, device)
            variants = ("naive", "tiled") if op in ("transpose", "matmul") else ("tiled",)
            for backend in names:
                for variant in variants:
                    check = check_output(make_callable(case, backend, variant), case.reference)
                    label = f"{op}_{variant}" if op in ("transpose", "matmul") else op
                    print(f"PASS {backend:6s} {label:18s} shape={case.shape} max_abs={check['max_absolute_error']:.3g}")
        if args.op in ("all", "attention"):
            # A non-square causal decode example that a square-only test would miss.
            q, k, v = (torch.randn(shape, device=device) for shape in ((1, 2, 1, 32), (1, 2, 17, 32), (1, 2, 17, 32)))
            reference = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
            for backend in names:
                module = importlib.import_module(f"gpu_prep.{backend}_ops")
                out = module.attention(q, k, v, causal=True, query_start=16)
                torch.testing.assert_close(out, reference, atol=3e-4, rtol=3e-4)
                print(f"PASS {backend:6s} causal decode: L=1, S=17, query_start=16")


if __name__ == "__main__":
    main()
