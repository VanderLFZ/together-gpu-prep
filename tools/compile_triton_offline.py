"""Compile every teaching Triton kernel for an explicit target, without a GPU.

This is compiler validation, not a numerical, race, or performance test.
The ASTSource API here is validated against Triton 3.1.0 and is not a stable
cross-version public compatibility promise. Run GPU tests on your actual stack.

Usage from an installed checkout:
    python tools/compile_triton_offline.py --arch 80 --output results/triton_offline_compile.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource

from gpu_prep import triton_ops as ops


def configurations():
    for dtype in ("fp32", "fp16", "bf16"):
        ptr = "*" + dtype
        def entry(name, pointers, constants, warps=4, stages=2):
            return {
                "kernel": name, "dtype": dtype,
                "signature": {arg: ptr for arg in pointers.split()},
                "constants": constants,
                "options": {"num_warps": warps, "num_stages": stages},
            }
        yield entry("_vector_add_kernel", "X Y Z", {"N": 1003, "BLOCK": 256})
        yield entry("_bias_relu_kernel", "X Bias Y", {"TOTAL": 3 * 769, "N": 769, "BLOCK": 256})
        for is_max in (False, True):
            yield entry("_row_reduce_kernel", "X Y", {"N": 769, "IS_MAX": is_max, "BLOCK": 1024})
        for n, block, warps in ((769, 1024, 4), (8192, 8192, 8)):
            yield entry("_softmax_kernel", "X Y", {"N": n, "BLOCK": block}, warps)
        yield entry("_transpose_naive_kernel", "X Y", {"M": 127, "N": 65, "BLOCK": 256})
        yield entry("_transpose_tiled_kernel", "X Y", {"M": 127, "N": 65, "TILE": 32})
        yield entry("_matmul_naive_kernel", "A B C", {"M": 17, "N": 33, "K": 19, "BLOCK": 128})
        yield entry("_matmul_tiled_kernel", "A B C", {"M": 127, "N": 65, "K": 93, "BM": 32, "BN": 32, "BK": 32})
        for rms in (False, True):
            yield entry("_norm_kernel", "X W Bias Y", {"N": 769, "EPS": 1e-5, "IS_RMS": rms, "BLOCK": 1024})
        for dim, causal, mask, length, source, start in (
            (16, False, False, 19, 37, 0),
            (32, True, False, 1, 65, 64),
            (64, True, True, 19, 37, 0),
            (128, False, True, 19, 37, 0),
        ):
            config = entry("_attention_kernel", "Q K V Mask O", {
                "L": length, "S": source, "D": dim, "SCALE": dim ** -0.5,
                "CAUSAL": causal, "QUERY_START": start, "HAS_MASK": mask,
                "MASK_BH_STRIDE": length * source if mask else 0,
                "BM": 16, "BN": 32,
            }, stages=1)
            if mask:
                config["signature"]["Mask"] = "*i1"
            yield config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", type=int, default=80)
    parser.add_argument("--output", type=Path, default=Path("results/triton_offline_compile.json"))
    args = parser.parse_args()
    target = GPUTarget("cuda", args.arch, 32)
    results = []
    for index, config in enumerate(configurations(), 1):
        record = dict(config)
        try:
            compiled = triton.compile(
                ASTSource(getattr(ops, config["kernel"]), signature=config["signature"], constants=config["constants"]),
                target=target, options=config["options"],
            )
            ptx, cubin = compiled.asm["ptx"], compiled.asm["cubin"]
            record.update({
                "status": "compiled", "ptx_characters": len(ptx), "cubin_bytes": len(cubin),
                "ptx_sha256": hashlib.sha256(ptx.encode()).hexdigest(),
                "cubin_sha256": hashlib.sha256(cubin).hexdigest(),
            })
            print(f"{index:02d} COMPILED {config['dtype']:4s} {config['kernel']}", flush=True)
        except Exception as exc:
            record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            print(f"{index:02d} FAILED {config['dtype']:4s} {config['kernel']}: {exc}", flush=True)
        results.append(record)
    report = {
        "triton_version": triton.__version__, "target": {"backend": "cuda", "arch": args.arch, "warp_size": 32},
        "scope": "Offline AST -> PTX -> cubin compilation only. No GPU numerical execution, race checks, or timing.",
        "successes": sum(r["status"] == "compiled" for r in results),
        "failures": sum(r["status"] != "compiled" for r in results), "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Result: {report['successes']} compiled, {report['failures']} failed; {args.output}")
    raise SystemExit(1 if report["failures"] else 0)


if __name__ == "__main__":
    main()
