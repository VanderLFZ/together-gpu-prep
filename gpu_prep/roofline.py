"""Exercise 6: an explicit, ideal-traffic roofline model.

All throughput inputs use decimal SI units: 1 TFLOP/s = 1e12 FLOP/s,
1 TB/s = 1e12 byte/s. Memory *capacity* may be displayed in GiB elsewhere.
These functions make mathematical estimates; they do not measure a GPU.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import json
import math


@dataclass(frozen=True)
class Workload:
    name: str
    flops: int
    ideal_bytes: int
    assumptions: str

    @property
    def arithmetic_intensity(self) -> float:
        return self.flops / self.ideal_bytes if self.ideal_bytes else 0.0


def _dims(*values: int) -> None:
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in values):
        raise ValueError("Dimensions must be non-negative integers.")


def _element_bytes(value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("element_bytes must be a positive integer.")


def vector_add_workload(n: int, element_bytes: int = 4) -> Workload:
    _dims(n)
    _element_bytes(element_bytes)
    return Workload(
        "vector_add", n, 3 * n * element_bytes,
        "Read each of two inputs once, write one output once; ignore caching and launch latency.",
    )


def gemm_workload(m: int, n: int, k: int, element_bytes: int = 4) -> Workload:
    _dims(m, n, k)
    _element_bytes(element_bytes)
    # If the output is empty, a well-written wrapper skips the operation.
    if m == 0 or n == 0:
        return Workload("gemm", 0, 0, "Empty output: skip the launch.")
    return Workload(
        "gemm", 2 * m * n * k, element_bytes * (m * k + k * n + m * n),
        "Conventional 2MNK FLOP count, beta=0; load each A/B element once, store each C once. "
        "This ideal reuse is not guaranteed by a real kernel.",
    )


def roofline_estimate(workload: Workload, peak_tflops: float, bandwidth_tbps: float) -> dict:
    if not math.isfinite(peak_tflops) or peak_tflops <= 0:
        raise ValueError("peak_tflops must be finite and positive.")
    if not math.isfinite(bandwidth_tbps) or bandwidth_tbps <= 0:
        raise ValueError("bandwidth_tbps must be finite and positive.")
    peak = peak_tflops * 1e12
    bandwidth = bandwidth_tbps * 1e12
    compute_seconds = workload.flops / peak
    memory_seconds = workload.ideal_bytes / bandwidth
    lower_bound = max(compute_seconds, memory_seconds)
    ceiling = workload.flops / lower_bound / 1e12 if lower_bound else 0.0
    return {
        **asdict(workload),
        "arithmetic_intensity_flop_per_byte": workload.arithmetic_intensity,
        "ridge_point_flop_per_byte": peak / bandwidth,
        "hypothetical_peak_tflops": peak_tflops,
        "hypothetical_bandwidth_tbps": bandwidth_tbps,
        "ideal_compute_time_ms": compute_seconds * 1e3,
        "ideal_memory_time_ms": memory_seconds * 1e3,
        "ideal_time_lower_bound_ms": lower_bound * 1e3,
        "roofline_tflops": ceiling,
        "model_limiting_resource": (
            "empty" if workload.ideal_bytes == 0 and workload.flops == 0
            else "memory" if memory_seconds >= compute_seconds else "compute"
        ),
        "measured": False,
    }


def amdahl_speedup(fraction: float, local_speedup: float) -> float:
    """Serial critical-path model, with no new overhead and fixed workload."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must lie in [0, 1].")
    if not math.isfinite(local_speedup) or local_speedup <= 0:
        raise ValueError("local_speedup must be finite and positive.")
    return 1.0 / ((1.0 - fraction) + fraction / local_speedup)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peak-tflops", type=float, default=100.0,
                        help="Hypothetical peak for the SAME datatype/compute path.")
    parser.add_argument("--bandwidth-tbps", type=float, default=2.0)
    args = parser.parse_args()
    workloads = [
        vector_add_workload(1_000_000),
        gemm_workload(1, 4096, 4096),
        gemm_workload(1024, 1024, 1024),
        gemm_workload(127, 65, 93),
    ]
    print(json.dumps({
        "kind": "mathematical_estimates_not_measurements",
        "examples": [
            roofline_estimate(w, args.peak_tflops, args.bandwidth_tbps)
            for w in workloads
        ],
        "amdahl_example": {
            "original_time_fraction": 0.05,
            "local_speedup": 2.0,
            "end_to_end_speedup": amdahl_speedup(0.05, 2.0),
            "note": "Speedup factor and percent latency reduction are different quantities.",
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
