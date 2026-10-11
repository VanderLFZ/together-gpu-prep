import math
import pytest
from gpu_prep.roofline import (
    amdahl_speedup, gemm_workload, roofline_estimate, vector_add_workload,
)


def test_vector_add_units_and_roofline():
    result = roofline_estimate(vector_add_workload(1_000_000), 100.0, 2.0)
    assert result["arithmetic_intensity_flop_per_byte"] == pytest.approx(1 / 12)
    assert result["ridge_point_flop_per_byte"] == pytest.approx(50)
    assert result["ideal_time_lower_bound_ms"] == pytest.approx(0.006)
    assert result["roofline_tflops"] == pytest.approx(1 / 6)


def test_gemm_changes_regime_with_shape():
    small_m = roofline_estimate(gemm_workload(1, 4096, 4096), 100, 2)
    square = roofline_estimate(gemm_workload(1024, 1024, 1024), 100, 2)
    assert small_m["model_limiting_resource"] == "memory"
    assert square["model_limiting_resource"] == "compute"
    assert square["arithmetic_intensity_flop_per_byte"] == pytest.approx(1024 / 6)


def test_empty_work_and_zero_k_have_distinct_traffic():
    assert roofline_estimate(gemm_workload(0, 3, 4), 100, 2)["ideal_time_lower_bound_ms"] == 0
    zero_k = gemm_workload(2, 3, 0)
    assert zero_k.flops == 0
    assert zero_k.ideal_bytes == 24


def test_amdahl_limits_and_latency_vs_throughput():
    assert amdahl_speedup(0, 2) == 1
    assert amdahl_speedup(1, 2) == 2
    speedup = amdahl_speedup(0.05, 2)
    assert speedup == pytest.approx(1 / 0.975)
    assert 1 - 1 / speedup == pytest.approx(0.025)


@pytest.mark.parametrize("bad", [-1, 1.5, True])
def test_dimensions_are_explicit(bad):
    with pytest.raises(ValueError):
        gemm_workload(bad, 3, 4)


@pytest.mark.parametrize("bad", [0, -1, math.inf, math.nan])
def test_invalid_roofline_hardware(bad):
    with pytest.raises(ValueError):
        roofline_estimate(vector_add_workload(10), bad, 2)
