"""Tests for benchmark accounting; these do not simulate GPU performance."""
import pytest
import torch

from gpu_prep.benchmark import build_case, parse_shape, summarize_samples


def test_summary_is_repeat_mean_quantile():
    result = summarize_samples([1.0, 2.0, 4.0])
    assert result["median_ms"] == 2.0
    assert result["p95_of_repeat_means_ms"] == pytest.approx(3.8)
    assert result["samples_ms"] == [1.0, 2.0, 4.0]


@pytest.mark.parametrize("op,shape", [
    ("vector_add", (13,)), ("bias_relu", (3, 7)), ("row_sum", (3, 7)),
    ("row_max", (3, 7)), ("softmax", (3, 7)), ("transpose", (3, 7)),
    ("matmul", (3, 7, 5)), ("layer_norm", (3, 7)), ("rms_norm", (3, 7)),
    ("attention", (1, 2, 3, 7, 4)),
])
def test_reference_and_torch_baseline_describe_same_operation(op, shape):
    # Verifies shapes, oracle equations, and timed baseline semantics on CPU.
    # It does not exercise CUDA-event measurement.
    case = build_case(op, shape, torch.float32, torch.device("cpu"))
    torch.testing.assert_close(case.torch_fn(), case.reference, atol=1e-5, rtol=1e-5)


def test_physical_transpose_reference():
    case = build_case("transpose", (3, 7), torch.float32, torch.device("cpu"))
    assert case.torch_fn().is_contiguous()
    assert case.logical_bytes == 2 * 3 * 7 * 4


def test_shape_order_is_m_n_k():
    case = build_case("matmul", parse_shape("3,7,5"), torch.float32, torch.device("cpu"))
    assert tuple(case.inputs[0].shape) == (3, 5)
    assert tuple(case.inputs[1].shape) == (5, 7)
    assert case.flops == 2 * 3 * 7 * 5
