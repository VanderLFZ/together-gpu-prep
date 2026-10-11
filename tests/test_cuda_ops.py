"""CUDA teaching kernels: correctness, contract and stream tests.

These tests are skipped without an NVIDIA GPU. A skip is NOT GPU validation.
Run from the repository root: pytest -q tests/test_cuda_ops.py
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
from gpu_prep import cuda_ops

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="NVIDIA GPU unavailable; CUDA tests not executed"
)


@pytest.fixture(scope="module", autouse=True)
def compiled_extension():
    # GPU 存在但构建失败时应报错；不能把编译错误伪装成测试跳过。
    cuda_ops.load_extension()
    torch.manual_seed(2027)


def assert_close(actual, expected, *, atol=3e-5, rtol=3e-5):
    assert actual.shape == expected.shape
    assert actual.dtype == torch.float32
    assert actual.device == expected.device
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("n", [0, 1, 31, 32, 33, 255, 256, 257, 1_000_003])
def test_vector_add_tail(n):
    x = torch.randn(n, device="cuda")
    y = torch.randn_like(x)
    assert_close(cuda_ops.vector_add(x, y), x + y, atol=0, rtol=0)


@pytest.mark.parametrize("shape", [(), (3, 7), (0, 4), (2, 0, 3)])
def test_vector_add_preserves_shape(shape):
    x = torch.randn(shape, device="cuda")
    y = torch.randn_like(x)
    assert_close(cuda_ops.vector_add(x, y), x + y, atol=0, rtol=0)


@pytest.mark.parametrize("m,n", [(5, 257), (1, 1), (0, 13), (4, 0)])
def test_bias_relu(m, n):
    x = torch.randn(m, n, device="cuda")
    bias = torch.randn(n, device="cuda")
    assert_close(cuda_ops.bias_relu(x, bias), torch.relu(x + bias), atol=0, rtol=0)


def test_bias_relu_preserves_nan():
    x = torch.tensor([[float("nan"), -2.0, 3.0]], device="cuda")
    b = torch.zeros(3, device="cuda")
    torch.testing.assert_close(
        cuda_ops.bias_relu(x, b), torch.relu(x + b), equal_nan=True, atol=0, rtol=0
    )


@pytest.mark.parametrize("m,n", [(7, 1), (5, 33), (3, 513), (2, 1001), (2, 8193), (0, 31)])
def test_row_reductions(m, n):
    x = torch.randn(m, n, device="cuda")
    assert_close(cuda_ops.row_sum(x), x.double().sum(dim=-1).float(), atol=6e-5, rtol=4e-5)
    assert_close(cuda_ops.row_max(x), x.max(dim=-1).values, atol=0, rtol=0)


def test_row_max_negative_values():
    x = -torch.rand(4, 513, device="cuda") - 0.25
    assert_close(cuda_ops.row_max(x), x.max(dim=-1).values, atol=0, rtol=0)


@pytest.mark.parametrize("m,n", [(7, 1), (5, 33), (3, 513), (2, 1001), (2, 8193), (0, 31)])
def test_softmax(m, n):
    x = torch.randn(m, n, device="cuda") * 7
    actual = cuda_ops.softmax(x)
    expected = torch.softmax(x.double(), dim=-1).float()
    assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    if m:
        torch.testing.assert_close(actual.sum(-1), torch.ones(m, device="cuda"), atol=3e-6, rtol=3e-6)


def test_softmax_extreme_finite_scores():
    x = torch.tensor(
        [[10000.0, 10001.0, 9999.0], [-10000.0, -10001.0, -9999.0], [8.0, 8.0, 8.0]],
        device="cuda",
    )
    actual = cuda_ops.softmax(x)
    assert torch.isfinite(actual).all()
    assert_close(actual, torch.softmax(x.double(), dim=-1).float(), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("op", [cuda_ops.transpose_naive, cuda_ops.transpose_tiled])
@pytest.mark.parametrize("m,n", [(1, 1), (3, 65), (31, 33), (1003, 769), (0, 13), (7, 0)])
def test_transpose(op, m, n):
    x = torch.randn(m, n, device="cuda")
    assert_close(op(x), x.t().contiguous(), atol=0, rtol=0)


@pytest.mark.parametrize("op", [cuda_ops.matmul_naive, cuda_ops.matmul_tiled])
@pytest.mark.parametrize("m,n,k", [(1, 1, 1), (7, 5, 3), (127, 65, 93), (16, 16, 16),
                                      (3, 5, 0), (0, 5, 7), (3, 0, 7), (0, 0, 0)])
def test_matmul(op, m, n, k):
    a = torch.randn(m, k, device="cuda")
    b = torch.randn(k, n, device="cuda")
    # FP64 reference 避免受 PyTorch TF32/FP32 matmul 全局设置影响。
    expected = (a.double() @ b.double()).float()
    assert_close(op(a, b), expected, atol=2e-4, rtol=5e-5)


@pytest.mark.parametrize("shape", [(1,), (7,), (2, 3, 1), (2, 3, 33), (3, 513), (0, 4, 33)])
@pytest.mark.parametrize("eps", [1e-5, 1e-3])
def test_norms(shape, eps):
    x = torch.randn(shape, device="cuda")
    w = torch.randn(shape[-1], device="cuda")
    b = torch.randn(shape[-1], device="cuda")
    xd = x.double()
    mean = xd.mean(dim=-1, keepdim=True)
    var = ((xd - mean) ** 2).mean(dim=-1, keepdim=True)
    layer_reference = ((xd - mean) * torch.rsqrt(var + eps) * w.double() + b.double()).float()
    rms_reference = (xd * torch.rsqrt((xd * xd).mean(dim=-1, keepdim=True) + eps) * w.double()).float()
    assert_close(cuda_ops.layer_norm(x, w, b, eps), layer_reference, atol=4e-5, rtol=4e-5)
    assert_close(cuda_ops.rms_norm(x, w, eps), rms_reference, atol=4e-5, rtol=4e-5)


def test_layer_norm_large_mean_small_variance():
    # 比较 two-pass 中心方差与高精度 reference；避免 E[x²]-E[x]² 的消减问题。
    x = 1000.0 + torch.randn(3, 513, device="cuda")
    w = torch.ones(513, device="cuda")
    b = torch.zeros(513, device="cuda")
    xd = x.double()
    center = xd - xd.mean(-1, keepdim=True)
    expected = (center * torch.rsqrt((center ** 2).mean(-1, keepdim=True) + 1e-5)).float()
    assert_close(cuda_ops.layer_norm(x, w, b), expected, atol=3e-4, rtol=2e-4)


def attention_oracle(q, k, v, causal, query_start):
    # 显式 absolute-position mask，避免非方阵 is_causal 的对齐语义歧义。
    scores = (q.double() @ k.double().transpose(-1, -2)) / math.sqrt(q.shape[-1])
    if causal:
        query_positions = torch.arange(q.shape[-2], device=q.device) + query_start
        key_positions = torch.arange(k.shape[-2], device=q.device)
        visible = key_positions[None, :] <= query_positions[:, None]
        scores = scores.masked_fill(~visible, float("-inf"))
    return (torch.softmax(scores, dim=-1) @ v.double()).float()


@pytest.mark.parametrize(
    "b,h,l,s,d,causal,offset",
    [(2, 3, 7, 11, 17, False, 0), (1, 2, 33, 33, 32, True, 0),
     (2, 1, 1, 65, 16, True, 64), (1, 1, 3, 8, 7, True, 5),
     (1, 1, 0, 8, 7, False, 0), (0, 2, 3, 8, 7, False, 0),
     (1, 1, 1, 1, 1, True, 0), (1, 1, 1, 4096, 8, False, 0)],
)
def test_attention(b, h, l, s, d, causal, offset):
    q = torch.randn(b, h, l, d, device="cuda")
    k = torch.randn(b, h, s, d, device="cuda")
    v = torch.randn_like(k)
    expected = attention_oracle(q, k, v, causal, offset)
    assert_close(cuda_ops.attention(q, k, v, causal, offset), expected, atol=4e-5, rtol=4e-5)


def test_attention_decode_matches_full_last_query():
    q = torch.randn(1, 2, 17, 13, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    full = cuda_ops.attention(q, k, v, causal=True)
    decode = cuda_ops.attention(q[:, :, -1:, :].contiguous(), k, v,
                               causal=True, query_start=16)
    assert_close(decode, full[:, :, -1:, :].contiguous(), atol=2e-6, rtol=2e-6)


def test_rejects_wrong_contract():
    x = torch.randn(3, 7, device="cuda")
    with pytest.raises(RuntimeError, match="float32"):
        cuda_ops.vector_add(x.half(), x.half())
    with pytest.raises(RuntimeError, match="contiguous"):
        cuda_ops.transpose_tiled(x.t())
    with pytest.raises(RuntimeError, match="identical shapes"):
        cuda_ops.vector_add(x, x[:1].contiguous())
    with pytest.raises(RuntimeError, match="CUDA tensor"):
        cuda_ops.vector_add(torch.ones(7), torch.ones(7))
    with pytest.raises(RuntimeError, match="bias must have shape"):
        cuda_ops.bias_relu(x, torch.ones(3, device="cuda"))
    with pytest.raises(RuntimeError, match="inner dimensions"):
        cuda_ops.matmul_tiled(x, x)
    for op in [cuda_ops.row_sum, cuda_ops.row_max, cuda_ops.softmax]:
        with pytest.raises(RuntimeError, match="positive"):
            op(torch.empty(3, 0, device="cuda"))
    with pytest.raises(RuntimeError, match="normalization width"):
        cuda_ops.rms_norm(torch.empty(2, 0, device="cuda"), torch.empty(0, device="cuda"))
    for eps in [0.0, -1.0, float("inf"), float("nan"), 1e-300]:
        with pytest.raises(RuntimeError, match="eps must"):
            cuda_ops.rms_norm(x, torch.ones(7, device="cuda"), eps)


def test_attention_rejects_invalid_dimensions():
    q = torch.randn(1, 1, 1, 8, device="cuda")
    with pytest.raises(RuntimeError, match="1 <= key length"):
        empty = torch.empty(1, 1, 0, 8, device="cuda")
        cuda_ops.attention(q, empty, empty)
    with pytest.raises(RuntimeError, match="1 <= key length"):
        too_long = torch.empty(1, 1, 4097, 8, device="cuda")
        cuda_ops.attention(q, too_long, too_long)
    with pytest.raises(RuntimeError, match="query_start must"):
        cuda_ops.attention(q, q, q, causal=True, query_start=-1)


def test_forward_only_guard_and_no_grad_escape():
    x = torch.randn(7, device="cuda", requires_grad=True)
    with pytest.raises(RuntimeError, match="forward-only"):
        cuda_ops.vector_add(x, x)
    with torch.no_grad():
        out = cuda_ops.vector_add(x, x)
    assert not out.requires_grad
    torch.testing.assert_close(out, 2 * x.detach())


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA devices required")
def test_device_guard_and_device_mismatch():
    x = torch.randn(257, device="cuda:1")
    y = torch.randn_like(x)
    with torch.cuda.device(0):
        out = cuda_ops.vector_add(x, y)
        assert torch.cuda.current_device() == 0
        assert out.device.index == 1
        assert_close(out, x + y, atol=0, rtol=0)
        other = torch.randn(257, device="cuda:0")
        with pytest.raises(RuntimeError, match="must be on"):
            cuda_ops.vector_add(x, other)


def test_respects_non_default_stream():
    # 先让输入在 default stream 上有确定的初始值。
    x = torch.zeros(4097, device="cuda")
    y = torch.zeros_like(x)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # 若可用，延迟 producer，增强错误使用 default stream 时的检测能力。
        # 这是测试辅助；kernel 和用户 API 均不依赖私有 _sleep。
        if hasattr(torch.cuda, "_sleep"):
            torch.cuda._sleep(5_000_000)
        x.fill_(3.0)
        y.fill_(5.0)
        out = cuda_ops.vector_add(x, y)
        row = cuda_ops.row_sum(out.view(1, -1))
        done = torch.cuda.Event()
        done.record()
    done.synchronize()
    assert_close(out, torch.full_like(x, 8.0), atol=0, rtol=0)
    assert_close(row, torch.full((1,), 4097.0 * 8.0, device="cuda"), atol=0, rtol=0)
