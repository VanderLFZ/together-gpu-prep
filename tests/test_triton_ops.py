"""Correctness tests requiring a real NVIDIA GPU; absent GPU/Triton => skip.

These tests intentionally include awkward shapes and explicit reference math.
They test correctness, not speed. Their existence is not evidence that a GPU
run has happened in the environment in which this package was authored.
"""

import importlib.util
import math

import pytest

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise  # An installed but broken dependency is not an absent-package skip.
    torch = None

if torch is None:
    SKIP_REASON = "PyTorch is not installed"
elif importlib.util.find_spec("triton") is None:
    SKIP_REASON = "Triton is not installed"
elif not torch.cuda.is_available() or torch.version.hip is not None:
    SKIP_REASON = "Triton exercises require an NVIDIA CUDA GPU"
elif torch.cuda.get_device_capability()[0] < 8:
    SKIP_REASON = "Triton exercises target NVIDIA compute capability >= 8.0"
else:
    SKIP_REASON = ""

# Keep collecting tests even without a GPU so running this file alone exits
# normally with explicit skips, instead of pytest's exit 5 (nothing collected).
pytestmark = pytest.mark.skipif(bool(SKIP_REASON), reason=SKIP_REASON)
if not SKIP_REASON:
    from gpu_prep import triton_ops as ops

DTYPES = (torch.float32, torch.float16, torch.bfloat16) if torch is not None else ("torch_missing",)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(2027)


def _randn(shape, dtype):
    return torch.randn(shape, dtype=dtype, device="cuda")


def _assert_close(actual, reference, *, kind="normal"):
    assert actual.shape == reference.shape
    assert actual.dtype == reference.dtype
    assert actual.device == reference.device
    if actual.dtype == torch.float32:
        atol, rtol = (3e-4, 5e-5) if kind == "sum" else (4e-5, 2e-4)
    elif actual.dtype == torch.float16:
        atol, rtol = (1e-2, 3e-3)
    else:
        atol, rtol = (3e-2, 2e-2)
    torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", [0, 1, 31, 32, 33, 255, 256, 257, 1_000_003])
def test_vector_add(dtype, n):
    x, y = _randn((n,), dtype), _randn((n,), dtype)
    out = ops.vector_add(x, y)
    reference = (x.float() + y.float()).to(dtype)
    torch.testing.assert_close(out, reference, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(0, 7), (3, 0), (1, 1), (3, 33), (5, 513)])
def test_bias_relu(dtype, shape):
    x, bias = _randn(shape, dtype), _randn((shape[1],), dtype)
    reference = (x.float() + bias.float()).clamp_min(0).to(dtype)
    torch.testing.assert_close(ops.bias_relu(x, bias), reference, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(0, 7), (3, 1), (5, 33), (3, 513), (7, 1001), (2, 8192)])
def test_row_reductions(dtype, shape):
    x = _randn(shape, dtype)
    # Double precision separates reference arithmetic from the kernel's reduction order.
    _assert_close(ops.row_sum(x), x.double().sum(dim=-1).to(dtype), kind="sum")
    torch.testing.assert_close(ops.row_max(x), x.amax(dim=-1), atol=0, rtol=0)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(0, 7), (3, 1), (5, 33), (3, 513), (7, 1001), (2, 8192)])
def test_softmax(dtype, shape):
    x = _randn(shape, dtype)
    reference = torch.softmax(x.double(), dim=-1).to(dtype)
    actual = ops.softmax(x)
    _assert_close(actual, reference)
    if actual.numel():
        torch.testing.assert_close(
            actual.float().sum(dim=-1), torch.ones(shape[0], device="cuda"),
            atol=1e-2 if dtype == torch.bfloat16 else 1e-3, rtol=0,
        )


@pytest.mark.parametrize("dtype", DTYPES)
def test_softmax_stability(dtype):
    x = torch.tensor(
        [[1000, 1001, -1000], [17, 17, 17], [-1000, -1001, -1002]],
        dtype=dtype, device="cuda",
    )
    actual = ops.softmax(x)
    assert torch.isfinite(actual).all()
    _assert_close(actual, torch.softmax(x.double(), dim=-1).to(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(0, 23), (3, 0), (1, 1), (31, 33), (1003, 769)])
@pytest.mark.parametrize("name", ["transpose_naive", "transpose_tiled"])
def test_transpose(dtype, shape, name):
    x = _randn(shape, dtype)
    actual = getattr(ops, name)(x)
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, x.T.contiguous(), atol=0, rtol=0)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("m,k,n", [
    (0, 3, 7), (7, 3, 0), (5, 0, 9), (1, 1, 1),
    (7, 19, 3), (31, 33, 17), (127, 93, 65),
])
@pytest.mark.parametrize("name", ["matmul_naive", "matmul_tiled"])
def test_matmul(dtype, m, k, n, name):
    a, b = _randn((m, k), dtype), _randn((k, n), dtype)
    reference = (a.double() @ b.double()).to(dtype)
    actual = getattr(ops, name)(a, b)
    _assert_close(actual, reference)
    if k == 0:
        assert torch.count_nonzero(actual) == 0


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(1,), (33,), (0, 7), (2, 3, 1), (2, 3, 33), (3, 1001), (2, 8192)])
def test_norms(dtype, shape):
    x = _randn(shape, dtype)
    weight, bias = _randn((shape[-1],), dtype), _randn((shape[-1],), dtype)
    eps = 1e-5
    xf, wf, bf = x.double(), weight.double(), bias.double()
    mean = xf.mean(dim=-1, keepdim=True)
    variance = ((xf - mean) ** 2).mean(dim=-1, keepdim=True)
    layer_reference = ((xf - mean) * torch.rsqrt(variance + eps) * wf + bf).to(dtype)
    rms_reference = (xf * torch.rsqrt((xf * xf).mean(dim=-1, keepdim=True) + eps) * wf).to(dtype)
    _assert_close(ops.layer_norm(x, weight, bias, eps), layer_reference)
    _assert_close(ops.rms_norm(x, weight, eps), rms_reference)


@pytest.mark.parametrize("dtype", DTYPES)
def test_norm_constant_row_and_padding(dtype):
    # A constant row must stay well behaved when variance is zero.
    x = torch.full((3, 33), 7.0, dtype=dtype, device="cuda")
    w, b = _randn((33,), dtype), _randn((33,), dtype)
    actual = ops.layer_norm(x, w, b)
    torch.testing.assert_close(actual, b.expand_as(x), atol=2e-3, rtol=2e-3)


def test_norm_padding_does_not_pollute_variance():
    # With a large nonzero mean, mistakenly counting padded (0-mean)^2 values
    # would dominate the variance and almost erase the normalized signal.
    x = _randn((3, 33), torch.float32) + 100.0
    w = torch.ones(33, device="cuda")
    b = torch.zeros(33, device="cuda")
    xf = x.double()
    centered = xf - xf.mean(dim=-1, keepdim=True)
    reference = (centered * torch.rsqrt((centered * centered).mean(dim=-1, keepdim=True) + 1e-5)).float()
    torch.testing.assert_close(ops.layer_norm(x, w, b), reference, atol=1e-4, rtol=1e-4)


def _attention_reference(q, k, v, *, causal=False, query_start=0, attn_mask=None):
    """Explicit double-precision reference with the same all-masked-row policy."""
    _, _, length, d = q.shape
    s = k.shape[2]
    scores = (q.double() @ k.double().transpose(-2, -1)) / math.sqrt(d)
    visible = torch.ones_like(scores, dtype=torch.bool)
    if causal:
        query_position = query_start + torch.arange(length, device=q.device)
        key_position = torch.arange(s, device=q.device)
        visible = visible & (key_position[None, :] <= query_position[:, None])
    if attn_mask is not None:
        visible = visible & attn_mask
    scores = scores.masked_fill(~visible, -float("inf"))
    maximum = scores.amax(dim=-1, keepdim=True)
    safe_maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    numerator = torch.exp(scores - safe_maximum)
    denominator = numerator.sum(dim=-1, keepdim=True)
    safe_denominator = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    probabilities = numerator / safe_denominator
    return (probabilities @ v.double()).to(q.dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("b,h,length,s,d", [
    (1, 2, 1, 13, 16), (2, 2, 17, 37, 32),
    (1, 1, 33, 17, 64), (1, 2, 7, 65, 128),
])
@pytest.mark.parametrize("causal,query_start", [(False, 0), (True, 0), (True, 23)])
def test_attention(dtype, b, h, length, s, d, causal, query_start):
    q = _randn((b, h, length, d), dtype)
    k, v = _randn((b, h, s, d), dtype), _randn((b, h, s, d), dtype)
    actual = ops.attention(q, k, v, causal=causal, query_start=query_start)
    reference = _attention_reference(q, k, v, causal=causal, query_start=query_start)
    _assert_close(actual, reference)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batched_mask", [False, True])
def test_attention_masked_rows_and_empty_early_tiles(dtype, batched_mask):
    b, h, length, s, d = 2, 2, 19, 65, 32
    q = _randn((b, h, length, d), dtype)
    k, v = _randn((b, h, s, d), dtype), _randn((b, h, s, d), dtype)
    mask = torch.ones((length, s), dtype=torch.bool, device="cuda")
    mask[0, :] = False             # Empty row for every key tile.
    mask[1, :] = False
    mask[1, 40] = True             # First tile empty; only a later key becomes visible.
    mask[2, 32:] = False           # Valid early tile, then entirely empty later tiles.
    if batched_mask:
        mask = mask.expand(b, h, length, s).clone()
        mask[1, 1, 3, :] = False   # A different empty row in just one B/H slice.
    actual = ops.attention(q, k, v, attn_mask=mask)
    reference = _attention_reference(q, k, v, attn_mask=mask)
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual[:, :, 0]) == 0
    _assert_close(actual[:, :, 1], v[:, :, 40])
    _assert_close(actual, reference)


def test_attention_causal_and_explicit_mask_are_combined():
    q = _randn((1, 2, 17, 32), torch.float32)
    k, v = _randn((1, 2, 37, 32), torch.float32), _randn((1, 2, 37, 32), torch.float32)
    mask = torch.rand((17, 37), device="cuda") > 0.4
    mask[0] = False
    actual = ops.attention(q, k, v, causal=True, query_start=20, attn_mask=mask)
    reference = _attention_reference(q, k, v, causal=True, query_start=20, attn_mask=mask)
    _assert_close(actual, reference)


def test_decode_offset_matches_last_row_of_full_causal_attention():
    q, k, v = [_randn((1, 2, 37, 32), torch.float32) for _ in range(3)]
    full = ops.attention(q, k, v, causal=True)
    decode = ops.attention(q[:, :, -1:].contiguous(), k, v, causal=True, query_start=36)
    _assert_close(decode, full[:, :, -1:])


@pytest.mark.parametrize("b,h,length", [(0, 2, 17), (1, 0, 17), (1, 2, 0)])
def test_attention_empty_output(b, h, length):
    q = _randn((b, h, length, 32), torch.float32)
    k, v = [_randn((b, h, 7, 32), torch.float32) for _ in range(2)]
    actual = ops.attention(q, k, v)
    assert actual.shape == q.shape and actual.numel() == 0


@pytest.mark.parametrize("name", ["row_sum", "row_max", "softmax"])
@pytest.mark.parametrize("n", [0, 8193])
def test_reduction_width_validation(name, n):
    with pytest.raises(ValueError, match="row width"):
        getattr(ops, name)(_randn((3, n), torch.float32))


def test_reject_noncontiguous_inputs_and_bad_shapes():
    x = _randn((4, 6), torch.float32)
    with pytest.raises(ValueError, match="contiguous"):
        ops.row_sum(x.T)
    with pytest.raises(ValueError, match="identical shapes"):
        ops.vector_add(x.flatten(), x.flatten()[:3])
    with pytest.raises(ValueError, match="bias must have shape"):
        ops.bias_relu(x, _randn((5,), x.dtype))
    with pytest.raises(ValueError, match="a.shape"):
        ops.matmul_tiled(x, _randn((5, 2), x.dtype))
    with pytest.raises(ValueError, match="same device and dtype"):
        ops.vector_add(x.flatten(), x.flatten().half())


def test_forward_only_and_no_grad_inference():
    x = _randn((17,), torch.float32).requires_grad_()
    y = _randn((17,), torch.float32)
    with pytest.raises(ValueError, match="forward-only"):
        ops.vector_add(x, y)
    with torch.no_grad():
        actual = ops.vector_add(x, y)
    assert not actual.requires_grad
    torch.testing.assert_close(actual, x.detach() + y, atol=0, rtol=0)


@pytest.mark.parametrize("eps", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_norm_eps(eps):
    x, w = _randn((3, 7), torch.float32), _randn((7,), torch.float32)
    with pytest.raises(ValueError, match="eps"):
        ops.rms_norm(x, w, eps)


def test_invalid_norm_width():
    x, w = _randn((3, 0), torch.float32), _randn((0,), torch.float32)
    with pytest.raises(ValueError, match="normalized dimension"):
        ops.rms_norm(x, w)


def test_attention_input_validation():
    q = _randn((1, 2, 3, 32), torch.float32)
    k, v = [_randn((1, 2, 7, 32), torch.float32) for _ in range(2)]
    with pytest.raises(ValueError, match="positions"):
        ops.attention(q, k, v, query_start=-1)
    with pytest.raises(TypeError, match="Python int"):
        ops.attention(q, k, v, query_start=True)
    with pytest.raises(ValueError, match="length S"):
        ops.attention(q, k[:, :, :0].contiguous(), v[:, :, :0].contiguous())
    with pytest.raises(ValueError, match="head dimension"):
        ops.attention(q[..., :24].contiguous(), k[..., :24].contiguous(), v[..., :24].contiguous())
    with pytest.raises(ValueError, match="bool"):
        ops.attention(q, k, v, attn_mask=torch.ones(3, 7, device="cuda"))
    with pytest.raises(ValueError, match="shape"):
        ops.attention(q, k, v, attn_mask=torch.ones(1, 1, 3, 7, dtype=torch.bool, device="cuda"))


def test_respects_nondefault_stream():
    x = torch.zeros(1001, device="cuda")
    y = torch.zeros_like(x)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        if hasattr(torch.cuda, "_sleep"):
            torch.cuda._sleep(1_000_000)  # Delay producer to expose a wrong default-stream launch.
        x.fill_(3.0)
        y.fill_(5.0)
        actual = ops.vector_add(x, y)
    stream.synchronize()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, torch.full_like(actual, 8.0), atol=0, rtol=0)


@pytest.mark.skipif(torch is None or torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_nondefault_device_and_current_device_restore():
    original_device = torch.cuda.current_device()
    other_device = 1 if original_device == 0 else 0
    if torch.cuda.get_device_capability(other_device)[0] < 8:
        pytest.skip("other device is older than the supported Triton target")
    x = torch.randn(257, device=f"cuda:{other_device}")
    y = torch.randn_like(x)
    actual = ops.vector_add(x, y)
    assert torch.cuda.current_device() == original_device
    torch.testing.assert_close(actual, x + y, atol=0, rtol=0)
