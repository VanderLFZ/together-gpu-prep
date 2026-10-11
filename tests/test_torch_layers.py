"""CPU-executable mathematical and integration checks for exercises 7--9.

GPU kernel correctness/performance is deliberately not inferred from these tests.
"""

import pytest
import torch
from torch.nn import functional as F

from gpu_prep.torch_layers import (
    KVCache,
    ManualLayerNorm,
    ManualLinear,
    ManualRMSNorm,
    MultiHeadSelfAttention,
    scaled_dot_product_attention,
    split_qkv_heads,
)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(314159)


@pytest.mark.parametrize("shape", [(4,), (3, 4), (2, 3, 4), (0, 3, 4)])
@pytest.mark.parametrize("use_bias", [False, True])
def test_linear_matches_reference_forward_and_backward(shape, use_bias):
    layer = ManualLinear(4, 5, bias=use_bias, dtype=torch.float64)
    x = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    wr = layer.weight.detach().clone().requires_grad_()
    br = layer.bias.detach().clone().requires_grad_() if use_bias else None
    actual = layer(x)
    expected = F.linear(xr, wr, br)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(x.grad, xr.grad)
    torch.testing.assert_close(layer.weight.grad, wr.grad)
    if use_bias:
        torch.testing.assert_close(layer.bias.grad, br.grad)
    assert "weight" in dict(layer.named_parameters())


def test_qkv_split_preserves_feature_order_and_supports_noncontiguous_input():
    # Use visibly different numbers in each Q/K/V segment; a wrong permutation
    # can otherwise look plausible when tested only by shape.
    source = torch.arange(2 * 4 * 24, dtype=torch.float64).reshape(2, 4, 24)
    packed = source[:, ::2, :]
    assert not packed.is_contiguous()
    q, k, v = split_qkv_heads(packed, num_heads=2)
    assert q.shape == (2, 2, 2, 4)
    for b in range(2):
        for h in range(2):
            for t in range(2):
                torch.testing.assert_close(q[b, h, t], packed[b, t, h * 4 : h * 4 + 4])
                torch.testing.assert_close(k[b, h, t], packed[b, t, 8 + h * 4 : 12 + h * 4])
                torch.testing.assert_close(v[b, h, t], packed[b, t, 16 + h * 4 : 20 + h * 4])


@pytest.mark.parametrize("shape,normalized", [((4,), 4), ((2, 3, 4), 4), ((2, 3, 4), (3, 4)), ((0, 3, 4), 4)])
@pytest.mark.parametrize("norm_class", [ManualLayerNorm, ManualRMSNorm])
def test_norm_matches_pytorch_forward_and_backward(shape, normalized, norm_class):
    layer = norm_class(normalized, eps=1e-5, dtype=torch.float64)
    with torch.no_grad():
        layer.weight.uniform_(0.5, 1.5)
        if isinstance(layer, ManualLayerNorm):
            layer.bias.normal_()
    x = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    wr = layer.weight.detach().clone().requires_grad_()
    normalized_tuple = (normalized,) if isinstance(normalized, int) else normalized
    if norm_class is ManualLayerNorm:
        br = layer.bias.detach().clone().requires_grad_()
        reference = F.layer_norm(xr, normalized_tuple, wr, br, eps=1e-5)
    else:
        reference = F.rms_norm(xr, normalized_tuple, wr, eps=1e-5)
    actual = layer(x)
    torch.testing.assert_close(actual, reference, atol=1e-11, rtol=1e-11)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    reference.backward(upstream)
    torch.testing.assert_close(x.grad, xr.grad, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(layer.weight.grad, wr.grad, atol=1e-10, rtol=1e-10)
    if norm_class is ManualLayerNorm:
        torch.testing.assert_close(layer.bias.grad, br.grad, atol=1e-10, rtol=1e-10)


def test_layernorm_statistics_constant_and_single_feature():
    layer = ManualLayerNorm(4, elementwise_affine=False, dtype=torch.float64)
    x = torch.tensor([[1.0, 2.0, 4.0, 8.0]], dtype=torch.float64)
    y = layer(x)
    torch.testing.assert_close(y.mean(-1), torch.zeros(1, dtype=x.dtype), atol=1e-14, rtol=0)
    original_variance = x.var(-1, correction=0)
    expected_variance = original_variance / (original_variance + layer.eps)
    torch.testing.assert_close(y.var(-1, correction=0), expected_variance)
    constant = torch.full((2, 4), 7.0, dtype=torch.float64, requires_grad=True)
    out = layer(constant)
    assert torch.equal(out, torch.zeros_like(out))
    out.sum().backward()
    assert torch.isfinite(constant.grad).all()
    singleton = ManualLayerNorm(1, elementwise_affine=False)(torch.tensor([[3.0], [-2.0]]))
    assert torch.equal(singleton, torch.zeros_like(singleton))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("norm_class", [ManualLayerNorm, ManualRMSNorm])
def test_low_precision_norm_accumulates_in_float32(dtype, norm_class):
    layer = norm_class(7, eps=1e-5)  # fp32 parameters with a low-precision input
    x = (torch.randn(2, 3, 7) * 5).to(dtype)
    actual = layer(x)
    if norm_class is ManualLayerNorm:
        expected = F.layer_norm(x.float(), (7,), layer.weight, layer.bias, eps=1e-5).to(dtype)
    else:
        expected = F.rms_norm(x.float(), (7,), layer.weight, eps=1e-5).to(dtype)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, atol=2e-3 if dtype == torch.float16 else 2e-2, rtol=2e-3 if dtype == torch.float16 else 2e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("norm_class", [ManualLayerNorm, ManualRMSNorm])
@pytest.mark.parametrize("eps", [1e-300, 1e300])
def test_norm_rejects_epsilon_that_rounds_to_zero_or_inf_in_accumulator(dtype, norm_class, eps):
    layer = norm_class(3, eps=eps, dtype=dtype)
    with pytest.raises(ValueError, match="represented as float32"):
        layer(torch.zeros(2, 3, dtype=dtype))


@pytest.mark.parametrize("norm_class", [ManualLayerNorm, ManualRMSNorm])
def test_float64_norm_keeps_valid_tiny_epsilon(norm_class):
    layer = norm_class(3, eps=1e-300, dtype=torch.float64)
    output = layer(torch.zeros(2, 3, dtype=torch.float64))
    assert output.dtype == torch.float64
    assert torch.equal(output, torch.zeros_like(output))


@pytest.mark.parametrize("norm_class", [ManualLayerNorm, ManualRMSNorm])
def test_float32_norm_accepts_positive_representable_subnormal_epsilon(norm_class):
    # The requirement is representability, not eps >= torch.finfo(float32).tiny.
    # Check with the default denormal handling used by this CPU environment.
    layer = norm_class(3, eps=1e-40, dtype=torch.float32)
    output = layer(torch.zeros(2, 3, dtype=torch.float32))
    assert torch.equal(output, torch.zeros_like(output))


@pytest.mark.parametrize("norm_class", [ManualLayerNorm, ManualRMSNorm])
def test_norm_float64_gradcheck_including_parameters(norm_class):
    layer = norm_class(3, dtype=torch.float64)
    x = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(3, dtype=torch.float64, requires_grad=True)
    if norm_class is ManualLayerNorm:
        bias = torch.randn(3, dtype=torch.float64, requires_grad=True)
        fn = lambda a, w, b: torch.func.functional_call(layer, {"weight": w, "bias": b}, (a,))
        assert torch.autograd.gradcheck(fn, (x, weight, bias))
    else:
        fn = lambda a, w: torch.func.functional_call(layer, {"weight": w}, (a,))
        assert torch.autograd.gradcheck(fn, (x, weight))


def test_attention_matches_sdpa_with_explicit_padding_and_causal_mask():
    q = torch.randn(2, 2, 3, 4, dtype=torch.float64, requires_grad=True)
    k = torch.randn(2, 2, 5, 4, dtype=torch.float64, requires_grad=True)
    v = torch.randn(2, 2, 5, 6, dtype=torch.float64, requires_grad=True)
    # Queries have absolute positions 2,3,4. Every row has at least one key.
    causal = torch.arange(5)[None, :] <= (torch.arange(3) + 2)[:, None]
    padding = torch.tensor([[True, True, True, False, False], [True, True, True, True, True]])
    combined = causal & padding[:, None, None, :]
    actual = scaled_dot_product_attention(q, k, v, attn_mask=padding[:, None, None, :], is_causal=True, query_start=2)
    qr, kr, vr = (t.detach().clone().requires_grad_() for t in (q, k, v))
    expected = F.scaled_dot_product_attention(qr, kr, vr, attn_mask=combined, dropout_p=0.0, is_causal=False)
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    for own, ref in ((q, qr), (k, kr), (v, vr)):
        torch.testing.assert_close(own.grad, ref.grad, atol=1e-10, rtol=1e-10)


def test_non_square_causality_has_correct_absolute_offset():
    q = torch.zeros(1, 1, 2, 1, dtype=torch.float64)
    k = torch.zeros(1, 1, 4, 1, dtype=torch.float64)
    v = torch.tensor([1.0, 3.0, 7.0, 9.0], dtype=torch.float64).reshape(1, 1, 4, 1)
    out = scaled_dot_product_attention(q, k, v, is_causal=True, query_start=2)
    expected = torch.tensor([11.0 / 3.0, 5.0], dtype=torch.float64).reshape(1, 1, 2, 1)
    torch.testing.assert_close(out, expected)


def test_all_masked_attention_has_zero_output_and_finite_zero_gradients():
    q = torch.randn(1, 2, 3, 4, dtype=torch.float64, requires_grad=True)
    k = torch.randn(1, 2, 5, 4, dtype=torch.float64, requires_grad=True)
    v = torch.randn(1, 2, 5, 2, dtype=torch.float64, requires_grad=True)
    out, weights = scaled_dot_product_attention(q, k, v, attn_mask=torch.zeros(3, 5, dtype=torch.bool), return_weights=True)
    assert torch.equal(out, torch.zeros_like(out))
    assert torch.equal(weights, torch.zeros_like(weights))
    out.sum().backward()
    for tensor in (q, k, v):
        assert torch.isfinite(tensor.grad).all()
        assert torch.equal(tensor.grad, torch.zeros_like(tensor.grad))


def test_attention_gradcheck_with_partially_and_fully_masked_rows():
    args = tuple(torch.randn(*shape, dtype=torch.float64, requires_grad=True) for shape in
                 ((1, 1, 2, 2), (1, 1, 3, 2), (1, 1, 3, 2)))
    mask = torch.tensor([[True, False, True], [False, False, False]])
    fn = lambda a, b, c: scaled_dot_product_attention(a, b, c, attn_mask=mask)
    assert torch.autograd.gradcheck(fn, args)


def test_outer_autocast_does_not_downgrade_attention_accumulators():
    q = torch.randn(1, 2, 3, 4).bfloat16()
    k = torch.randn(1, 2, 5, 4).bfloat16()
    v = torch.randn(1, 2, 5, 6).bfloat16()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out, probabilities = scaled_dot_product_attention(q, k, v, return_weights=True)
    assert probabilities.dtype == torch.float32
    assert out.dtype == torch.bfloat16
    reference = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), dropout_p=0.0)
    torch.testing.assert_close(out, reference.bfloat16(), atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("length,key_length", [(0, 3), (2, 0), (0, 0)])
def test_empty_attention_preserves_shapes_and_autograd(length, key_length):
    q = torch.randn(1, 2, length, 4, dtype=torch.float64, requires_grad=True)
    k = torch.randn(1, 2, key_length, 4, dtype=torch.float64, requires_grad=True)
    v = torch.randn(1, 2, key_length, 6, dtype=torch.float64, requires_grad=True)
    out = scaled_dot_product_attention(q, k, v)
    assert out.shape == (1, 2, length, 6)
    out.sum().backward()
    for tensor in (q, k, v):
        assert tensor.grad is not None
        assert torch.equal(tensor.grad, torch.zeros_like(tensor.grad))


def test_mha_matches_reference_and_does_not_leak_future_tokens():
    model = MultiHeadSelfAttention(8, 2, dropout=0.0, dtype=torch.float64).eval()
    x = torch.randn(2, 5, 8, dtype=torch.float64)
    output = model(x)
    q, k, v = split_qkv_heads(F.linear(x, model.qkv.weight, model.qkv.bias), 2)
    mask = torch.ones(5, 5, dtype=torch.bool).tril()
    attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
    expected = F.linear(attended.transpose(1, 2).reshape(2, 5, 8), model.out_proj.weight, model.out_proj.bias)
    torch.testing.assert_close(output, expected, atol=1e-11, rtol=1e-11)
    modified = x.clone()
    modified[:, 3:] = torch.randn_like(modified[:, 3:]) * 20
    torch.testing.assert_close(model(modified)[:, :3], output[:, :3], atol=1e-11, rtol=1e-11)


def test_eval_disables_attention_dropout():
    model = MultiHeadSelfAttention(8, 2, dropout=0.8).eval()
    x = torch.randn(2, 4, 8)
    first = model(x)
    second = model(x)
    torch.testing.assert_close(first, second, atol=0, rtol=0)


def test_cache_prefill_and_decode_match_full_causal_attention():
    model = MultiHeadSelfAttention(8, 2, dtype=torch.float64).eval()
    x = torch.randn(2, 7, 8, dtype=torch.float64)
    cache = KVCache(2, 2, 7, 4, device="cpu", dtype=torch.float64)
    initial_storage = cache.key.data_ptr()
    with torch.inference_mode():
        expected = model(x)
        pieces = [model.forward_step(x[:, :3], cache)]
        pieces.append(model.forward_step(x[:, 3:4], cache))
        pieces.append(model.forward_step(x[:, 4:6], cache))  # non-square causal chunk
        pieces.append(model.forward_step(x[:, 6:7], cache))
        actual = torch.cat(pieces, dim=1)
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
    assert cache.length == 7
    assert cache.key.data_ptr() == initial_storage
    cache.reset()
    assert cache.length == 0
    assert cache.get()[0].shape[-2] == 0


def test_cache_rejects_overflow_without_changing_active_prefix():
    cache = KVCache(1, 2, 3, 4, device="cpu", dtype=torch.float32)
    with torch.inference_mode():
        block = torch.randn(1, 2, 2, 4)
        cache.append(block, block)
        saved_key = cache.get()[0].clone()
        with pytest.raises(ValueError, match="capacity"):
            cache.append(block, block)
        assert cache.length == 2
        torch.testing.assert_close(cache.get()[0], saved_key)


def test_cache_is_explicitly_inference_only():
    cache = KVCache(1, 2, 3, 4, device="cpu", dtype=torch.float32)
    model = MultiHeadSelfAttention(8, 2)
    x = torch.randn(1, 1, 8)
    with torch.no_grad(), pytest.raises(RuntimeError, match="eval"):
        model.forward_step(x, cache)
    model.eval()
    with pytest.raises(RuntimeError, match="no_grad"):
        model.forward_step(x, cache)


def test_invalid_shapes_and_masks_are_reported():
    with pytest.raises(ValueError):
        ManualLinear(0, 4)
    with pytest.raises(ValueError):
        ManualLinear(4, 3)(torch.randn(2, 5))
    with pytest.raises(ValueError):
        ManualLayerNorm(0)
    with pytest.raises(ValueError):
        ManualLayerNorm(4)(torch.randn(2, 3))
    with pytest.raises(ValueError):
        MultiHeadSelfAttention(7, 2)
    with pytest.raises(ValueError):
        split_qkv_heads(torch.randn(1, 2, 17), 2)
    q = torch.randn(1, 1, 3, 4)
    with pytest.raises(TypeError, match="boolean"):
        scaled_dot_product_attention(q, q, q, attn_mask=torch.ones(3, 3))
    with pytest.raises(ValueError, match="broadcastable"):
        scaled_dot_product_attention(q, q, q, attn_mask=torch.ones(2, 3, dtype=torch.bool))
    with pytest.raises(ValueError, match="query_start"):
        scaled_dot_product_attention(q, q, q, query_start=2)
    with pytest.raises(TypeError, match="same dtype"):
        scaled_dot_product_attention(q, q.double(), q)
