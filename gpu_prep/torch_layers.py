"""Readable PyTorch implementations for interview exercises 7--9.

These are correctness-oriented teaching implementations, not fused production
kernels.  Half and bfloat16 statistics/attention math use float32; float64 is
preserved so numerical gradient checks remain meaningful.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)

# Round-to-nearest-even float32 boundaries, computed as ordinary host doubles.
# Half the smallest subnormal rounds to zero; the overflow midpoint rounds to
# infinity. Strict inequalities are equivalent to a finite, positive float32
# conversion, without a device Tensor, .item(), or nontraceable struct calls.
_F32_ZERO_ROUNDING_BOUNDARY = 2.0 ** -150
_F32_INFINITY_ROUNDING_BOUNDARY = 2.0 ** 128 - 2.0 ** 103


def _check_float(x: Tensor, name: str) -> None:
    if not isinstance(x, Tensor) or x.dtype not in _FLOAT_DTYPES:
        raise TypeError(f"{name} must be a floating Tensor (fp16/bf16/fp32/fp64)")


def _accumulator(x: Tensor) -> Tensor:
    return x.float() if x.dtype in (torch.float16, torch.bfloat16) else x


def _check_accumulator_eps(eps: float, dtype: torch.dtype) -> None:
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if dtype == torch.float32 and not (
        _F32_ZERO_ROUNDING_BOUNDARY < eps < _F32_INFINITY_ROUNDING_BOUNDARY
    ):
        raise ValueError("eps must remain finite and positive when represented as float32")


def _normalized_shape(shape: int | Sequence[int]) -> tuple[int, ...]:
    result = (shape,) if isinstance(shape, int) else tuple(shape)
    if not result or any(not isinstance(d, int) or d <= 0 for d in result):
        raise ValueError("normalized_shape must contain positive integers")
    return result


def _check_normalized_input(x: Tensor, shape: tuple[int, ...]) -> None:
    _check_float(x, "x")
    if x.ndim < len(shape) or tuple(x.shape[-len(shape) :]) != shape:
        raise ValueError(f"expected input trailing shape {shape}, got {tuple(x.shape)}")


class ManualLinear(nn.Module):
    """y = x @ weight.T + bias, with genuinely registered Parameters.

    Input shape is (..., in_features); arbitrary leading dimensions, including
    empty ones, are supported.  Feature dimensions themselves must be positive.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        self.in_features = in_features
        self.out_features = out_features
        factory = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty(out_features, in_features, **factory))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        _check_float(x, "x")
        if x.ndim == 0 or x.shape[-1] != self.in_features:
            raise ValueError(f"last input dimension must be {self.in_features}")
        out = x @ self.weight.transpose(0, 1)
        return out if self.bias is None else out + self.bias


def split_qkv_heads(qkv: Tensor, num_heads: int) -> tuple[Tensor, Tensor, Tensor]:
    """[B,T,3*D] -> three [B,H,T,D/H] tensors.

    Projection output layout is [all Q features, all K features, all V features]
    for each token. Returned tensors are views when the input layout permits;
    no contiguity promise is made after permute.
    """
    _check_float(qkv, "qkv")
    if qkv.ndim != 3 or num_heads <= 0:
        raise ValueError("qkv must be [B,T,3*D] and num_heads must be positive")
    if qkv.shape[-1] == 0 or qkv.shape[-1] % (3 * num_heads) != 0:
        raise ValueError("qkv last dimension must be a positive multiple of 3*num_heads")
    batch, length, packed_dim = qkv.shape
    head_dim = packed_dim // (3 * num_heads)
    # [B,T,3,H,d] -> [3,B,H,T,d]; unbind selects Q, K and V.
    packed = qkv.reshape(batch, length, 3, num_heads, head_dim)
    q, k, v = packed.permute(2, 0, 3, 1, 4).unbind(0)
    return q, k, v


class ManualLayerNorm(nn.Module):
    """LayerNorm with population variance, explicit epsilon and autograd.

    Statistics reduce all normalized_shape axes. Empty leading dimensions are
    supported; an empty reduction domain is rejected at construction time.
    """

    def __init__(self, normalized_shape: int | Sequence[int], eps: float = 1e-5,
                 elementwise_affine: bool = True, bias: bool = True,
                 device=None, dtype=None) -> None:
        super().__init__()
        self.normalized_shape = _normalized_shape(normalized_shape)
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape, device=device, dtype=dtype))
            if bias:
                self.bias = nn.Parameter(torch.zeros(self.normalized_shape, device=device, dtype=dtype))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        _check_normalized_input(x, self.normalized_shape)
        acc = _accumulator(x)
        _check_accumulator_eps(self.eps, acc.dtype)
        axes = tuple(range(acc.ndim - len(self.normalized_shape), acc.ndim))
        mean = acc.mean(dim=axes, keepdim=True)
        centered = acc - mean
        # mean of squared deviations is correction=0, not sample variance.
        variance = centered.square().mean(dim=axes, keepdim=True)
        out = centered * torch.rsqrt(variance + self.eps)
        if self.weight is not None:
            out = out * self.weight.to(dtype=acc.dtype)
        if self.bias is not None:
            out = out + self.bias.to(dtype=acc.dtype)
        return out.to(dtype=x.dtype)


class ManualRMSNorm(nn.Module):
    """RMSNorm without mean subtraction; epsilon is deliberately explicit."""

    def __init__(self, normalized_shape: int | Sequence[int], eps: float = 1e-5,
                 elementwise_affine: bool = True, device=None, dtype=None) -> None:
        super().__init__()
        self.normalized_shape = _normalized_shape(normalized_shape)
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape, device=device, dtype=dtype))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: Tensor) -> Tensor:
        _check_normalized_input(x, self.normalized_shape)
        acc = _accumulator(x)
        _check_accumulator_eps(self.eps, acc.dtype)
        axes = tuple(range(acc.ndim - len(self.normalized_shape), acc.ndim))
        out = acc * torch.rsqrt(acc.square().mean(dim=axes, keepdim=True) + self.eps)
        if self.weight is not None:
            out = out * self.weight.to(dtype=acc.dtype)
        return out.to(dtype=x.dtype)


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Tensor | None = None,
    is_causal: bool = False,
    query_start: int = 0,
    dropout_p: float = 0.0,
    training: bool = False,
    return_weights: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Explicit, differentiable attention with safe all-masked rows.

    q: [B,H,L,d]; k: [B,H,S,d]; v: [B,H,S,dv]. Batch/head broadcasting and GQA
    are intentionally not implicit. All three inputs share dtype and device.

    A BOOLEAN mask is broadcastable to [B,H,L,S], with True = allowed. For a
    [B,S] key padding mask, explicitly pass mask[:,None,None,:]. Causality means
    key_position <= query_start + local_query_position. Keys start at position 0.
    All-masked rows, and the S=0 case, produce zero pre-projection outputs with
    finite zero gradients for finite inputs/logits. The explicit S=0 convention
    differs from the restricted CUDA/Triton teaching kernels in this package.

    Attention probabilities (if requested) use the accumulator dtype. Returned
    output uses q.dtype. Optional dropout is applied only when training=True.
    """
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        _check_float(tensor, name)
        if tensor.ndim != 4:
            raise ValueError(f"{name} must have four dimensions [B,H,T,d]")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k and v must be on the same device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise TypeError("q, k and v must have the same dtype")
    if q.shape[:2] != k.shape[:2] or q.shape[:2] != v.shape[:2]:
        raise ValueError("q, k and v must have matching batch and head dimensions")
    if q.shape[-1] <= 0 or v.shape[-1] <= 0 or q.shape[-1] != k.shape[-1]:
        raise ValueError("q/k head dimensions must match and head dimensions must be positive")
    if k.shape[-2] != v.shape[-2]:
        raise ValueError("k and v sequence lengths must match")
    if not isinstance(query_start, int) or query_start < 0:
        raise ValueError("query_start must be a nonnegative integer")
    if not is_causal and query_start != 0:
        raise ValueError("query_start is meaningful only with is_causal=True")
    if not 0.0 <= dropout_p < 1.0:
        raise ValueError("dropout_p must lie in [0,1)")
    if attn_mask is not None:
        if attn_mask.dtype != torch.bool:
            raise TypeError("attn_mask must be boolean, with True meaning allowed")
        if attn_mask.device != q.device:
            raise ValueError("attn_mask must be on the same device as q")
    q_acc, k_acc, v_acc = _accumulator(q), _accumulator(k), _accumulator(v)
    # Explicit float32 operands alone are not sufficient inside an outer AMP
    # region: matmul can be autocast back to half/bfloat16. Disable autocast for
    # these reference contractions so our advertised accumulator policy holds.
    with torch.autocast(device_type=q.device.type, enabled=False):
        scores = (q_acc @ k_acc.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
    query_len, key_len = q.shape[-2], k.shape[-2]
    allowed = torch.ones((query_len, key_len), dtype=torch.bool, device=q.device)
    if is_causal:
        query_positions = query_start + torch.arange(query_len, device=q.device)
        key_positions = torch.arange(key_len, device=q.device)
        allowed = key_positions[None, :] <= query_positions[:, None]
    if attn_mask is not None:
        try:
            user_allowed = torch.broadcast_to(attn_mask, scores.shape)
        except RuntimeError as error:
            raise ValueError("attn_mask is not broadcastable to [B,H,L,S]") from error
        allowed = allowed & user_allowed
    if key_len == 0:
        # Matrix products on the empty contraction domain give zero outputs and
        # preserve the gradient connection to all three (possibly empty) inputs.
        with torch.autocast(device_type=q.device.type, enabled=False):
            out = (scores @ v_acc).to(dtype=q.dtype)
        return (out, scores) if return_weights else out
    masked_scores = scores.masked_fill(~allowed, float("-inf"))
    has_allowed_key = allowed.any(dim=-1, keepdim=True)
    # Never run softmax on an all -inf row. Replacing NaNs after softmax is too
    # late: its backward can still contain NaN. Make the input safe beforehand.
    safe_scores = torch.where(has_allowed_key, masked_scores, torch.zeros_like(masked_scores))
    row_max = safe_scores.amax(dim=-1, keepdim=True)
    exponentials = torch.exp(safe_scores - row_max)
    exponentials = torch.where(allowed, exponentials, torch.zeros_like(exponentials))
    # A finite nonempty row has an exponent equal to 1. An empty masked row has
    # denominator 0; clamping to 1 gives the explicitly chosen all-zero result.
    probabilities = exponentials / exponentials.sum(dim=-1, keepdim=True).clamp_min(1.0)
    if training and dropout_p:
        probabilities = F.dropout(probabilities, p=dropout_p, training=True)
    with torch.autocast(device_type=q.device.type, enabled=False):
        out = (probabilities @ v_acc).to(dtype=q.dtype)
    return (out, probabilities) if return_weights else out


class KVCache:
    """Preallocated, contiguous K/V storage for one attention layer and batch.

    This teaching cache has a single shared sequence length for the whole batch.
    It is inference-only, append-only, and not thread-safe. No paging, eviction,
    RoPE, prefix sharing or per-request ragged lengths are hidden inside it.
    Allocate one cache per layer and independent active batch/session.
    """

    def __init__(self, batch_size: int, num_heads: int, max_seq_len: int,
                 head_dim: int, device, dtype: torch.dtype) -> None:
        dimensions = (batch_size, num_heads, max_seq_len, head_dim)
        if any(not isinstance(d, int) for d in dimensions):
            raise TypeError("cache dimensions must be integers")
        if batch_size < 0 or max_seq_len < 0 or num_heads <= 0 or head_dim <= 0:
            raise ValueError("invalid cache dimensions")
        if dtype not in _FLOAT_DTYPES:
            raise TypeError("cache dtype must be floating point")
        shape = (batch_size, num_heads, max_seq_len, head_dim)
        self.key = torch.empty(shape, device=device, dtype=dtype)
        self.value = torch.empty(shape, device=device, dtype=dtype)
        self.length = 0
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.device = self.key.device
        self.dtype = dtype

    def reset(self) -> None:
        # Old storage is not zeroed; get() and append() expose only the active
        # prefix. External code must not treat unused capacity as valid tokens.
        self.length = 0

    def get(self) -> tuple[Tensor, Tensor]:
        return self.key[:, :, : self.length], self.value[:, :, : self.length]

    def append(self, key: Tensor, value: Tensor) -> tuple[Tensor, Tensor]:
        if torch.is_grad_enabled():
            raise RuntimeError("KVCache is inference-only; use torch.no_grad() or torch.inference_mode()")
        if key.ndim != 4 or value.shape != key.shape:
            raise ValueError("new key/value must have matching [B,H,T,d] shapes")
        if key.shape[:2] != (self.batch_size, self.num_heads) or key.shape[-1] != self.head_dim:
            raise ValueError("new key/value shape does not match this cache")
        if key.device != self.device or value.device != self.device:
            raise ValueError("new key/value device does not match this cache")
        if key.dtype != self.dtype or value.dtype != self.dtype:
            raise TypeError("new key/value dtype does not match this cache")
        end = self.length + key.shape[-2]
        if end > self.max_seq_len:
            raise ValueError(f"cache capacity exceeded: {end} > {self.max_seq_len}")
        # All metadata/capacity checks happen before either mutation.
        self.key[:, :, self.length : end].copy_(key)
        self.value[:, :, self.length : end].copy_(value)
        self.length = end
        return self.get()


class MultiHeadSelfAttention(nn.Module):
    """Packed QKV + manual attention + output projection, batch-first."""

    def __init__(self, embed_dim: int, num_heads: int, bias: bool = True,
                 dropout: float = 0.0, device=None, dtype=None) -> None:
        super().__init__()
        if embed_dim <= 0 or num_heads <= 0 or embed_dim % num_heads:
            raise ValueError("positive embed_dim must be divisible by positive num_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0,1)")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.qkv = ManualLinear(embed_dim, 3 * embed_dim, bias=bias, device=device, dtype=dtype)
        self.out_proj = ManualLinear(embed_dim, embed_dim, bias=bias, device=device, dtype=dtype)

    def _project(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if x.ndim != 3 or x.shape[-1] != self.embed_dim:
            raise ValueError(f"x must be [B,T,{self.embed_dim}]")
        return split_qkv_heads(self.qkv(x), self.num_heads)

    def _merge(self, x: Tensor) -> Tensor:
        batch, _, length, _ = x.shape
        return self.out_proj(x.transpose(1, 2).reshape(batch, length, self.embed_dim))

    def forward(self, x: Tensor, attn_mask: Tensor | None = None,
                is_causal: bool = True, return_weights: bool = False
                ) -> Tensor | tuple[Tensor, Tensor]:
        q, k, v = self._project(x)
        attended, weights = scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=is_causal,
            dropout_p=self.dropout, training=self.training, return_weights=True,
        )
        out = self._merge(attended)
        # An all-masked row is zero BEFORE out_proj; a nonzero output bias may
        # make this module's final row nonzero. This is intentional and explicit.
        return (out, weights) if return_weights else out

    def forward_step(self, x: Tensor, cache: KVCache) -> Tensor:
        """Append a prefill chunk or new decode token(s), then attend to cache.

        Requires eval() AND a no_grad/inference_mode context. When a chunk starts
        after p cached tokens, query i is at absolute position p+i, not i.
        """
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("forward_step requires model.eval() and torch.no_grad()/inference_mode()")
        if not isinstance(cache, KVCache):
            raise TypeError("cache must be a KVCache")
        if cache.num_heads != self.num_heads or cache.head_dim != self.head_dim:
            raise ValueError("cache head layout does not match this attention module")
        q, new_k, new_v = self._project(x)
        query_start = cache.length
        all_k, all_v = cache.append(new_k, new_v)
        attended = scaled_dot_product_attention(
            q, all_k, all_v, is_causal=True, query_start=query_start,
            dropout_p=0.0, training=False,
        )
        return self._merge(attended)
