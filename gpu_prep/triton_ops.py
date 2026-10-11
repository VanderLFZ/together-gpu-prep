"""Readable forward-only Triton kernels for GPU interview preparation.

Contract
--------
* NVIDIA CUDA, compute capability >= 8.0, contiguous tensors only.
* float32 / float16 / bfloat16; all floating operands share dtype and device.
* Reductions, statistics, GEMM accumulators, and attention states use float32.
* Floating outputs retain the input dtype. Row width is capped at 8192 for
  one-program-per-row reduction, softmax, and normalization exercises.
* These are forward kernels, not differentiable PyTorch operators. Calls with
  requires_grad inputs are rejected while grad mode is enabled. Use no_grad()
  for inference, or supply a separately implemented autograd rule.
* This module does not auto-copy noncontiguous inputs or scan their values.
  Unless otherwise documented, numerical examples assume finite inputs and no
  intermediate overflow. No NaN/Inf propagation compatibility is promised.

The code is intentionally original and compact enough to explain at a board.
Fixed tiles make it easier to study; they are not claimed to be optimal.
"""

import math
from numbers import Real
import struct

import torch
import triton
import triton.language as tl


MAX_ROW_WIDTH = 8192
MAX_TENSOR_ELEMENTS = 2**31 - 1  # Valid element offsets fit i32; tail accesses are masked.
SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
ATTENTION_HEAD_DIMS = (16, 32, 64, 128)

__all__ = [
    "vector_add", "bias_relu", "row_sum", "row_max", "softmax",
    "transpose_naive", "transpose_tiled", "matmul_naive", "matmul_tiled",
    "layer_norm", "rms_norm", "attention",
]


def _check_tensor(x: torch.Tensor, name: str, ndim: int | None = None) -> None:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if ndim is not None and x.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions; got {x.ndim}")
    if not x.is_cuda or torch.version.hip is not None:
        raise ValueError(f"{name} must be on an NVIDIA CUDA device")
    if x.dtype not in SUPPORTED_DTYPES:
        raise TypeError(f"{name} must have dtype float32, float16, or bfloat16")
    if not x.is_contiguous():
        raise ValueError(f"{name} must be contiguous; call contiguous() explicitly")
    if x.numel() > MAX_TENSOR_ELEMENTS:
        raise ValueError(f"{name} exceeds this exercise's i32 indexing limit")
    if torch.is_grad_enabled() and x.requires_grad:
        raise ValueError("forward-only kernel: detach inputs or use torch.no_grad()")
    if torch.cuda.get_device_capability(x.device)[0] < 8:
        raise ValueError("these Triton exercises require NVIDIA compute capability >= 8.0")


def _check_peer(x: torch.Tensor, y: torch.Tensor, name: str, ndim: int | None = None) -> None:
    _check_tensor(y, name, ndim)
    if x.device != y.device or x.dtype != y.dtype:
        raise ValueError(f"{name} must have the same device and dtype as the first input")


def _row_shape(x: torch.Tensor) -> tuple[int, int]:
    _check_tensor(x, "x", ndim=2)
    m, n = x.shape
    if n == 0 or n > MAX_ROW_WIDTH:
        raise ValueError(f"row width must be in [1, {MAX_ROW_WIDTH}]")
    return m, n


def _row_launch(n: int) -> tuple[int, int]:
    block = triton.next_power_of_2(n)
    return block, 4 if block <= 2048 else 8


def _check_eps(eps: float) -> float:
    if isinstance(eps, bool) or not isinstance(eps, Real):
        raise TypeError("eps must be a positive finite real number")
    eps = float(eps)
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be positive and finite")
    # Python floats have a wider exponent range than the FP32 kernel constant.
    # Reject values that disappear to zero or overflow when converted, without
    # allocating a tensor or synchronizing/reading any device data.
    try:
        eps_fp32 = struct.unpack(">f", struct.pack(">f", eps))[0]
    except OverflowError as exc:
        raise ValueError("eps must remain positive and finite when represented as float32") from exc
    if not math.isfinite(eps_fp32) or eps_fp32 <= 0:
        raise ValueError("eps must remain positive and finite when represented as float32")
    return eps_fp32


@triton.jit
def _vector_add_kernel(X, Y, Z, N: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = idx < N
    x = tl.load(X + idx, valid, other=0).to(tl.float32)
    y = tl.load(Y + idx, valid, other=0).to(tl.float32)
    tl.store(Z + idx, x + y, valid)


def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """z = x + y for contiguous one-dimensional tensors of identical shape."""
    _check_tensor(x, "x", ndim=1)
    _check_peer(x, y, "y", ndim=1)
    if x.shape != y.shape:
        raise ValueError("x and y must have identical shapes")
    out = torch.empty_like(x)
    if x.numel() == 0:
        return out
    with torch.cuda.device(x.device):
        _vector_add_kernel[(triton.cdiv(x.numel(), 256),)](
            x, y, out, x.numel(), BLOCK=256, num_warps=4,
        )
    return out


@triton.jit
def _bias_relu_kernel(X, Bias, Y, TOTAL: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = idx < TOTAL
    x = tl.load(X + idx, valid, other=0).to(tl.float32)
    b = tl.load(Bias + idx % N, valid, other=0).to(tl.float32)
    tl.store(Y + idx, tl.maximum(x + b, 0.0), valid)


def bias_relu(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """y[m,n] = max(x[m,n] + bias[n], 0); no materialized broadcast tensor."""
    _check_tensor(x, "x", ndim=2)
    _check_peer(x, bias, "bias", ndim=1)
    if bias.shape[0] != x.shape[1]:
        raise ValueError("bias must have shape [x.shape[1]]")
    out = torch.empty_like(x)
    if x.numel() == 0:  # Also prevents taking a remainder by N=0 inside the kernel.
        return out
    with torch.cuda.device(x.device):
        _bias_relu_kernel[(triton.cdiv(x.numel(), 256),)](
            x, bias, out, x.numel(), x.shape[1], BLOCK=256, num_warps=4,
        )
    return out


@triton.jit
def _row_reduce_kernel(X, Y, N: tl.constexpr, IS_MAX: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    if IS_MAX:
        values = tl.load(X + row * N + col, col < N, other=-float("inf")).to(tl.float32)
        result = tl.max(values, axis=0)
    else:
        values = tl.load(X + row * N + col, col < N, other=0).to(tl.float32)
        result = tl.sum(values, axis=0)
    tl.store(Y + row, result)


def _row_reduce(x: torch.Tensor, is_max: bool) -> torch.Tensor:
    m, n = _row_shape(x)
    out = torch.empty((m,), dtype=x.dtype, device=x.device)
    if m == 0:
        return out
    block, warps = _row_launch(n)
    with torch.cuda.device(x.device):
        _row_reduce_kernel[(m,)](x, out, n, is_max, BLOCK=block, num_warps=warps)
    return out


def row_sum(x: torch.Tensor) -> torch.Tensor:
    """Sum each row in float32 and cast the result back to x.dtype; N must be > 0."""
    return _row_reduce(x, False)


def row_max(x: torch.Tensor) -> torch.Tensor:
    """Maximum of each nonempty row; output dtype equals x.dtype."""
    return _row_reduce(x, True)


@triton.jit
def _softmax_kernel(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    values = tl.load(X + row * N + col, col < N, other=-float("inf")).to(tl.float32)
    shifted = values - tl.max(values, axis=0)
    numerator = tl.exp(shifted)
    result = numerator / tl.sum(numerator, axis=0)
    tl.store(Y + row * N + col, result, col < N)


def softmax(x: torch.Tensor) -> torch.Tensor:
    """Stable row softmax for finite inputs; 1 <= N <= MAX_ROW_WIDTH.

    This standalone operation has no mask argument. A row consisting only of
    -inf is outside its finite-input contract. attention() separately defines
    its boolean-mask all-masked-row behavior as a zero output.
    """
    m, n = _row_shape(x)
    out = torch.empty_like(x)
    if m == 0:
        return out
    block, warps = _row_launch(n)
    with torch.cuda.device(x.device):
        _softmax_kernel[(m,)](x, out, n, BLOCK=block, num_warps=warps)
    return out


@triton.jit
def _transpose_naive_kernel(X, Y, M: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = idx // N, idx % N
    value = tl.load(X + idx, idx < M * N, other=0)
    tl.store(Y + col * M + row, value, idx < M * N)


@triton.jit
def _transpose_tiled_kernel(X, Y, M: tl.constexpr, N: tl.constexpr, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tiles_n = tl.cdiv(N, TILE)
    row = (pid // tiles_n) * TILE + tl.arange(0, TILE)
    col = (pid % tiles_n) * TILE + tl.arange(0, TILE)
    values = tl.load(
        X + row[:, None] * N + col[None, :],
        (row[:, None] < M) & (col[None, :] < N), other=0,
    )
    # Express the output tile in [output-row, output-column] order.
    # Triton, rather than Python code, chooses lane layout and any conversions.
    tl.store(
        Y + col[:, None] * M + row[None, :], tl.trans(values),
        (col[:, None] < N) & (row[None, :] < M),
    )


def _transpose(x: torch.Tensor, tiled: bool) -> torch.Tensor:
    _check_tensor(x, "x", ndim=2)
    m, n = x.shape
    out = torch.empty((n, m), dtype=x.dtype, device=x.device)
    if x.numel() == 0:
        return out
    with torch.cuda.device(x.device):
        if tiled:
            _transpose_tiled_kernel[(triton.cdiv(m, 32) * triton.cdiv(n, 32),)](
                x, out, m, n, TILE=32, num_warps=4,
            )
        else:
            _transpose_naive_kernel[(triton.cdiv(x.numel(), 256),)](
                x, out, m, n, BLOCK=256, num_warps=4,
            )
    return out


def transpose_naive(x: torch.Tensor) -> torch.Tensor:
    """Materialize x.T using flattened input indices; educational strided stores."""
    return _transpose(x, False)


def transpose_tiled(x: torch.Tensor) -> torch.Tensor:
    """Materialize x.T with 2D tiles; this is not an explicit CUDA shared-memory tile."""
    return _transpose(x, True)


@triton.jit
def _matmul_naive_kernel(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                         BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = idx // N, idx % N
    valid = idx < M * N
    acc = tl.full((BLOCK,), 0, tl.float32)
    # A sequential K loop makes the repeated global loads easy to see.
    for k in range(0, K):
        a = tl.load(A + row * K + k, valid, other=0).to(tl.float32)
        b = tl.load(B + k * N + col, valid, other=0).to(tl.float32)
        acc = tl.fma(a, b, acc)
    tl.store(C + idx, acc, valid)


@triton.jit
def _matmul_tiled_kernel(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    tiles_n = tl.cdiv(N, BN)
    row = (pid // tiles_n) * BM + tl.arange(0, BM)
    col = (pid % tiles_n) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for tile in range(0, tl.cdiv(K, BK)):
        k = tile * BK + inner
        a = tl.load(A + row[:, None] * K + k[None, :],
                    (row[:, None] < M) & (k[None, :] < K), other=0)
        b = tl.load(B + k[:, None] * N + col[None, :],
                    (k[:, None] < K) & (col[None, :] < N), other=0)
        # For f32 inputs, do not silently compare TF32 math with a strict f32 baseline.
        acc = tl.dot(a, b, acc, input_precision="ieee")
    tl.store(C + row[:, None] * N + col[None, :], acc,
             (row[:, None] < M) & (col[None, :] < N))


def _matmul(a: torch.Tensor, b: torch.Tensor, tiled: bool) -> torch.Tensor:
    _check_tensor(a, "a", ndim=2)
    _check_peer(a, b, "b", ndim=2)
    m, k = a.shape
    kb, n = b.shape
    if k != kb:
        raise ValueError("a.shape[1] must equal b.shape[0]")
    if m * n > MAX_TENSOR_ELEMENTS:
        raise ValueError("output exceeds this exercise's i32 indexing limit")
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    if m == 0 or n == 0:
        return out
    if k == 0:
        return out.zero_()
    with torch.cuda.device(a.device):
        if tiled:
            _matmul_tiled_kernel[(triton.cdiv(m, 32) * triton.cdiv(n, 32),)](
                a, b, out, m, n, k, BM=32, BN=32, BK=32,
                num_warps=4, num_stages=2,
            )
        else:
            _matmul_naive_kernel[(triton.cdiv(m * n, 128),)](
                a, b, out, m, n, k, BLOCK=128, num_warps=4,
            )
    return out


def matmul_naive(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """A[M,K] @ B[K,N]; sequential K loop per output, FP32 accumulation."""
    return _matmul(a, b, False)


def matmul_tiled(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """A[M,K] @ B[K,N]; fixed 32x32x32 tiles, FP32 accumulation, f32 IEEE math."""
    return _matmul(a, b, True)


@triton.jit
def _norm_kernel(X, W, Bias, Y, N: tl.constexpr, EPS: tl.constexpr,
                 IS_RMS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    valid = col < N
    x = tl.load(X + row * N + col, valid, other=0).to(tl.float32)
    w = tl.load(W + col, valid, other=0).to(tl.float32)
    if IS_RMS:
        second_moment = tl.sum(x * x, axis=0) / N
        normalized = x * tl.rsqrt(second_moment + EPS)
        out = normalized * w
    else:
        mean = tl.sum(x, axis=0) / N
        # Padding must contribute zero, not (0 - mean)^2, to the variance.
        centered = tl.where(valid, x - mean, 0.0)
        variance = tl.sum(centered * centered, axis=0) / N
        normalized = centered * tl.rsqrt(variance + EPS)
        b = tl.load(Bias + col, valid, other=0).to(tl.float32)
        out = normalized * w + b
    tl.store(Y + row * N + col, out, valid)


def _norm_shape(x: torch.Tensor, weight: torch.Tensor, eps: float) -> tuple[int, int, float]:
    _check_tensor(x, "x")
    if x.ndim < 1:
        raise ValueError("x must have at least one dimension")
    n = x.shape[-1]
    if n == 0 or n > MAX_ROW_WIDTH:
        raise ValueError(f"normalized dimension must be in [1, {MAX_ROW_WIDTH}]")
    _check_peer(x, weight, "weight", ndim=1)
    if weight.shape[0] != n:
        raise ValueError("weight must have shape [x.shape[-1]]")
    return x.numel() // n, n, _check_eps(eps)


def layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
               eps: float = 1e-5) -> torch.Tensor:
    """Last-dimension LayerNorm; population variance (divide by N), affine=True."""
    rows, n, eps = _norm_shape(x, weight, eps)
    _check_peer(x, bias, "bias", ndim=1)
    if bias.shape != weight.shape:
        raise ValueError("bias must have the same shape as weight")
    out = torch.empty_like(x)
    if rows == 0:
        return out
    block, warps = _row_launch(n)
    with torch.cuda.device(x.device):
        _norm_kernel[(rows,)](
            x, weight, bias, out, n, eps, False, BLOCK=block, num_warps=warps,
        )
    return out


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Last-dimension RMSNorm with learnable weight, no bias, explicit eps."""
    rows, n, eps = _norm_shape(x, weight, eps)
    out = torch.empty_like(x)
    if rows == 0:
        return out
    block, warps = _row_launch(n)
    with torch.cuda.device(x.device):
        # Bias is unused in the IS_RMS specialization, so weight is a harmless placeholder.
        _norm_kernel[(rows,)](
            x, weight, weight, out, n, eps, True, BLOCK=block, num_warps=warps,
        )
    return out


@triton.jit
def _attention_kernel(Q, K, V, Mask, O, L: tl.constexpr, S: tl.constexpr,
                      D: tl.constexpr, SCALE: tl.constexpr,
                      CAUSAL: tl.constexpr, QUERY_START: tl.constexpr,
                      HAS_MASK: tl.constexpr, MASK_BH_STRIDE: tl.constexpr,
                      BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    q_blocks = tl.cdiv(L, BM)
    q_block = pid % q_blocks
    bh = pid // q_blocks
    r = q_block * BM + tl.arange(0, BM)
    d = tl.arange(0, D)
    q = tl.load(Q + bh * L * D + r[:, None] * D + d[None, :],
                r[:, None] < L, other=0)

    # Three online-softmax states for each query: maximum, denominator, numerator.
    running_max = tl.full((BM,), -float("inf"), tl.float32)
    denominator = tl.full((BM,), 0.0, tl.float32)
    numerator = tl.full((BM, D), 0.0, tl.float32)

    for tile in range(0, tl.cdiv(S, BN)):
        c = tile * BN + tl.arange(0, BN)
        k = tl.load(K + bh * S * D + c[None, :] * D + d[:, None],
                    c[None, :] < S, other=0)
        v = tl.load(V + bh * S * D + c[:, None] * D + d[None, :],
                    c[:, None] < S, other=0).to(tl.float32)
        scores = tl.dot(q, k, input_precision="ieee") * SCALE
        visible = (r[:, None] < L) & (c[None, :] < S)
        if CAUSAL:
            visible = visible & (c[None, :] <= QUERY_START + r[:, None])
        if HAS_MASK:
            keep = tl.load(Mask + bh * MASK_BH_STRIDE + r[:, None] * S + c[None, :],
                           (r[:, None] < L) & (c[None, :] < S), other=0)
            visible = visible & keep
        scores = tl.where(visible, scores, -float("inf"))
        new_max = tl.maximum(running_max, tl.max(scores, axis=1))

        # Before any key has been visible, new_max is -inf. Replace it only in
        # exponent calculations so we never evaluate -inf - (-inf).
        safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
        old_scale = tl.exp(running_max - safe_max)
        probabilities = tl.exp(scores - safe_max[:, None])
        denominator = denominator * old_scale + tl.sum(probabilities, axis=1)
        numerator = numerator * old_scale[:, None]
        # Keep the probabilities in FP32. This deliberately sacrifices some
        # Tensor Core throughput to avoid a hidden FP16/BF16 probability cast.
        numerator = tl.dot(probabilities, v, numerator, input_precision="ieee")
        running_max = new_max

    safe_denominator = tl.where(denominator > 0.0, denominator, 1.0)
    output = numerator / safe_denominator[:, None]
    tl.store(O + bh * L * D + r[:, None] * D + d[None, :], output, r[:, None] < L)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              causal: bool = False, query_start: int = 0, *,
              attn_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Tiled online-softmax attention forward, without materializing [L,S].

    q: [B,H,L,D], k/v: [B,H,S,D], S>0, D in {16,32,64,128}.
    Keys have positions 0..S-1. For causal=True, query i has absolute position
    query_start+i and sees key j exactly when j <= query_start+i.
    query_start is explicit: decode with a length-S cache usually uses S-1.

    Optional bool attn_mask is [L,S] shared across B/H or exactly [B,H,L,S].
    True means visible; it is ANDed with the causal mask. All-masked rows return
    zeros, by this exercise's explicit convention. No additive mask, dropout,
    GQA, custom scale, backward, or arbitrary-stride inputs are implemented.
    """
    _check_tensor(q, "q", ndim=4)
    _check_peer(q, k, "k", ndim=4)
    _check_peer(q, v, "v", ndim=4)
    b, h, length, d = q.shape
    if k.shape != v.shape or k.shape[:2] != (b, h) or k.shape[-1] != d:
        raise ValueError("expected q[B,H,L,D] and matching k/v[B,H,S,D]")
    s = k.shape[2]
    if s <= 0:
        raise ValueError("attention key/value length S must be positive")
    if d not in ATTENTION_HEAD_DIMS:
        raise ValueError(f"head dimension D must be one of {ATTENTION_HEAD_DIMS}")
    if not isinstance(causal, bool):
        raise TypeError("causal must be bool")
    if isinstance(query_start, bool) or not isinstance(query_start, int):
        raise TypeError("query_start must be a nonnegative Python int")
    if query_start < 0 or query_start + length > MAX_TENSOR_ELEMENTS:
        raise ValueError("query positions must be nonnegative and fit i32")

    mask_stride = 0
    if attn_mask is not None:
        if not isinstance(attn_mask, torch.Tensor):
            raise TypeError("attn_mask must be a bool torch.Tensor")
        if attn_mask.device != q.device or attn_mask.dtype != torch.bool:
            raise ValueError("attn_mask must be bool and on q.device")
        if not attn_mask.is_contiguous():
            raise ValueError("attn_mask must be contiguous")
        if attn_mask.numel() > MAX_TENSOR_ELEMENTS:
            raise ValueError("attn_mask exceeds this exercise's i32 indexing limit")
        if attn_mask.shape == (length, s):
            mask_stride = 0
        elif attn_mask.shape == (b, h, length, s):
            mask_stride = length * s
        else:
            raise ValueError("attn_mask must have shape [L,S] or exactly [B,H,L,S]")

    out = torch.empty_like(q)
    if q.numel() == 0:
        return out
    mask_pointer = q if attn_mask is None else attn_mask
    with torch.cuda.device(q.device):
        _attention_kernel[(triton.cdiv(length, 16) * b * h,)](
            q, k, v, mask_pointer, out, length, s, d, 1.0 / math.sqrt(d),
            causal, query_start, attn_mask is not None, mask_stride,
            BM=16, BN=32, num_warps=4, num_stages=1,
        )
    return out
