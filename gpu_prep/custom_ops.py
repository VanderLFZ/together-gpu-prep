"""Exercise 11: one operator integrated with fake tensors and autograd.

Import registers metadata only: it does NOT import/compile the CUDA extension.
The CPU implementation makes registration and gradient checks possible on a
CPU-only machine. It is a reference path, not a GPU performance substitute.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _validate(x: Tensor, bias: Tensor) -> None:
    if x.ndim != 2 or bias.ndim != 1:
        raise ValueError("bias_relu expects x[M,N] and bias[N]")
    if x.shape[1] != bias.shape[0]:
        raise ValueError("bias length must equal x.shape[1]")
    if x.device != bias.device:
        raise ValueError("x and bias must be on the same device")
    if x.dtype != bias.dtype:
        raise TypeError("x and bias must have the same dtype")
    if not x.is_contiguous() or not bias.is_contiguous():
        raise ValueError("this teaching operator requires contiguous inputs")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError("bias_relu supports floating tensors only")
    if x.device.type not in ("cpu", "cuda"):
        raise ValueError("bias_relu supports CPU and CUDA only")
    if x.device.type == "cuda" and x.dtype != torch.float32:
        raise TypeError("the CUDA teaching implementation supports float32 only")


@torch.library.custom_op("gpu_prep::bias_relu", mutates_args=())
def bias_relu(x: Tensor, bias: Tensor) -> Tensor:
    """Fresh contiguous y = relu(x + bias), without input mutation.

    CPU: fp16/bf16/fp32/fp64 reference. CUDA: fp32 custom kernel.
    Empty M or N is supported. Backward's derivative at exactly zero is zero.
    The backward contract and numerical tests use finite inputs.
    """
    _validate(x, bias)
    if x.device.type == "cuda":
        # Lazy import: CPU users do not need nvcc or the extension at import time.
        from gpu_prep.cuda_ops import bias_relu as cuda_bias_relu

        # The standalone CUDA exercise is explicitly forward-only; autograd is
        # supplied by this custom operator's registration below.
        with torch.no_grad():
            return cuda_bias_relu(x, bias)
    return torch.relu(x + bias)


@bias_relu.register_fake
def _bias_relu_fake(x: Tensor, bias: Tensor) -> Tensor:
    _validate(x, bias)
    # Match the real output's fresh, contiguous storage and metadata. Do not use
    # tensor values, .item(), .cpu(), or the actual CUDA kernel in a fake kernel.
    return torch.empty(x.shape, dtype=x.dtype, device=x.device)


def _setup_context(ctx, inputs, output) -> None:
    x, bias = inputs
    ctx.save_for_backward(x, bias)


def _backward(ctx, grad_output: Tensor) -> tuple[Tensor, Tensor]:
    x, bias = ctx.saved_tensors
    active = (x + bias) > 0
    # A branch is numerically different from multiplying by a 0/1 mask:
    # inactive ReLU entries must return zero even if upstream is inf or NaN.
    # Multiplication would evaluate 0 * inf / 0 * NaN and create NaN instead.
    grad_input = torch.where(active, grad_output, torch.zeros_like(grad_output))
    # Bias was broadcast over rows, so its VJP sums those rows, including M=0.
    grad_bias = grad_input.sum(dim=0)
    return grad_input, grad_bias


bias_relu.register_autograd(_backward, setup_context=_setup_context)


@torch.library.custom_op("gpu_prep::bias_relu_triton", mutates_args=())
def bias_relu_triton(x: Tensor, bias: Tensor) -> Tensor:
    """The same operator contract with a Triton GPU implementation.

    Uses the PyTorch >=2.5 custom_op interface, so its internals are opaque to
    the compiler. Newer torch.library.triton_op APIs are deliberately optional,
    not required by this package. CPU remains an ordinary reference fallback.
    """
    _validate(x, bias)
    if x.device.type == "cuda":
        from gpu_prep.triton_ops import bias_relu as triton_bias_relu

        with torch.no_grad():
            return triton_bias_relu(x, bias)
    return torch.relu(x + bias)


bias_relu_triton.register_fake(_bias_relu_fake)
bias_relu_triton.register_autograd(_backward, setup_context=_setup_context)


__all__ = ["bias_relu", "bias_relu_triton"]
