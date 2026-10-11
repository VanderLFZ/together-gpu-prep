"""Run with: python -m examples.torch_layers_demo --device cpu

Use --device cuda on a GPU machine. This demonstrates mathematical agreement,
not fused-kernel performance. It needs PyTorch only, not Triton or nvcc.
"""

import argparse

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Choose --device cpu for the PyTorch reference exercises.")
    torch.manual_seed(2027)
    # float64 makes small-reference comparisons easy to inspect. GPU throughput
    # exercises should choose their target precision and tolerances separately.
    dtype = torch.float64
    x = torch.randn(2, 5, 8, device=device, dtype=dtype)
    linear = ManualLinear(8, 12, device=device, dtype=dtype)
    torch.testing.assert_close(linear(x), F.linear(x, linear.weight, linear.bias))
    print("ManualLinear: forward matches F.linear; parameters:", list(dict(linear.named_parameters())))

    packed = ManualLinear(8, 24, device=device, dtype=dtype)(x)
    q, k, v = split_qkv_heads(packed, 2)
    print("QKV:", tuple(packed.shape), "->", tuple(q.shape), tuple(k.shape), tuple(v.shape))
    print("Q stride / contiguous:", q.stride(), q.is_contiguous())

    for layer in (ManualLayerNorm(8, device=device, dtype=dtype),
                  ManualRMSNorm(8, device=device, dtype=dtype)):
        out = layer(x)
        if isinstance(layer, ManualLayerNorm):
            ref = F.layer_norm(x, (8,), layer.weight, layer.bias, eps=1e-5)
        else:
            ref = F.rms_norm(x, (8,), layer.weight, eps=1e-5)
        torch.testing.assert_close(out, ref, atol=1e-10, rtol=1e-10)
        print(type(layer).__name__, "reference match; output dtype:", out.dtype)

    causal_mask = torch.ones(5, 5, device=device, dtype=torch.bool).tril()
    own = scaled_dot_product_attention(q, k, v, is_causal=True)
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask, dropout_p=0.0)
    torch.testing.assert_close(own, ref, atol=1e-10, rtol=1e-10)
    print("Manual attention: matches SDPA using an explicit causal mask")

    model = MultiHeadSelfAttention(8, 2, device=device, dtype=dtype).eval()
    cache = KVCache(2, 2, 5, 4, device=device, dtype=dtype)
    with torch.inference_mode():
        full = model(x)
        parts = [model.forward_step(x[:, :3], cache)]
        parts.extend(model.forward_step(x[:, t:t + 1], cache) for t in range(3, 5))
        incremental = torch.cat(parts, dim=1)
    torch.testing.assert_close(full, incremental, atol=1e-10, rtol=1e-10)
    print("KV cache: 3-token prefill + 2 decode steps match full causal output")
    print("Cache active length / capacity:", cache.length, "/", cache.max_seq_len)

    # Show the all-masked contract before any output projection bias.
    blocked = torch.zeros(5, 5, device=device, dtype=torch.bool)
    zero = scaled_dot_product_attention(q, k, v, attn_mask=blocked)
    assert torch.equal(zero, torch.zeros_like(zero))
    print("All-masked attention rows: zero output (before MHA output projection)")


if __name__ == "__main__":
    main()
