#pragma once

#include <ATen/ATen.h>
#include <cstdint>

// 原始教学实现：连续 CUDA FP32 张量；仅前向，不注册 autograd。
namespace gpu_prep {

at::Tensor vector_add_cuda(const at::Tensor& x, const at::Tensor& y);
at::Tensor bias_relu_cuda(const at::Tensor& x, const at::Tensor& bias);
at::Tensor row_sum_cuda(const at::Tensor& x);
at::Tensor row_max_cuda(const at::Tensor& x);
at::Tensor softmax_cuda(const at::Tensor& x);
at::Tensor transpose_naive_cuda(const at::Tensor& x);
at::Tensor transpose_tiled_cuda(const at::Tensor& x);
at::Tensor matmul_naive_cuda(const at::Tensor& a, const at::Tensor& b);
at::Tensor matmul_tiled_cuda(const at::Tensor& a, const at::Tensor& b);
at::Tensor layer_norm_cuda(const at::Tensor& x, const at::Tensor& weight,
                           const at::Tensor& bias, double eps);
at::Tensor rms_norm_cuda(const at::Tensor& x, const at::Tensor& weight,
                         double eps);
at::Tensor attention_cuda(const at::Tensor& q, const at::Tensor& k,
                          const at::Tensor& v, bool causal,
                          int64_t query_start);

}  // namespace gpu_prep
