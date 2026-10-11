#include "ops.h"

#include <c10/core/GradMode.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace gpu_prep {
namespace {

constexpr int kThreads = 256;       // reduction 要求为 2 的幂。
constexpr int kGridCap = 65535;     // 用 grid-stride loop 覆盖更大的输入。
constexpr int kTransposeTile = 32;
constexpr int kTransposeRows = 8;
constexpr int kMatmulTile = 16;
constexpr int64_t kMaxAttentionKeys = 4096;

// 使用除法和余数求 ceil，避免 (n + d - 1) 的整数溢出。
int64_t ceil_div(int64_t n, int64_t d) { return n / d + (n % d != 0); }

int blocks_for(int64_t work, int64_t work_per_block = 1) {
  return static_cast<int>(
      std::min<int64_t>(ceil_div(work, work_per_block), kGridCap));
}

int64_t checked_product(int64_t a, int64_t b, const char* what) {
  TORCH_CHECK(a >= 0 && b >= 0, what, ": negative dimension");
  TORCH_CHECK(a == 0 || b <= std::numeric_limits<int64_t>::max() / a,
              what, ": number of elements overflows int64");
  return a * b;
}

void check_tensor(const at::Tensor& x, const char* name) {
  TORCH_CHECK(x.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(x.scalar_type() == at::kFloat, name, " must be torch.float32");
  TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(!c10::GradMode::is_enabled() || !x.requires_grad(), name,
              ": these teaching kernels are forward-only; use torch.no_grad() "
              "for inference, or implement/register a backward formula");
}

void check_same_device(const at::Tensor& a, const at::Tensor& b,
                       const char* name) {
  TORCH_CHECK(a.device() == b.device(), name, " must be on ", a.device());
}

void check_matrix(const at::Tensor& x, const char* name) {
  check_tensor(x, name);
  TORCH_CHECK(x.dim() == 2, name, " must have shape [M, N]");
}

float check_eps(double eps) {
  const float e = static_cast<float>(eps);
  TORCH_CHECK(std::isfinite(eps) && eps > 0.0 && std::isfinite(e) && e > 0.0f,
              "eps must be finite, positive, and representable in float32");
  return e;
}

void check_norm_weight(const at::Tensor& x, const at::Tensor& weight) {
  check_tensor(x, "x");
  TORCH_CHECK(x.dim() >= 1, "x must have at least one dimension");
  TORCH_CHECK(x.size(-1) > 0, "normalization width must be positive");
  check_tensor(weight, "weight");
  check_same_device(x, weight, "weight");
  TORCH_CHECK(weight.dim() == 1 && weight.numel() == x.size(-1),
              "weight must have shape [x.size(-1)]");
}

// 所有线程都必须调用。本例用完整 shared-memory 树，便于观察同步。
// 最后一个 barrier 保证每个线程都读到了结果，scratch 才能被下次复用。
template <bool IsMax>
__device__ float block_reduce(float value, float* scratch) {
  const int tid = threadIdx.x;
  scratch[tid] = value;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      scratch[tid] = IsMax ? fmaxf(scratch[tid], scratch[tid + stride])
                           : scratch[tid] + scratch[tid + stride];
    }
    __syncthreads();
  }
  const float result = scratch[0];
  __syncthreads();
  return result;
}

__global__ void vector_add_kernel(const float* x, const float* y, float* out,
                                  int64_t n) {
  const int64_t first = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = first; i < n; i += step) {
    out[i] = x[i] + y[i];
  }
}

__global__ void bias_relu_kernel(const float* x, const float* bias, float* out,
                                 int64_t numel, int64_t width) {
  const int64_t first = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = first; i < numel; i += step) {
    const float z = x[i] + bias[i % width];
    // 正常有限值下等价于 ReLU；z 是 NaN 时本分支仍保留 NaN。
    out[i] = z < 0.0f ? 0.0f : z;
  }
}

template <bool IsMax>
__global__ void row_reduce_kernel(const float* x, float* out,
                                  int64_t rows, int64_t width) {
  __shared__ float scratch[kThreads];
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    float local = IsMax ? -CUDART_INF_F : 0.0f;
    // 相邻线程读相邻列；width 不需要是 2 的幂。
    for (int64_t col = threadIdx.x; col < width; col += blockDim.x) {
      const float value = x[row * width + col];
      local = IsMax ? fmaxf(local, value) : local + value;
    }
    const float reduced = block_reduce<IsMax>(local, scratch);
    if (threadIdx.x == 0) out[row] = reduced;
  }
}

__global__ void softmax_kernel(const float* x, float* out,
                               int64_t rows, int64_t width) {
  __shared__ float scratch[kThreads];
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const int64_t base = row * width;
    float local_max = -CUDART_INF_F;
    for (int64_t col = threadIdx.x; col < width; col += blockDim.x)
      local_max = fmaxf(local_max, x[base + col]);
    const float row_max = block_reduce<true>(local_max, scratch);

    float local_sum = 0.0f;
    for (int64_t col = threadIdx.x; col < width; col += blockDim.x)
      local_sum += expf(x[base + col] - row_max);
    const float sum = block_reduce<false>(local_sum, scratch);

    // 为了让实现适用于任意 width，不把整行放进 shared memory。
    // 代价是再次读取 x 和再次计算 exp；优化版可在寄存器/共享内存中复用。
    for (int64_t col = threadIdx.x; col < width; col += blockDim.x)
      out[base + col] = expf(x[base + col] - row_max) / sum;
  }
}

__global__ void transpose_naive_kernel(const float* x, float* out,
                                       int64_t rows, int64_t cols) {
  const int64_t n = rows * cols;
  const int64_t first = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = first; i < n; i += step) {
    const int64_t row = i / cols;
    const int64_t col = i % cols;
    out[col * rows + row] = x[i];
  }
}

__global__ void transpose_tiled_kernel(const float* x, float* out,
                                       int64_t rows, int64_t cols,
                                       int64_t tile_cols, int64_t tiles) {
  // 针对经典 FP32 32-bank 模型加一列 padding。
  __shared__ float tile[kTransposeTile][kTransposeTile + 1];
  for (int64_t tile_id = blockIdx.x; tile_id < tiles; tile_id += gridDim.x) {
    const int64_t tile_row = tile_id / tile_cols;
    const int64_t tile_col = tile_id % tile_cols;
    const int64_t in_col = tile_col * kTransposeTile + threadIdx.x;
    for (int j = 0; j < kTransposeTile; j += kTransposeRows) {
      const int r = threadIdx.y + j;
      const int64_t in_row = tile_row * kTransposeTile + r;
      tile[r][threadIdx.x] = (in_row < rows && in_col < cols)
                                ? x[in_row * cols + in_col] : 0.0f;
    }
    __syncthreads();  // 写 tile 的线程与读转置 tile 的线程不同。

    const int64_t out_col = tile_row * kTransposeTile + threadIdx.x;
    for (int j = 0; j < kTransposeTile; j += kTransposeRows) {
      const int r = threadIdx.y + j;
      const int64_t out_row = tile_col * kTransposeTile + r;
      if (out_row < cols && out_col < rows)
        out[out_row * rows + out_col] = tile[threadIdx.x][r];
    }
    __syncthreads();  // 防止下一 tile 提前覆盖仍在被读取的 shared memory。
  }
}

__global__ void matmul_naive_kernel(const float* a, const float* b, float* out,
                                    int64_t m, int64_t n, int64_t k) {
  const int64_t total = m * n;
  const int64_t first = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = first; i < total; i += step) {
    const int64_t row = i / n;
    const int64_t col = i % n;
    float acc = 0.0f;
    for (int64_t kk = 0; kk < k; ++kk)
      acc = fmaf(a[row * k + kk], b[kk * n + col], acc);
    out[i] = acc;
  }
}

__global__ void matmul_tiled_kernel(const float* a, const float* b, float* out,
                                    int64_t m, int64_t n, int64_t k,
                                    int64_t tile_cols, int64_t tiles) {
  __shared__ float as[kMatmulTile][kMatmulTile];
  __shared__ float bs[kMatmulTile][kMatmulTile];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int64_t stages = k / kMatmulTile + (k % kMatmulTile != 0);

  for (int64_t tile_id = blockIdx.x; tile_id < tiles; tile_id += gridDim.x) {
    const int64_t row = (tile_id / tile_cols) * kMatmulTile + ty;
    const int64_t col = (tile_id % tile_cols) * kMatmulTile + tx;
    float acc = 0.0f;
    for (int64_t stage = 0; stage < stages; ++stage) {
      const int64_t a_col = stage * kMatmulTile + tx;
      const int64_t b_row = stage * kMatmulTile + ty;
      // 尾块显式填零；不能让越界线程提前退出而跳过同步。
      as[ty][tx] = (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;
      bs[ty][tx] = (b_row < k && col < n) ? b[b_row * n + col] : 0.0f;
      __syncthreads();
      #pragma unroll
      for (int kk = 0; kk < kMatmulTile; ++kk)
        acc = fmaf(as[ty][kk], bs[kk][tx], acc);
      __syncthreads();  // 所有乘加完成后，才允许覆盖 as/bs。
    }
    if (row < m && col < n) out[row * n + col] = acc;
  }
}

template <bool IsRms>
__global__ void norm_kernel(const float* x, const float* weight,
                            const float* bias, float* out,
                            int64_t rows, int64_t width, float eps) {
  __shared__ float scratch[kThreads];
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const int64_t base = row * width;
    float mean = 0.0f;
    if constexpr (!IsRms) {
      float local_sum = 0.0f;
      for (int64_t col = threadIdx.x; col < width; col += blockDim.x)
        local_sum += x[base + col];
      mean = block_reduce<false>(local_sum, scratch) / static_cast<float>(width);
    }
    float local_sq = 0.0f;
    for (int64_t col = threadIdx.x; col < width; col += blockDim.x) {
      const float centered = x[base + col] - mean;
      local_sq = fmaf(centered, centered, local_sq);
    }
    const float variance = block_reduce<false>(local_sq, scratch)
                           / static_cast<float>(width);
    const float inv_std = rsqrtf(variance + eps);
    for (int64_t col = threadIdx.x; col < width; col += blockDim.x) {
      const float normalized = (x[base + col] - mean) * inv_std;
      const float affine = normalized * weight[col];
      if constexpr (IsRms) out[base + col] = affine;
      else out[base + col] = affine + bias[col];
    }
  }
}

// 教学版 dense attention：每个 CTA 负责一个 query，shared memory 保存 S 个分数。
// 这不是 FlashAttention；并未做 QK/PV 矩阵分块或 online softmax。
__global__ void attention_kernel(const float* q, const float* k, const float* v,
                                 float* out, int64_t query_rows,
                                 int64_t query_len, int64_t key_len,
                                 int64_t head_dim, bool causal,
                                 int64_t query_start) {
  extern __shared__ float scores[];
  __shared__ float scratch[kThreads];
  const float scale = rsqrtf(static_cast<float>(head_dim));
  for (int64_t qr = blockIdx.x; qr < query_rows; qr += gridDim.x) {
    const int64_t bh = qr / query_len;
    const int64_t qi = qr % query_len;
    const int64_t q_base = qr * head_dim;
    const int64_t kv_base = bh * key_len * head_dim;
    float local_max = -CUDART_INF_F;
    for (int64_t key = threadIdx.x; key < key_len; key += blockDim.x) {
      float score = -CUDART_INF_F;
      if (!causal || key <= query_start + qi) {
        float dot = 0.0f;
        for (int64_t d = 0; d < head_dim; ++d)
          dot = fmaf(q[q_base + d], k[kv_base + key * head_dim + d], dot);
        score = dot * scale;
      }
      scores[key] = score;
      local_max = fmaxf(local_max, score);
    }
    const float maximum = block_reduce<true>(local_max, scratch);

    float local_sum = 0.0f;
    for (int64_t key = threadIdx.x; key < key_len; key += blockDim.x) {
      const float value = expf(scores[key] - maximum);
      scores[key] = value;
      local_sum += value;
    }
    const float inv_sum = 1.0f / block_reduce<false>(local_sum, scratch);

    for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
      float acc = 0.0f;
      for (int64_t key = 0; key < key_len; ++key)
        acc = fmaf(scores[key] * inv_sum,
                   v[kv_base + key * head_dim + d], acc);
      out[q_base + d] = acc;
    }
    __syncthreads();  // 下一 query 不能覆盖其他线程仍在读取的 scores。
  }
}

at::Tensor transpose_impl(const at::Tensor& x, bool tiled) {
  check_matrix(x, "x");
  const c10::cuda::CUDAGuard guard(x.device());
  const int64_t m = x.size(0), n = x.size(1);
  auto out = at::empty({n, m}, x.options().requires_grad(false));
  if (x.numel() == 0) return out;
  const auto stream = c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
  if (tiled) {
    const int64_t tile_cols = ceil_div(n, kTransposeTile);
    const int64_t tiles = checked_product(ceil_div(m, kTransposeTile),
                                           tile_cols, "transpose tiles");
    transpose_tiled_kernel<<<blocks_for(tiles),
                             dim3(kTransposeTile, kTransposeRows), 0, stream>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), m, n, tile_cols, tiles);
  } else {
    transpose_naive_kernel<<<blocks_for(x.numel(), kThreads), kThreads, 0, stream>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), m, n);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor matmul_impl(const at::Tensor& a, const at::Tensor& b, bool tiled) {
  check_matrix(a, "a");
  check_matrix(b, "b");
  check_same_device(a, b, "b");
  TORCH_CHECK(a.size(1) == b.size(0), "matmul inner dimensions must match");
  const c10::cuda::CUDAGuard guard(a.device());
  const int64_t m = a.size(0), k = a.size(1), n = b.size(1);
  const int64_t total = checked_product(m, n, "matmul output");
  // 空的求和有单位元 0；不能把未初始化的 empty 当作 K=0 的答案。
  if (k == 0) return at::zeros({m, n}, a.options().requires_grad(false));
  auto out = at::empty({m, n}, a.options().requires_grad(false));
  if (total == 0) return out;
  const auto stream = c10::cuda::getCurrentCUDAStream(a.get_device()).stream();
  if (tiled) {
    const int64_t tile_cols = ceil_div(n, kMatmulTile);
    const int64_t tiles = checked_product(ceil_div(m, kMatmulTile),
                                           tile_cols, "matmul tiles");
    matmul_tiled_kernel<<<blocks_for(tiles), dim3(kMatmulTile, kMatmulTile),
                          0, stream>>>(a.data_ptr<float>(), b.data_ptr<float>(),
                                        out.data_ptr<float>(), m, n, k,
                                        tile_cols, tiles);
  } else {
    matmul_naive_kernel<<<blocks_for(total, kThreads), kThreads, 0, stream>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), m, n, k);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

template <bool IsMax>
at::Tensor row_reduce_impl(const at::Tensor& x) {
  check_matrix(x, "x");
  TORCH_CHECK(x.size(1) > 0, "reduction width N must be positive");
  const c10::cuda::CUDAGuard guard(x.device());
  auto out = at::empty({x.size(0)}, x.options().requires_grad(false));
  if (x.size(0) == 0) return out;
  const auto stream = c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
  row_reduce_kernel<IsMax><<<blocks_for(x.size(0)), kThreads, 0, stream>>>(
      x.data_ptr<float>(), out.data_ptr<float>(), x.size(0), x.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace

at::Tensor vector_add_cuda(const at::Tensor& x, const at::Tensor& y) {
  check_tensor(x, "x");
  check_tensor(y, "y");
  check_same_device(x, y, "y");
  TORCH_CHECK(x.sizes() == y.sizes(), "x and y must have identical shapes");
  const c10::cuda::CUDAGuard guard(x.device());
  auto out = at::empty(x.sizes(), x.options().requires_grad(false));
  if (x.numel() == 0) return out;
  const auto stream = c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
  vector_add_kernel<<<blocks_for(x.numel(), kThreads), kThreads, 0, stream>>>(
      x.data_ptr<float>(), y.data_ptr<float>(), out.data_ptr<float>(), x.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor bias_relu_cuda(const at::Tensor& x, const at::Tensor& bias) {
  check_matrix(x, "x");
  check_tensor(bias, "bias");
  check_same_device(x, bias, "bias");
  TORCH_CHECK(bias.dim() == 1 && bias.numel() == x.size(1),
              "bias must have shape [N] for x[M, N]");
  const c10::cuda::CUDAGuard guard(x.device());
  auto out = at::empty(x.sizes(), x.options().requires_grad(false));
  if (x.numel() == 0) return out;  // 也避免在 width=0 时取模。
  const auto stream = c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
  bias_relu_kernel<<<blocks_for(x.numel(), kThreads), kThreads, 0, stream>>>(
      x.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(),
      x.numel(), x.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor row_sum_cuda(const at::Tensor& x) { return row_reduce_impl<false>(x); }
at::Tensor row_max_cuda(const at::Tensor& x) { return row_reduce_impl<true>(x); }

at::Tensor softmax_cuda(const at::Tensor& x) {
  check_matrix(x, "x");
  TORCH_CHECK(x.size(1) > 0, "softmax width N must be positive");
  const c10::cuda::CUDAGuard guard(x.device());
  auto out = at::empty(x.sizes(), x.options().requires_grad(false));
  if (x.size(0) == 0) return out;
  const auto stream = c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
  softmax_kernel<<<blocks_for(x.size(0)), kThreads, 0, stream>>>(
      x.data_ptr<float>(), out.data_ptr<float>(), x.size(0), x.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor transpose_naive_cuda(const at::Tensor& x) { return transpose_impl(x, false); }
at::Tensor transpose_tiled_cuda(const at::Tensor& x) { return transpose_impl(x, true); }
at::Tensor matmul_naive_cuda(const at::Tensor& a, const at::Tensor& b) {
  return matmul_impl(a, b, false);
}
at::Tensor matmul_tiled_cuda(const at::Tensor& a, const at::Tensor& b) {
  return matmul_impl(a, b, true);
}

at::Tensor layer_norm_cuda(const at::Tensor& x, const at::Tensor& weight,
                           const at::Tensor& bias, double eps) {
  check_norm_weight(x, weight);
  check_tensor(bias, "bias");
  check_same_device(x, bias, "bias");
  TORCH_CHECK(bias.dim() == 1 && bias.numel() == x.size(-1),
              "bias must have shape [x.size(-1)]");
  const float e = check_eps(eps);
  const c10::cuda::CUDAGuard guard(x.device());
  auto out = at::empty(x.sizes(), x.options().requires_grad(false));
  if (x.numel() == 0) return out;
  const int64_t width = x.size(-1), rows = x.numel() / width;
  const auto stream = c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
  norm_kernel<false><<<blocks_for(rows), kThreads, 0, stream>>>(
      x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
      out.data_ptr<float>(), rows, width, e);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor rms_norm_cuda(const at::Tensor& x, const at::Tensor& weight, double eps) {
  check_norm_weight(x, weight);
  const float e = check_eps(eps);
  const c10::cuda::CUDAGuard guard(x.device());
  auto out = at::empty(x.sizes(), x.options().requires_grad(false));
  if (x.numel() == 0) return out;
  const int64_t width = x.size(-1), rows = x.numel() / width;
  const auto stream = c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
  norm_kernel<true><<<blocks_for(rows), kThreads, 0, stream>>>(
      x.data_ptr<float>(), weight.data_ptr<float>(), nullptr, out.data_ptr<float>(),
      rows, width, e);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor attention_cuda(const at::Tensor& q, const at::Tensor& k,
                          const at::Tensor& v, bool causal, int64_t query_start) {
  check_tensor(q, "q");
  check_tensor(k, "k");
  check_tensor(v, "v");
  check_same_device(q, k, "k");
  check_same_device(q, v, "v");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
              "attention expects q[B,H,L,D], k/v[B,H,S,D]");
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(1) == k.size(1)
                  && q.size(3) == k.size(3) && k.sizes() == v.sizes(),
              "q/k/v batch, head count and head dimension must agree; k/v shapes match");
  TORCH_CHECK(q.size(3) > 0, "head dimension D must be positive");
  TORCH_CHECK(k.size(2) > 0 && k.size(2) <= kMaxAttentionKeys,
              "teaching attention requires 1 <= key length S <= 4096");
  TORCH_CHECK(query_start >= 0, "query_start must be nonnegative");
  TORCH_CHECK(q.size(2) == 0 ||
                  query_start <= std::numeric_limits<int64_t>::max() - (q.size(2) - 1),
              "query_start + query index overflows int64");
  const c10::cuda::CUDAGuard guard(q.device());
  auto out = at::empty(q.sizes(), q.options().requires_grad(false));
  if (q.numel() == 0) return out;
  const int64_t query_rows = q.numel() / q.size(3);
  const auto stream = c10::cuda::getCurrentCUDAStream(q.get_device()).stream();
  const size_t shared_bytes = static_cast<size_t>(k.size(2)) * sizeof(float);
  attention_kernel<<<blocks_for(query_rows), kThreads, shared_bytes, stream>>>(
      q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
      out.data_ptr<float>(), query_rows, q.size(2), k.size(2), q.size(3),
      causal, query_start);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace gpu_prep
