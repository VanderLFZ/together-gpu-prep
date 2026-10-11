# CUDA C++：从能写对，到能解释性能

本章对应练习 1、2、3、4、5、8、9。代码是本教材编写的教学实现；目标是让线程映射、边界、同步和数据流可以逐行解释。它没有承诺比 PyTorch、cuBLAS 或成熟 attention 库更快。

## 共同约定与运行方式

完整源码位于 `gpu_prep/csrc/ops.h`、`gpu_prep/csrc/bindings.cpp`、`gpu_prep/csrc/kernels.cu`；Python 入口为 `gpu_prep/cuda_ops.py`。`import gpu_prep.cuda_ops` 不编译，第一次实际调用或显式 `load_extension()` 才进行 JIT 构建。

```python
import torch
from gpu_prep import cuda_ops

cuda_ops.load_extension()  # 先构建，别把构建时间算进性能数字。
x = torch.randn(257, device="cuda", dtype=torch.float32)
y = torch.randn_like(x)
torch.testing.assert_close(cuda_ops.vector_add(x, y), x + y)
```

所有 CUDA 入口要求输入在同一 NVIDIA GPU 上、`torch.float32`、连续存储。CUDA 部分只有 forward；启用梯度记录且输入需要梯度时明确报错。推理时可以在 `torch.no_grad()` 中调用。这个限制避免让一个缺少 backward 的实现产生看似可训练的错误结果；它不是说 PyTorch 自定义 CUDA 算子不能支持训练。

输入数值的主要约定是有限 FP32 值，并要求计算中间量不发生溢出。host 不扫描张量内容，否则会增加 GPU 操作和同步。`row_max` 使用 `fmaxf`，不承诺与 PyTorch 的 NaN 传播一致；softmax 和 attention 也没有为输入 NaN、正无穷或整行负无穷定义额外协议。融合 `bias_relu` 特别保留了加法结果的 NaN，便于与对应 PyTorch 表达式对照。

| API | 输入 | 输出 | 空维与其他限制 |
|---|---|---|---|
| `vector_add(x, y)` | 相同 shape | 相同 shape | 允许零元素，也允许标量 |
| `bias_relu(x, bias)` | `[M,N]`、`[N]` | `[M,N]` | M/N 可为 0 |
| `row_sum` / `row_max` | `[M,N]` | `[M]` | N > 0；M 可为 0 |
| `softmax` | `[M,N]` | `[M,N]` | N > 0；M 可为 0 |
| 两种 `transpose` | `[M,N]` | 连续的 `[N,M]` | M/N 可为 0 |
| 两种 `matmul` | `[M,K]`、`[K,N]` | `[M,N]` | K=0 时返回零；M/N=0 时为空 |
| `layer_norm` | `[...,D]`、weight/bias `[D]` | 与 x 同形状 | D > 0；前导维可为 0；eps 正且能以 FP32 表示 |
| `rms_norm` | `[...,D]`、weight `[D]` | 与 x 同形状 | 同上 |
| `attention` | q `[B,H,L,D]`，k/v `[B,H,S,D]` | `[B,H,L,D]` | 1 ≤ S ≤ 4096；D > 0；L/B/H 可为 0；query_start ≥ 0 |

### Host 代码的六个关键点

1. **先检查 shape、dtype、device、contiguous。** 内核按连续 FP32 数据解释指针，不能把调用者传错的 layout 当成正确输入。
2. **用 `CUDAGuard` 切换到输入的设备。** 用户当前设备可能不是 x 所在设备；函数离开后 guard 恢复原来的设备状态。
3. **在 PyTorch 当前 stream 上 launch。** 输入的生成与后续使用都可能发生在非默认 stream；擅自使用默认 stream 会破坏依赖顺序。
4. **先处理空张量，避免零 block launch。** GEMM 的 K=0 是空求和，因此返回 0；它不同于未初始化的 `empty`。
5. **指针偏移与维度用 `int64_t`。** `blockIdx.x * blockDim.x` 在相乘前就转换为 64 位，避免先发生 32 位溢出再转换。
6. **launch 后检查 CUDA 错误。** `C10_CUDA_KERNEL_LAUNCH_CHECK()` 不等于等待 kernel 执行结束；异步执行中的越界错误可能在后续同步才暴露。常规算子内不加入 `cudaDeviceSynchronize()`，否则会改变上层的并发与计时行为。

这些 API 的设备与 stream 语义可参考 [PyTorch CUDA Support](https://docs.pytorch.org/cppdocs/api/cuda/index.html)；JIT extension 的构建入口参考 [PyTorch C++ Extensions](https://docs.pytorch.org/docs/stable/cpp_extension.html)。

### 怎样证明“没有漏算，也没有重复写”

元素级 kernel 使用 `first = blockIdx.x * blockDim.x + threadIdx.x` 和 `step = gridDim.x * blockDim.x`。每个线程处理 `first, first+step, first+2*step, ...`，直到超出元素个数。

对于任意合法下标 i，整数除法唯一地确定 `i % step` 对应的初始线程和 `i // step` 对应的循环次数。因此每个合法元素恰好被一个线程写一次。grid 的 block 数可以封顶，不会因此遗漏更大的输入。行级和 tile 级 kernel 使用相同原理在 rows 或 tiles 上循环。

<!-- exercise:1:start -->
## 练习 1 的 CUDA 实现：Vector Add 与 Bias + ReLU

### 题目与代码入口

先写 `z[i] = x[i] + y[i]`，再扩展成 `out[m,n] = max(x[m,n] + bias[n], 0)`。对应 `vector_add_kernel`、`bias_relu_kernel` 及其 host wrapper。

用下面的例子检查自己是否真的理解索引：N=257，block=256。初次 launch 有两个 block，第二个 block 的第 0 个线程处理 i=256，其余线程因为 `i < N` 为假而不写内存。这些线程可以直接结束，因为该 kernel 没有任何 block 内同步。

Bias 版本先把 `[M,N]` 展平为 `numel=M*N` 个元素。展平下标 i 对应列 `i % N`，因此使用 `bias[i % N]`；行号不需要显式算出。N=0 时 host 已提前返回，kernel 内不会发生除以零或模零。

### 数据流、复杂度与正确性

每个线程只读自己的输入位置、只写自己的输出位置，不与其他线程共享中间结果，因此没有原子操作或 `__syncthreads()`。相邻线程对 x/y/out 访问相邻 FP32 元素，适合合并访存。元素总工作量为 O(N)，每线程额外存储 O(1)。Bias + ReLU 为 O(MN)。

向量加法的理想有效数据量是 `12N Byte`：两次 FP32 读取和一次 FP32 写入。每元素只有 1 次浮点加法，算术强度约为 `1/12 FLOP/Byte`。这个字节数是算法层面的有效流量模型，缓存、写策略和事务粒度会影响实际 HBM 流量。

Bias + ReLU 若拆成两个 kernel，需要把中间 `x+bias` 写回并再次读取。融合版省去这份中间张量的读写，并少一次 launch。bias 能在缓存中被重复使用，但不能把“所有 bias 访问都命中缓存”当成无需验证的事实。

### 可以直接复述的追问答案

**问：为什么不需要同步？**

答：每个输出只依赖同下标的输入，线程之间没有生产者与消费者关系，也没有共享写入。加 barrier 不会补充正确性，只会增加等待。是否需要同步取决于数据依赖，不能因为是 CUDA 就习惯性加同步。

**问：block size 取 256 是最优吗？**

答：它是方便理解和比较的起点，不是对所有 GPU、长度和融合表达式的最优结论。需要比较若干 block size，并观察寄存器、可同时驻留的 blocks、launch 开销和有效带宽。简单大向量通常很容易获得足够并行度；小向量可能主要受固定启动成本影响。

**问：怎样继续优化？**

答：先排除首次构建，并明确测量范围。若测纯 kernel，要预分配输出并说明这一点；若测本教材的 wrapper，则保留它内部必要的输出分配，让 baseline 使用相同口径，并区分 GPU event 时间与 host wall 时间。随后检查是否接近合理的带宽水平，再考虑减少 launch、与相邻算子融合、一次处理多个元素、在满足对齐和尾部处理条件下做向量化 load/store。`float4` 不是无条件加速开关：起始地址、长度、stride 都要满足要求，尾部仍需正确处理。

**问：如果 x 是转置出来的非连续 tensor 呢？**

答：这份接口明确拒绝。扩展方法有两种：上层显式 `.contiguous()`，或者把 stride 作为参数，根据坐标计算真实地址。前者代码简单但可能发生额外拷贝；后者支持面广，但地址计算和访存模式更复杂。性能比较时必须把所选方案的必要转换成本纳入目标范围。

**问：FP16/BF16 版本怎么写？**

答：把 load/store 类型模板化；加载后按所需数值约定转换到 float 运算，再转换回输出类型。带宽需求会下降，但转换、对齐和实际指令路径需要测量。纯加法也可以研究 half2 等打包运算，不过这与保持 FP32 中间结果是不同的精度和实现选择，必须明确比较标准。

**问：原地写 out=x 可以吗？**

答：对于同形状逐元素加法，某些别名关系在数学上可以安全实现；但要逐一分析输入与输出是否有重叠、broadcast 是否导致一个位置被多处读取。本教材始终新分配输出，便于建立无别名的基础契约。若增加 `out=`，要同时设计别名检查与 autograd 语义。

### 练习顺序

从空文件写基础版本，测试 N=0、1、31、32、33、255、256、257、1000003；再写 bias 广播；最后解释融合省下的具体读写。代码与讲解都完成后，再比较性能。
<!-- exercise:1:end -->

<!-- exercise:2:start -->
## 练习 2 的 CUDA 实现：Row Sum 与 Row Max

### 线程映射，逐步拆开

一个 CTA/block 负责一行，固定 256 个线程。线程 t 读取列 `t, t+256, t+512, ...`，在寄存器里维护一个局部结果。所有列被这些互不相交的序列覆盖。

以 N=513 为例：线程 0 处理 0、256、512；线程 1 处理 1、257；线程 255 处理 255、511。这里完全不要求 N 是 2 的幂。要求为 2 的幂的是本例进行 shared-memory 树形归约的线程数。

局部计算结束后，每个线程把值放入 `scratch[tid]`。随后依次做 stride=128、64、32、16、8、4、2、1 的合并。第一步线程 0 合并 0 和 128，第二步合并先前结果与 64 的结果，最终 `scratch[0]` 包含整行结果。

Sum 的无效线程初值是 0；max 的无效线程初值是负无穷。把 max 初值设成 0 会使全负数输入出错。

### 同步为什么正好在这些位置

1. 所有线程写 scratch 后同步，防止读取尚未被写入的局部值。
2. 每一层合并后同步，防止下一层过早读取上一层的结果。
3. 把 `scratch[0]` 复制到每个线程的局部变量后，再同步一次，才允许下一次 reduction 覆盖 scratch。

第三点很容易忽略：这个 helper 在 softmax、norm 中会被连续调用，同一个 block 也可能继续处理下一行。如果某个线程已经覆盖 scratch，而其他线程还没读走旧结果，就存在 race。NVIDIA 对 block 协作和 barrier 的基本语义有说明：[Writing SIMT Kernels](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html)。

### 复杂度与带宽模型

总输入工作为 O(MN)，输出 O(M)。固定线程数 B=256 时，一行的计算路径大致为 O(ceil(N/B)+log B)，不能把这个路径长度直接等同于整块 GPU 的总运行时间。额外 shared memory 是每 CTA `256×4=1024 Byte`。理想有效流量约为 `4MN+4M Byte`。

### 可以直接复述的追问答案

**问：为什么 N 不是 2 的幂也正确？**

答：原始 N 个元素先被分给固定 256 个线程，每个线程产生一个局部值。树形归约处理的是这 256 个局部值。没有元素的线程使用 sum/max 的单位元，因此不影响结果。N 是否为 2 的幂与后面这棵固定大小的归约树无关。

**问：为什么不让所有线程对一个结果做 atomicAdd？**

答：大量线程更新同一个地址会产生争用。先在寄存器和 block 内合并，能够显著减少需要全局合并的次数。单行只用一个 block 时根本不需要全局原子操作；如果宽度太大要用多个 blocks，就需要第二阶段 reduction，或者对每个 block 的局部结果做原子更新。选择需要依据行数、行长、开销和确定性要求。

**问：为什么与 torch.sum 的结果不逐 bit 一样？**

答：浮点加法通常不满足结合律。不同 reduction tree 改变了求和顺序，FMA 或累加精度也会改变舍入。先用 FP64 reference 检查误差，并给出合理 atol/rtol；不能把所有差异当作 bug，也不能用很宽的容差掩盖索引错误。max 对普通有限值是取同一组元素的最大值，通常可以做精确比较。

**问：warp shuffle 版本怎样优化这棵树？**

答：每个 warp 先在寄存器之间用 shuffle 合并；每个 warp 的 lane 0 将一个结果写入 shared memory；第一个 warp 再合并这些 warp 结果。256 个线程只有 8 个 warp，shared-memory 通信量与 block barrier 数都会减少。必须确保 `_sync` mask 所指的 lanes 确实参与相同的 shuffle；尾部元素应该用单位元填充，而不是让被 mask 声明参与的线程提前退出。

对固定 256 线程、一维 block，可把 sum helper 改写为下面的进阶版本。它返回每个线程都可读取的结果，并保留 shared-memory 复用前的同步；不要未经修改拿来支持任意 block 大小。它是独立的替换示例，默认 `kernels.cu` 仍使用上面的 shared-memory 树；替换后需要重新编译并运行 correctness tests。

```cpp
__device__ float warp_sum_full(float value) {
    for (int offset = 16; offset > 0; offset /= 2)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return value;
}

__device__ float block_sum_256(float value) {
    __shared__ float warp_totals[8];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    value = warp_sum_full(value);  // 每个 warp 的所有 32 lanes 都参与。
    if (lane == 0) warp_totals[warp] = value;
    __syncthreads();

    if (warp == 0) {
        float total = lane < 8 ? warp_totals[lane] : 0.0f;
        total = warp_sum_full(total);
        if (lane == 0) warp_totals[0] = total;
    }
    __syncthreads();
    const float result = warp_totals[0];
    __syncthreads();
    return result;
}
```

**问：什么时候 one-block-per-row 不合适？**

答：当行数很少而单行特别长时，总 block 数不足以使用整块 GPU，可以把一行分成多个 chunks，再做第二阶段归约。相反，当每行只有几个元素而行数很多时，256 线程处理一行会浪费，可以让一个 warp 甚至更小线程组处理一行，一个 block 处理多行。两种情况需要不同映射，不存在只调大 block size 就解决所有尺寸的规则。

**问：怎么扩展低精度？**

答：输入可以是 FP16/BF16，但建议局部和及 block sum 用 FP32，输出最后再转换。累加精度和输出 dtype 是两个独立选择。若要求非常高的求和精度，还可以研究 FP64 或补偿求和，但会改变吞吐和寄存器成本。
<!-- exercise:2:end -->

<!-- exercise:3:start -->
## 练习 3 的 CUDA 实现：数值稳定的行 Softmax

### 三步计算

对一行 x，计算最大值 m、分母 l，再归一化：

\[
m=\max_i x_i,\qquad l=\sum_i e^{x_i-m},\qquad p_i=e^{x_i-m}/l.
\]

`softmax_kernel` 用一个 block 负责一行，依次调用 block max、block sum，然后每线程写自己的输出位置。与 reduction 相同，线程按 256 的步长读取列。

本实现不把整行保存在 shared memory 中，因此 N 可以超过 4096。它为清晰和通用宽度做了一个性能取舍：x 读取三遍，指数计算两遍。源码中的三段循环分别对应 max、exp sum、最终输出。

### 减最大值为什么不会改变答案

对分子分母同时乘以 `exp(-m)`，比值不变。对于有限输入，所有 `x_i-m ≤ 0`，指数不会因为巨大正数而溢出，且至少有一个最大元素对应 `exp(0)=1`，分母因此大于等于 1。很小的项仍可能下溢到 0，这不同于直接 exp 大正数导致无穷再相除产生 NaN。

### 复杂度与教学版成本

总工作 O(MN)，每 CTA shared memory 1024 Byte。按源码的显式全局访问计数，约有 3 次输入读、1 次输出写，即 `16MN Byte`，但缓存会改变最终 HBM 流量。单次输入读加输出写的理想最低模型则为 `8MN Byte`。这两个模型回答的是不同问题，不能把后者直接作为这份实现的真实 DRAM 字节数。

### 可以直接复述的追问答案

**问：为什么叫稳定，而不是完全无数值误差？**

答：减去最大值主要解决 exp 的正向溢出。有限精度减法、指数近似、求和顺序和输出舍入仍会造成误差。稳定不等于精确实数运算，更不代表自动处理所有 NaN、无穷或全 masked 行。

**问：怎样减少三次读取和两次 exp？**

答：若行宽和资源允许，可以将加载的行片段保存在寄存器中，或者把 exponentials 存到 shared memory，在算出分母之后复用。寄存器过多会限制驻留甚至 spill；shared memory 也会限制并行度。短行、普通行、超长行应当分别选择实现，不能单凭“少读一次”就断言端到端更快。

**问：online softmax 怎么更新？**

答：维护当前最大值 m 和相对 m 缩放的指数和 l。读到新块，其最大值为 b，令 `m_new=max(m,b)`，更新

\[
l_{new}=e^{m-m_{new}}l + \sum_{x\in\text{新块}}e^{x-m_{new}}.
\]

旧的 l 必须随最大值改变重新缩放。只更新 max 而不调整旧分母是错误的。对于只输出 softmax 本身的算子，得到全局统计量后通常还需要再访问输入以产生所有输出；online 统计并不自动让所有输出一次读完。

**问：mask 怎么加？**

答：将不可见位置的分数视为负无穷，使其 exp 变为零。必须额外定义整行都不可见时的行为：此时 `max=-inf`，直接 `-inf-(-inf)` 会出现 NaN，可以明确返回全零或按上层契约报错。本 CUDA softmax 入口只接受普通有限输入，没有任意 mask 参数；不要声称它已经覆盖该扩展。

**问：它一定 memory-bound 吗？**

答：需要结合 shape 和实现判断。softmax 的指数、归约和同步也有成本；短输入可能 launch-bound，小行数可能并行度不足。先根据模型估计流量和计算，再用实际计时及 profiler 检查瓶颈，不能仅凭算子名字判断。

**问：backward 是什么？**

答：设前向输出为 p，上游梯度为 g，则每行 `dx = p * (g - sum(g*p))`。因此 backward 也包含行 reduction，可以复用类似的映射。是否保存 p、重新计算 p、融合后续操作，取决于显存和算力的权衡。本 CUDA 教学接口未实现 backward。

**问：FP16/BF16 要注意什么？**

答：通常先转换到 FP32 做 max、exp、sum 和归一化，再转换回输出 dtype。低精度输入已经发生的量化误差不能靠 FP32 累加恢复，但能避免进一步用低精度统计造成额外损失。
<!-- exercise:3:end -->

<!-- exercise:4:start -->
## 练习 4 的 CUDA 实现：朴素转置与 Tiled 转置

### 先区分两种“转置”

本题实际搬运数据，输出是连续存储的 `[N,M]`。PyTorch 的 `x.t()` 经常只是一个共享底层存储的 view，因此公平的正确性和拷贝基准应考虑 `x.t().contiguous()`；不能把只改 metadata 的调用时间与真正的数据搬运混在一起。

朴素版本对输入连续下标 i 计算 `row=i/N`、`col=i%N`，再写 `out[col*M+row]`。相邻线程读相邻 x，但写到 out 时相隔 M 个 float，导致写入不利于合并。

### Shared tile 如何同时改善读与写

Tiled 版本一个 block 是 `dim3(32,8)`，共有 256 线程，处理 32×32 个元素。每线程沿 y 方向循环 4 次，合计覆盖 1024 个元素。

读取阶段：同一个 warp 的 x 从 0 到 31，读取输入同一行的 32 个连续元素，存到 `tile[local_row][local_col]`。

写入阶段：交换 tile 的行列角色，让同一个 warp 写输出同一行的连续元素；值从 `tile[local_col][local_row]` 取出。shared memory 在这一步承担了线程间重排数据的作用。

这不是只把两个下标交换就结束。全局输出 tile 的坐标也必须交换：输入 tile `(tile_row,tile_col)` 对应输出 tile `(tile_col,tile_row)`。尾部 tile 的读写都带边界检查，因此 M/N 不需要被 32 整除。

### 为什么用 `tile[32][33]`

在经典 32-bank、FP32 每元素 4 Byte 的分析模型下，`tile[r][c]` 的 bank 可用线性元素下标对 32 取模来理解。若每行长度 32，同一列的 32 个元素都落到同一个 bank；行长变成 33 后，同列不同 r 的 bank 会分散。额外的一列用于改变地址步长，不是输出数据的一部分。

这个具体转置示例和 bank 解释可对照 NVIDIA 的 [An Efficient Matrix Transpose in CUDA C/C++](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)。不要把这个 FP32 示例的 padding 结论不加分析地套用到任意 dtype、向量访问或 Tensor Core layout。

### 同步和正确性

第一次 barrier 位于 tile 全部写完之后，因为读取转置 tile 的线程可能不是写入该元素的线程。第二次 barrier 在输出阶段结束之后，防止 grid-stride tile 循环下一轮提前覆盖仍在读取的 tile。边界线程仍要经过这两个 barrier，不能看到越界就提前 return。

总工作 O(MN)，每个有效元素读一次写一次，理想有效流量 `8MN Byte`；每 CTA 的 shared memory 为 `32×33×4=4224 Byte`。

### 可以直接复述的追问答案

**问：shared memory 不是增加了一次中间读写吗，为什么可能更快？**

答：它增加的是片上重排，换来全局读写更好的合并。如果朴素转置每个 warp 触发很多分散的全局事务，减少这些昂贵事务的收益可能超过片上操作与同步成本。小矩阵时收益未必明显，所以要按规模测量。

**问：padding 是否增加 global memory 流量？**

答：本例 padding 只存在于 shared tile，没有对应的额外输入或输出列。它消耗一些额外 shared memory，但不会要求从输入读取不存在的第 33 列。

**问：M=1003、N=769 时怎么保证不越界？**

答：tile 数使用向上取整。读时检查原输入 row<M、col<N，越界槽填零；写时检查转置后的 out_row<N、out_col<M。只有合法输出才写，而且它索引到的 tile 槽对应一个合法输入。所有线程仍按相同顺序同步。

**问：为什么不用 32×32=1024 个线程？**

答：一个 block 的线程数和它处理的数据元素数不必相同。256 个线程通过每线程 4 次加载完成同一个 tile。线程数、每线程工作、寄存器和调度开销需要一起考虑；最大线程数并不自动带来最大带宽。

**问：in-place transpose 呢？**

答：它是另一个问题。方阵需要协调对角 tile 内转置与两个非对角 tile 的交换，避免覆盖尚未读取的数据；非方阵还涉及不同的置换循环。当前实现新分配输出，不包含这些同步与别名条件。

**问：能否彻底消除转置？**

答：如果下游算子能够直接读取该 stride，或通过 layout-aware kernel 在加载时完成需要的重排，就可能避免单独的物理转置。但也可能让下游访存更差。要比较整个消费链，而不只是这一个转置函数。
<!-- exercise:4:end -->

<!-- exercise:5:start -->
## 练习 5 的 CUDA 实现：FP32 GEMM 基线与 Shared-memory Tiling

### 朴素实现到底怎样算一个元素

对于 `C[M,N] = A[M,K] @ B[K,N]`，令一个线程负责 C[row,col]。该线程循环 kk，从 A 的第 row 行和 B 的第 col 列取数，累加 `A[row,kk]*B[kk,col]`。源码使用 `fmaf`，表示乘加合并执行并按相应规则舍入。

线程的行列由展平输出 i 得到：`row=i/N`、`col=i%N`。相邻线程通常负责相邻输出列，因此它们在同一 kk 上读取 B 相邻列；而 A[row,kk] 在多个线程之间存在明显复用机会。朴素代码没有主动将这些跨线程复用保存在 shared memory。

### 16×16 tile：分工与完整计算路径

一个 block 是 `dim3(16,16)`，每线程仍负责一个 C 元素。它共同加载一个 A 的 16×16 tile 和一个 B 的 16×16 tile。加载后同步，每个线程在 shared memory 上做 16 次乘加；再次同步，再加载 K 方向的下一组 tile。

以 M=127、N=65、K=93 为例：输出需要 8×5 个 tile，K 方向需要 6 个阶段。最后一个 K 阶段只有 13 个有效位置，其余 shared 槽填零。输出边界之外的线程不写 C，但仍参与 tile load、zero-fill 和所有 barrier。

每一个 C[row,col] 的累加器跨越所有 K stages 保存在寄存器中，最后只写一次。每个 stage 中 A 的一个元素被多个输出列复用，B 的一个元素被多个输出行复用。

### 为什么 K 循环有两个 barrier

第一次确保本阶段的 A/B tile 已被所有线程加载完成，才能开始消费。第二次确保本阶段 shared tile 已被所有线程读完，才能让快线程开始覆盖下一阶段的数据。第二次并非与第一次重复，它保护的是相反方向的依赖。

### FLOPs、数据量与 tile 大小

数学运算量约为 `2MNK FLOPs`，输出 O(MN)。忽略尾块、缓存和输出写回，对边长 T 的一个 K-stage：输入加载 `2T²` 个 float，执行约 `2T³` FLOPs，因此这一层 global-to-shared 复用的算术强度约为 `T/4 FLOP/Byte`。

这个结果并不是整个 GEMM 的理想最低流量模型。更高层的 cache 还可能在不同 CTA 之间复用数据；输出写回、尾块、事务粒度也需要考虑。全部输入各读一次、输出写一次的理想模型为

\[
I_{ideal}=\frac{2MNK}{4(MK+KN+MN)}.
\]

两种算式对应不同层次的假设，面试时应先说清层次，再用数字。

### 可以直接复述的追问答案

**问：tiled 版本一定比 naive 快吗？**

答：它改善了明确的数据复用，但增加了 shared-memory 操作和 barrier。非常小的矩阵可能没有足够工作来摊薄额外开销；形状不匹配会增加尾块浪费。两者都可能受 launch、并行度和缓存影响，必须实测。本教材提供二者是为了比较优化机制，不是预先宣称所有尺寸上的速度关系。

**问：tile 越大越好吗？**

答：更大 tile 增强局部复用，也可能增加每 CTA 的 shared memory 和寄存器占用，减少同时驻留的 blocks，增加尾部无效计算。若一线程一个输出，还会碰到最大线程数限制。应根据 GPU 资源、M/N/K 和实际 profiler 指标选择，不能只看算术强度。

**问：register tiling 怎么继续提升？**

答：让每线程维护一个小的输出子块，而不是一个标量。例如对同一个从 shared memory 读出的 A 元素，与多个 B 元素相乘，更新多个寄存器累加器。这样在 shared-to-register 层也形成复用，减少每 FLOP 的 shared-memory 读取。代价是更多寄存器、更多索引和更复杂的线程布局，需检查是否 spill。

**问：Tensor Core 版本与当前版本的区别？**

答：当前是普通 FP32 SIMT FMA，没有 WMMA/MMA/Tensor Core 指令路径。Tensor Core 路径通常让一个 warp 或协作线程组以特定 fragment/layout 执行矩阵块运算，需要满足数据类型、形状、对齐与布局等要求。不是把 FP32 指针改成 half 指针就完成了 Tensor Core GEMM。要分别验证允许的输入精度、累加精度和最终误差。

**问：double buffering / async copy 有什么作用？**

答：当当前 tile 在计算时，提前搬运下一 tile，把数据传输与计算重叠。双缓冲需要两套缓冲或其他正确的流水线存储管理，并有阶段间的到达与等待规则。只有搬运和计算足够大、硬件及资源允许重叠时才可能获益；同步错了会读取未完成的数据或提前覆盖缓冲。

**问：为什么 decode 时的 skinny GEMM 可能不适合大方块 tile？**

答：小 M 意味着输出行很少，大 tile 会产生许多无效行，CTA 数也可能不足。权重读流量不一定被足够多的 token 复用。可以考虑适合 skinny shape 的 tile、分 K 的并行化或针对该形状的 GEMV/GEMM 路径。Split-K 会增加部分和合并成本，浮点求和顺序也可能改变，应一起比较。

**问：K=0 为什么返回全零？**

答：每个 C 元素是一个空和，按约定等于加法单位元 0。代码在 host 用 zeros 返回，不进入 kernel。返回未初始化 empty 会违反算子的数学语义；M/N=0 则只需要返回正确 shape 的空输出。

**问：怎样公平地与 cuBLAS 比较？**

答：先明确 shape、layout、dtype、允许的内部精度、输出容差和是否包含分配／数据转换。预热并排除首次构建，使用可靠的 GPU 时间测量。若参考路径使用 TF32 或不同精度的 Tensor Core，不能把它与严格 FP32 的结果当作完全相同计算契约而不说明。性能和数值正确性都应单独报告。

### 正确性检查最少应包含什么

至少检查 (1,1,1)、非整除的 (127,65,93)、完整 tile、K=0、M=0 和 N=0。用 FP64 reference 可以避免测试受全局 TF32 设置影响。测试容差要同时考虑绝对与相对误差，特别是期望值接近 0 的输出。
<!-- exercise:5:end -->

<!-- exercise:8:start -->
## 练习 8 的 CUDA 实现：LayerNorm 与 RMSNorm

### Shape 与公式

本 CUDA 版本只沿最后一个维度 D 做归一化。`[B,T,D]` 在连续内存中看成 `B*T` 行、每行 D 列；输入也可以只有 `[D]` 一维，此时只有一行。

LayerNorm：

\[
\mu=\frac1D\sum_i x_i,\quad
\sigma^2=\frac1D\sum_i(x_i-\mu)^2,\quad
y_i=(x_i-\mu)(\sigma^2+\epsilon)^{-1/2}w_i+b_i.
\]

RMSNorm：

\[
r=\left(\frac1D\sum_i x_i^2+\epsilon\right)^{-1/2},\quad y_i=x_i r w_i.
\]

权重和 bias 是 `[D]`，对所有行共享。代码显式要求 eps 为正，且转换成 float 后仍是有限正数，避免极小的 Python double eps 转换为 0 后悄悄改变约定。

### 逐步实现与数值选择

`norm_kernel<false>` 先 reduce 得到 mean，再重新读取 x 求中心化平方和，最后重新读取 x 做归一化和 affine。它采用两遍中心方差，而不是 `E[x²]-E[x]²`；后者对大均值、小方差的数据容易发生两个相近大数相减的消减问题。

`norm_kernel<true>` 编译掉 mean 分支，将 mean 视为 0，直接归约 x²，再做缩放，没有 bias。两个版本都用 FP32 累加；这不能消除所有误差，但避免了输入低精度时继续低精度累加的额外问题。

PyTorch LayerNorm 的参考语义使用分母 D 的方差，而非无偏估计的 D−1；可以核对 [torch.nn.LayerNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)。

### 数据流和复杂度

每 CTA 负责一行，block reduction 与前面相同。LayerNorm 读取 x 三遍，RMSNorm 两遍，再各写输出一次；weight/bias 跨行复用的缓存效果取决于具体运行。总工作 O(元素数)，每 CTA 的 scratch 是 1024 Byte。D 很小时一个 CTA 一行可能浪费；D 很大、行数很少时又可能并行度不够。

### 可以直接复述的追问答案

**问：LayerNorm 方差为什么不用 D−1？**

答：归一化层按这一行完整的 D 个特征计算其归一化统计量，公式和 PyTorch 参考语义使用 D。这里不是估计未观察总体方差的统计问题。实现时若调用默认采用不同 correction 的方差 API，就可能产生系统性差异，D=1 还会暴露除以零问题。

**问：RMSNorm 是把 LayerNorm 的 mean 固定为 0 吗？**

答：公式上它不计算或减去样本 mean，使用 x 的均方根归一化；因此可复用类似的 sum-of-squares kernel，但两者不是对任意输入等价的变换。给所有输入加一个常数时，LayerNorm 的中心化部分会消除这个平移，RMSNorm 一般不会。

**问：eps 放在 sqrt 里面还是外面？**

答：这里的约定是 `rsqrt(statistic + eps)`，即在开平方之前加。`1/(sqrt(statistic)+eps)` 是另一个函数，在 statistic 很小时差异尤其明显。对齐参考结果时必须同时对齐 eps 值和它的位置。

**问：如何进一步改进方差算法？**

答：可以研究 Welford 状态 `(count, mean, M2)` 的单遍更新与并行合并。合并两组 A/B 时，令 delta=meanB−meanA，n=nA+nB，得到 `mean=meanA+delta*nB/n`，`M2=M2A+M2B+delta²*nA*nB/n`，最后 variance=M2/n。这样可以减少读输入的遍数并改善数值表现，但单个归约元素的状态和运算更多，不能只凭遍数就判定更快。

**问：低精度版本如何做？**

答：x、weight、bias 可使用 FP16/BF16 存储，加载后转 FP32，统计量和归一化中间值用 FP32，最后按约定写回输入 dtype。特别要检查 eps 不会在低精度中先被量化成 0；也要明确 affine 是在 FP32 中完成还是先截断，二者会影响误差。

**问：LayerNorm backward 怎么写？**

答：令 `xhat=(x−mean)*r`，`r=rsqrt(var+eps)`，`g=dy*w`，则每行

\[
dx=\frac rD\left(Dg-\sum g-xhat\sum(g\,xhat)\right).
\]

输入梯度需要两个行归约。参数梯度为跨行求和 `dw=sum_rows(dy*xhat)`、`db=sum_rows(dy)`，归约方向与输入梯度不同。实际 kernel 设计要分别考虑这两种数据流，不能因为 forward 是一行一个 block 就直接照搬全部 backward。

**问：RMSNorm backward 呢？**

答：令 `g=dy*w`、`r=rsqrt(mean(x²)+eps)`，则

\[
dx=r g-xr^3\operatorname{mean}(g x),\qquad
dw=\sum_{rows}dy\,x\,r.
\]

这里没有 LayerNorm 的减均值项，也没有 bias 梯度。教材的 CUDA forward-only 接口没有实现上述 backward；这些公式是扩展方向，不能误称现有扩展已经支持训练。

### 必测边界

D=1 时 LayerNorm 的标准化部分应为 0，因此输出等于 bias；RMSNorm 则一般不为 0。再测 D=33、513、常量行、大均值但小方差、前导维为 0、不同 eps。不能用宽容差把方差分母或 eps 位置写错的问题掩盖掉。
<!-- exercise:8:end -->

<!-- exercise:9:start -->
## 练习 9 的 CUDA 实现：朴素稳定 Attention 与 Causal Offset

### 接口语义先说明白

输入 q `[B,H,L,D]`，k/v `[B,H,S,D]`。B 是 batch，H 是 head 数，L 是 query 长度，S 是 key/value 长度，D 是 head dimension。它计算每个 head 内

\[
P=\operatorname{softmax}(QK^T/\sqrt D+mask),\quad O=PV.
\]

当前接口的 q/k/v head 数和 D 必须一致，不包含 QKV 投影、GQA、dropout 或任意 padding mask。输出 shape 与 q 相同。

`causal=True` 时，第 i 个 query 的绝对位置为 `query_start+i`，只能看到 `key_index <= query_start+i`。长度为 S 的完整 prefill 通常 L=S、query_start=0；缓存已有前缀后处理最后一个 token，通常 L=1、query_start=S−1。

特别注意：如果 decode 传 L=1、S=100，却错误沿用 query_start=0，这个 query 只会看到第 0 个 key。它不会因为 L=1 就自动知道自己是第 100 个 token。本接口将这个位置显式传入，避免非方阵 causal 对齐的歧义。

### 一个 block 如何算一条 query

先把 `[B,H,L]` 展平为 query 行号 qr。`bh=qr/L` 定位 batch/head，`qi=qr%L` 定位该 head 的 query。

1. 256 线程按 key 下标分工。每线程负责若干 keys，对每个 key 在 D 维串行累加 dot product，再乘 `1/sqrt(D)`。不可见位置的 score 设为负无穷。
2. 把所有 S 个 scores 写到动态 shared memory，并对各线程局部 max 做 block max，得到这一 query 的全局最大值。
3. 原地将 shared scores 替换成 `exp(score−max)`，再归约得到分母。
4. 改为按输出 d 分工。每个线程负责若干输出维度，遍历 keys，计算归一化权重乘 V 的和。
5. 输出完成后同步，下一条 query 才可以覆盖 shared scores。

S 最大为 4096，因此 scores 至多 `4096×4=16384 Byte`，另有 1024 Byte reduction scratch。该限制是本实现选择的教学范围，不是 attention 算子的数学限制。

### 正确性与稳定性

每条 query 独立计算完整的 key 分数，稳定 softmax 与前面相同。query_start≥0、S≥1 确保 causal 模式至少 key 0 可见，不会在这份接口中产生整行全 masked 的情况；有限输入和不溢出的 dot product 仍是数值约定。

mask 只改变不可见位置的指数权重为 0，因此它们不会贡献到输出。测试使用显式绝对位置 mask 的 PyTorch FP64 oracle，避免把另一个 API 的默认 causal 对齐约定误当成本接口的答案。

### 复杂度，以及它为什么不是高性能 Attention

总算术工作量为 O(BHLSD)。本实现没有把整个 `[B,H,L,S]` score 矩阵放到全局显存中，而是每条 query 临时保存 S 个分数；但这并不自动使它成为 FlashAttention。

主要限制很具体：QK 阶段中，同一 warp 的线程负责不同 key，固定 d 时访问 K 的地址间隔为 D，访存合并不理想；dot product 在 D 上由一个线程串行完成；相邻 queries 之间没有主动共享 K/V tiles；PV 阶段若 D 很小，只有少量线程参与输出计算。这些问题比笼统地说“可以进一步优化”更值得说明。

### 可以直接复述的追问答案

**问：为什么除以 sqrt(D)？**

答：在 q/k 各维近似独立且具有相近方差的粗略假设下，D 项点积的方差随 D 增长，典型尺度随 sqrt(D) 增长。这个缩放用于控制分数尺度，避免 softmax 随维数变大而过于尖锐。真实模型不必严格满足这些独立假设，但这是常用缩放的直观动机。

**问：causal mask 是在 softmax 前还是后加？**

答：在 softmax 前把不可见 score 排除，保证分母只归一化可见位置。若先对所有位置 softmax，再把未来概率置零，却不重新归一化，剩余权重和通常小于 1，输出就变了。

**问：怎样验证 KV cache 的 decode 正确？**

答：先做完整序列的 causal attention，再用最后一条 q 和完整 k/v 调用单 token 路径，显式传 query_start=S−1，比较两者最后位置输出。进一步可逐 token 增长缓存，比较每一步与全序列对应位置。测试必须使用非零、非均匀数据，否则位置 mask 错误也可能被巧合掩盖。

**问：不写全局 score 矩阵，为什么还不是 FlashAttention？**

答：减少全局中间矩阵只是其中一个特征。高效分块 attention 还要围绕 Q/K/V tiles 做数据复用和矩阵乘法，使用稳定的 online softmax 状态在 key blocks 之间累积输出，并针对线程布局和硬件执行路径优化。这个 CUDA 版本是每 query 独立的串行 dot-product 基线，没有这些完整机制。

**问：online softmax 怎样同时累积 attention 输出？**

答：除了最大值 m 和分母 l，还维护未归一化输出向量 a。对新 key block，设分数块为 s，令

\[
m'=\max(m,\max s),\quad \alpha=e^{m-m'},\quad p=e^{s-m'},
\]

\[
l'=\alpha l+\sum p,\qquad a'=\alpha a+pV_{block}.
\]

最终输出 `a/l`。当最大值变大，旧分母和旧输出累加器必须同时按 alpha 重缩放；只改 l 不改 a 会得到错误输出。初始 m=−∞、l=0、a=0，处理全 masked 块时还需避免出现 `−∞−(−∞)` 的 NaN。

**问：prefill 与 decode 的优化关注点为何不同？**

答：prefill 有较多 queries，K/V 可以在 query tile 内重复使用，矩阵计算更容易形成较高复用；单 token decode 的 L 很小，通常要为较少输出读取较长缓存，读取 K/V、并行度与 launch 开销更加突出。具体是否带宽受限取决于 batch、head、cache 长度、精度和硬件，不能只用阶段名称下结论。

**问：attention backward 的基本公式是什么？**

答：设 G 是输出梯度，前向概率矩阵为 P，则

\[
dV=P^T G,\quad dP=GV^T,
\]

\[
dS=P\odot\left(dP-\operatorname{rowsum}(dP\odot P)\right),
\]

\[
dQ=dS K/\sqrt D,\quad dK=dS^T Q/\sqrt D.
\]

mask 位置不应贡献梯度。训练实现要决定保存 P 还是保存少量统计量后重算，以及如何组织 dK/dV 的跨 query 累加。本 CUDA 接口只有 forward，公式用于准备追问。

**问：改为 FP16/BF16 应该注意什么？**

答：点积累加、softmax 统计量和输出累加可保持 FP32，输入与最终输出采用低精度存储；若改为 Tensor Core 矩阵路径，还需要新的 tile 和 fragment layout。测试应覆盖长序列、较尖锐的分数、不同 D 以及 decode offset，而不只比较一个随机方阵。

### 最小使用例子

```python
q = torch.randn(1, 2, 17, 13, device="cuda")
k = torch.randn_like(q)
v = torch.randn_like(q)
full = cuda_ops.attention(q, k, v, causal=True)

last_q = q[:, :, -1:, :].contiguous()
last = cuda_ops.attention(last_q, k, v, causal=True, query_start=16)
torch.testing.assert_close(last, full[:, :, -1:, :], atol=2e-6, rtol=2e-6)
```
<!-- exercise:9:end -->

## CUDA 检查与验证记录

本次在没有 NVIDIA GPU 的环境中，补齐了 NVIDIA 官方离线编译组件，实际完成下面的验证。总说明中同时保存全项目验证结果。

| 检查 | 实际结果 | 能说明什么 |
|---|---|---|
| Python 语法检查 | 通过 | Python 文件能被解析，不代表 GPU 逻辑正确 |
| `bindings.cpp` 的 g++ C++17 语法检查 | 通过，使用 PyTorch 2.5.1+cpu 头文件 | C++ 绑定声明与类型可以编译解析 |
| 完整 `kernels.cu` 离线编译 | NVCC 13.0.48、目标 sm_80，通过并生成 object | host/device 源码经编译器和 ptxas，未进行扩展链接或 GPU 执行 |
| CUDA pytest | 87 个案例全部 skip | 已收集测试；因无 GPU/双 GPU 而未执行，不能计作 correctness pass |

离线编译使用 CPU PyTorch 头文件提供的 `C10_CUDA_NO_CMAKE_CONFIGURE_FILE` 开关，避免要求 CPU wheel 没有附带的 CUDA 构建生成宏文件；这是检查命令中的条件，不加入交付 loader。CUDA 13 pip 组件的 `cicc` 路径也在本地工具环境中补齐，交付 kernel 源码没有为此改写。离线编译通过不能证明运行时 stream 顺序、共享内存竞争、越界或数值误差全部正确，仍需在真实 GPU 上运行以下测试。

在具备环境的机器上运行：

```bash
python -c 'from gpu_prep.cuda_ops import load_extension; load_extension()'
pytest -q tests/test_cuda_ops.py
```

测试包含非整除维度、N=1、空张量、K=0、负数 max、稳定 softmax、norm 的大均值数据、非方阵 attention、decode offset 和非默认 stream。GPU 存在但扩展构建失败时，测试会报错，不会把编译失败伪装为 skip。

若需要排查具体的越界或同步风险，可对已缩小的失败案例使用 CUDA 的 Compute Sanitizer；普通 kernel 的 launch error check 本身不足以证明没有共享内存 race。不要把无目的地反复扩大测试规模当作优化工作，先定位哪条正确性假设尚未得到验证。
