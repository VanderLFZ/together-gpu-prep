# Triton 实现、逐题解释与面试追问

本章对应源码 `gpu_prep/triton_ops.py` 和验证脚本 `tests/test_triton_ops.py`。源码是便于逐行解释的教学实现；固定 tile 配置让不同版本容易比较。这里不预设某个配置在你的 GPU 上最快，也不把没有测量的数据写成加速比。

## 先理解：CUDA 的线程视角，与 Triton 的 tile 视角

在 CUDA C++ 中，你通常先计算一个线程的全局编号，再决定这个线程读写哪些元素。在本章的 Triton 中，你先描述一个 program 处理的逻辑张量块，再由编译器把这个块分配给线程、warp 和寄存器。在这些常规 NVIDIA kernels 中，program 通常对应一个 CTA／thread block；但是 `tl.arange(0, 256)` 表示 256 个逻辑元素，不是“启动 256 个 CUDA 线程”。

例如 `BLOCK=256, num_warps=4`：一个 program 描述 256 个元素，配置 4 个 warp；具体哪些元素交给哪些 lane，是编译器布局的一部分。不要机械地把 CUDA `blockDim.x` 替换成 Triton 的 `BLOCK`。

| 代码构件 | 你应该怎样解释 |
|---|---|
| `tl.program_id(0)` | 当前 program 在 launch grid 中的编号 |
| `tl.arange(0, BLOCK)` | 一个编译期已知长度的逻辑索引向量 |
| `row[:, None]`、`col[None, :]` | 通过广播构造二维索引，而不是用 Python 循环逐元素处理 |
| `tl.load(ptr, mask, other=...)` | 只访问 mask 为真的地址；被屏蔽的位置使用指定占位值 |
| `tl.store(ptr, value, mask)` | 控制实际写入的地址，避免尾块越界 |
| `tl.sum`、`tl.max` | program 内的归约，编译器生成所需线程间通信 |
| `tl.constexpr` | 参数是编译期常量，可用于张量形状、分支特化等 |
| `num_warps` | program 使用的 warp 数，影响并行归约与资源分配 |
| `num_stages` | 流水线调度配置；效果与 kernel、后端、硬件有关 |

本实现把 M/N/K 等尺寸作为 `tl.constexpr`。尺寸变化可能触发新的编译特化，第一次调用时间不能直接作为稳态 kernel latency。实际部署可以进一步减少不必要的特化，并把编译成本单独统计。

### 所有接口共同遵守的范围

- NVIDIA CUDA、compute capability 至少为 8.0；不在这个模块中适配 ROCm、CPU 或旧架构。当前 Triton 官方仓库的 Compatibility 也将 NVIDIA 支持范围列为 8.0+。[Triton 官方仓库](https://github.com/triton-lang/triton)
- 支持 FP32／FP16／BF16。同一个操作中的浮点输入具有相同 dtype 和 device，均为 contiguous。
- 所有浮点输出保持输入 dtype。归约、Norm 统计量、矩乘累加器、attention 在线状态使用 FP32。输出转回 FP16 仍可能溢出，内部 FP32 累加并不等于输出自动变成 FP32。
- 普通数值测试针对有限输入，不承诺完整复刻各框架对 NaN、Inf、溢出和 signed zero 的全部行为。独立 softmax 不支持全 `-inf` 行；attention 对全布尔 mask 行有专门约定。
- 单 program 处理一行的 sum／max／softmax／Norm 限制最后一维在 1–8192，目的是控制教学实现的资源范围；8192 不是 Triton 的理论最大行宽。
- vector、bias+ReLU、transpose 的零元素输出直接返回。row sum／max／softmax 要求 N>0，M 可以为 0。Norm 的归一化维度必须大于 0，前导维可以为 0。GEMM 的 K=0 返回零矩阵；M=0 或 N=0 返回空矩阵。attention 要求 S>0，L=0 可以返回空输出。
- 索引范围限制为不超过 `2**31-1` 个元素；这是当前实现的显式限制。所有逻辑二维 grid 展平成一维 launch，避免额外依赖 CUDA grid 的 y 维上限。
- 这些 raw Triton kernels 只实现 forward。grad mode 开启、任何浮点输入 `requires_grad=True` 时明确报错；在 `torch.no_grad()` 下允许推理调用。若要训练，必须另写并注册正确的 backward。
- wrapper 用 `torch.cuda.device(input.device)` 选择输入设备，退出时恢复原来的 current device；没有强制全设备同步。

`tl.dot(..., input_precision="ieee")` 的目的，是明确 FP32 点积不默默使用 TF32。它不保证整个 softmax／Norm／attention 与 PyTorch 逐 bit 相同：归约顺序、`exp`、`rsqrt`、FMA 等仍会带来舍入差异。API 口径见 [Triton `dot`](https://triton-lang.org/main/python-api/generated/triton.language.dot.html)。

<!-- exercise:1:start -->
## 第 1 题的 Triton 版本：elementwise 与融合 bias+ReLU

### 对应实现与调用

- `vector_add(x, y)` → `_vector_add_kernel`
- `bias_relu(x, bias)` → `_bias_relu_kernel`

调用示例：

```python
import torch
from gpu_prep import triton_ops

x = torch.randn(257, device="cuda", dtype=torch.float32)
y = torch.randn_like(x)
z = triton_ops.vector_add(x, y)

matrix = torch.randn(5, 513, device="cuda")
bias = torch.randn(513, device="cuda")
activated = triton_ops.bias_relu(matrix, bias)
```

### 为什么索引是这样写的？

令一个 program 负责 256 个逻辑位置：

\[
i=256\times\mathrm{program\_id}+\{0,1,\ldots,255\}.
\]

长度为 257 时，grid 大小为 `ceil(257/256)=2`。第 0 个 program 处理 0–255；第 1 个 program 的逻辑范围是 256–511，但只有 `i=256` 的 mask 为真。所有 load 和 store 都使用 `i<N`，因此尾部无效位置不会产生实际内存访问。

注意，“构造了一个超出数组范围的逻辑偏移”与“真的从那个地址读取”是两件事。掩码必须放在 `tl.load`／`tl.store` 上，不能先无条件读取，再在数值结果上 `where` 掉越界元素。`tl.load` 的 mask 与占位值语义见 [官方 API](https://triton-lang.org/main/python-api/generated/triton.language.load.html)。

对 `X[M,N]+bias[N]`，先把连续矩阵看成长度 MN 的数组。位置 `i` 的列编号为 `i % N`，因此读取 `Bias+i%N` 就实现了列广播，不需要先创建 M 份 bias。N=0 时 wrapper 在 launch 前直接返回，避免 kernel 中出现模 0。

### 精度与复杂度

加载的半精度数据先转 FP32，完成加法／ReLU 后按输出指针 dtype 转回。对有限输入，这个计算约定可用 `(x.float()+y.float()).to(x.dtype)` 或 `(x.float()+bias.float()).clamp_min(0).to(x.dtype)` 作为参考。

向量加法总计算量 O(N)，输出空间 O(N)。FP32 情况下，最简单的数据流是读 x、读 y、写 z，约 12N Byte；每元素只有一次加法。大输入往往首先受访存约束，小输入则可能主要受 launch 与调度开销影响。

bias+ReLU 融合避免了中间加法结果的一次写入和一次读取。若中间结果是 FP32、共有 MN 个元素，这两次逻辑流量合计为 8MN Byte；实际 HBM 节省还取决于缓存和执行方式，不能直接把这个数字等同于实测带宽下降。

### 面试追问与可直接使用的回答

**问 1：为什么不用 `tl.arange(0,N)`？**

答：这里需要编译器支持的静态块形状，并且不希望一个 program 无限制地包含整个输入。我选固定 256 个元素，使用多个 programs 覆盖 N；行归约则会把行宽向上补到 2 的幂。真实有效范围仍由 mask 决定。这样既处理非整除长度，也控制单个 program 的资源使用。

**问 2：`BLOCK=256`、`num_warps=4` 为什么不是矛盾的？**

答：BLOCK 是逻辑元素数，num_warps 决定执行这个 program 的 warp 数；一个 lane 可以处理多个逻辑元素。Triton 编译器负责映射。不能认为 256 个逻辑元素就必须对应 256 个硬件线程。

**问 3：下一步怎么优化？**

答：先区分规模。大数组先检查实际访存吞吐、合并访问与对齐，再比较 BLOCK 和 num_warps；小数组先看能否与相邻算子融合，减少 launch。对于 bias+ReLU，我优先保留融合，并检查是否存在隐式 `.contiguous()`、dtype cast 或额外分配掩盖收益。只有 profiler 和基准显示有必要时，才继续调整布局或更复杂的调度。

**问 4：把 BLOCK 提到 4096 一定更好吗？**

答：不一定。每个 program 处理更多元素会减少 program 数，也可能增加寄存器占用、减少可同时驻留的 programs；小 N 还会产生更多无效逻辑元素。需要覆盖代表性 N 测试，而不是只选一个大输入。

**问 5：为什么拒绝 noncontiguous 输入，不自动复制？**

答：当前地址公式利用了连续布局这一合同。自动复制会隐藏数据搬运成本，使我误把“算子加复制”的性能当成算子性能。教学版本明确拒绝；若产品要求支持一般 stride，我会把 strides 纳入地址计算，并把不同布局单独测试。也可以显式复制，但必须把复制的成本放到正确的计时范围里。

### 这题怎样练到能现场写？

先从空文件写 `idx`、`valid`、两次 load、一次 store；使用 N=257 验证第二个 program 的尾部。再改成二维 bias 广播。最后遮住代码，用 90 秒解释为什么不用同步，以及融合到底少了哪些数据搬运。
<!-- exercise:1:end -->

<!-- exercise:2:start -->
## 第 2 题的 Triton 版本：逐行 sum 与 max

### 对应实现与思路

`row_sum(x)` 和 `row_max(x)` 共用 `_row_reduce_kernel`，通过编译期 `IS_MAX` 选择归约。输入 X[M,N]；每个 program 负责一行，输出 Y[M]。

例如 N=33，会选择 BLOCK=64。load 得到 64 个逻辑数，其中后 31 个是占位值：sum 使用 0，max 使用负无穷。随后在 program 内执行 `tl.sum(values, axis=0)` 或 `tl.max(values, axis=0)`。无效元素不改变归约结果。

如果 max 的无效位置错误填 0，一行全为负数时就会返回 0；这是一个非常有价值的最小反例。只测 `rand()` 生成的正数可能发现不了这个错误。

总工作量 O(MN)，输出 O(M)。program 内的并行归约有树形合并过程，但每个线程可能处理多个数，因此不要把整道题的总工作量写成 O(log N)。对于一般输入，所有 MN 个元素都需要读取。

### 精度与形状范围

sum 在 FP32 中累加，然后转回输入 dtype。max 也用 FP32 中间表示；FP16/BF16 有限值能精确表示为 FP32。结果与 `torch.sum` 的误差仍可能来自不同归约顺序；浮点加法不满足结合律。

本版本要求 1≤N≤8192，M=0 允许空输出。N>8192 时直接报错，不悄悄生成一个资源占用未知的超大 program。要支持任意大 N，应该明确增加分块和第二阶段归约。

### 面试追问与回答

**问 1：为什么 sum 填 0，max 填负无穷？**

答：它们分别是这两个运算的单位元：任何数加 0 不变，任何有限数与负无穷取 max 不变。mask 只保证不越界；占位值仍会参与后续计算，所以必须按照归约运算选择。

**问 2：Triton 没有显式 `__syncthreads()`，归约如何正确？**

答：我使用 program 内的 `tl.sum`／`tl.max`，编译器根据数据布局生成所需的 lane、warp 间通信与同步。这里不同 programs 分别处理独立的行，彼此没有数据依赖。若我要跨 program 合并，就必须增加显式的算法阶段，不能假设 program 之间有隐式全局 barrier。

**问 3：N 很大应该怎么改？**

答：第一阶段将每一行分成若干块，每个 program 输出一个 partial sum 或 partial max；第二阶段归约这些 partials。它多一个临时数组和 kernel launch，但减少单个 program 的资源压力，也可让一行由多个 programs 并行处理。sum 的合并仍然存在浮点顺序差异，需要按目标精度验证。

**问 4：可以用 atomicAdd 吗？**

答：可以作为一种实现选择，但需要先正确初始化输出，还要考虑竞争和浮点合并顺序。直接把所有块的结果原子加到同一行，可能使结果的末位随调度变化。对于这份便于解释和验证的代码，我先选两阶段归约来扩展大行，不默认 atomic 更快。

**问 5：N=8、M 很大时，一行一个 program 合适吗？**

答：可能浪费执行资源。此时可以让一个 program 处理多行，使每个 warp 或子组处理一行，减少 program 调度开销。相反，M 很小、N 很大时，需要增加一行内部的并行度。因此 M 和 N 都会影响布局选择，不能只按总元素数选择配置。

**问 6：如何判断误差容忍度不是在掩盖 bug？**

答：先用可精确验证的输入：全 1、全负数、单个非零、N=1；再用随机输入与更高精度参考比较。对 sum 同时看绝对误差和相对误差，因为真实结果接近 0 时相对误差会夸大。若错误集中在 33、513 这类尾块尺寸，优先怀疑 mask；若误差随长度平滑增长，再分析舍入和归约顺序。

学习 `tl.sum` 与行归约映射时，可对照 [官方 Fused Softmax 教程](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)，但本题应先独立完成 sum／max，避免把归约逻辑藏在完整 softmax 里。
<!-- exercise:2:end -->

<!-- exercise:3:start -->
## 第 3 题的 Triton 版本：稳定 softmax

### 对应实现与公式

`softmax(x)` 调用 `_softmax_kernel`。仍是一行一个 program，先加载一整行的逻辑块，再依次执行 max、减 max、exp、sum、除法和写回：

\[
m=\max_j x_j,\quad e_j=\exp(x_j-m),\quad Z=\sum_j e_j,\quad p_j=e_j/Z.
\]

减去同一个 m 不改变 softmax，因为分子和分母同时乘了 `exp(-m)`。对于有限输入，所有有效指数的自变量都不大于 0，最大元素的指数为 1，这样可以避免直接计算 `exp(1000)` 一类溢出。

尾部加载为负无穷，因此它们不改变 max，指数变成 0，不改变 denominator。虽然它们已经在数值上变为 0，最终 store 仍然需要 mask，不能写入不属于输出的地址。

这里把行留在 program 的逻辑张量中，完成所有步骤后一次写回，避免把减法、exp、中间和等结果分别写到全局内存。总工作量是 O(MN)，输出空间 O(MN)，并没有创建同样规模的多个全局中间张量。

### 与 attention 中 masked softmax 的区别

独立 `softmax` 要求 N>0 且采用有限输入合同。一个全 `-inf` 行会出现 `-inf-(-inf)`，普通稳定 softmax 公式并不能自动解决这个问题。不要把“加了减 max”说成“所有输入都不会出现 NaN”。

本模块的 `attention` 额外定义：布尔 mask 全为假的 query 行输出零，并在在线递推中避免非法差值。两个接口的合同不同，详见第 9 题。

### 面试追问与回答

**问 1：为什么不能先求 exp 再统一缩放？**

答：如果先求 `exp(1000)`，溢出已经发生，后续除法无法恢复丢失的信息。减 max 必须发生在 exp 之前。这个变换既保留数学结果，又让指数输入处在更安全的范围内。

**问 2：减 max 是不是导致最小值都下溢？**

答：很小的概率确实可能下溢到 0，但这些项相对最大项本来就极小；关键是最大项保持为 1、分母非零。是否能接受这些小项的损失由精度需求决定。若任务要求保留很小的 log 概率，可能应直接计算 log-softmax，避免先产生概率再取 log。

**问 3：代码为什么把 FP16 输入转成 FP32？**

答：max 后的 exp 与 denominator 累加采用 FP32，可以扩大统计量范围并降低累加误差，尤其是行较长时。最后概率才转回输出 dtype。这不消除输入量化误差，也不意味着 output 具有 FP32 精度。

**问 4：如果一行太长，放不进当前 program 呢？**

答：我可以先对每个块求局部 max 和局部归一化分母，然后用可合并的在线 softmax 状态得到整行的全局 max 与分母，最后重新读取各块并输出概率。若下游马上计算加权和，就可以把加权和也纳入在线状态，避免单独物化概率矩阵；attention 就使用这种办法。

**问 5：这个版本一定比 PyTorch 快吗？**

答：不能保证。PyTorch 可能已使用高度优化的融合 kernel；单个 program 的寄存器使用、行宽、行数和 dtype 都影响性能。我的优势首先是数据流清楚、没有全局中间张量，是否更快要在目标形状和相同计时口径下测量。

**问 6：`tl.exp` 与 PyTorch 结果为什么不是完全相同？**

答：实现的数学函数、归约顺序和舍入路径可能不同。我会用有限输入验证归一化和逐元素误差，并选择与 dtype、行宽相称的容忍度。不能只因为结果总和接近 1 就认为每个概率都算对了；还应覆盖大幅度 logits、常数行、长度 1 和尾块。

官方教程：[Fused Softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)。在线分母的原始论文：[Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)。
<!-- exercise:3:end -->

<!-- exercise:4:start -->
## 第 4 题的 Triton 版本：矩阵转置

### 两个完整版本分别说明什么？

`transpose_naive(x)`：把输入 X[M,N] 展平，`row=i//N, col=i%N`，读取 `X+i`，写入 `Y+col*M+row`。输入是连续读取，但逻辑相邻元素的输出地址间隔 M，适合用来讨论跨步写入。

`transpose_tiled(x)`：每个 program 处理一个 32×32 输入块。输入索引为 `row[:,None]*N+col[None,:]`；使用 `tl.trans(values)` 将逻辑 tensor 的轴交换，再构造 `[输出行,输出列]` 顺序的地址 `col[:,None]*M+row[None,:]`。

M=31、N=33 时，既有列方向尾块，也有行方向尾块。load 和 store 分别根据自己的二维坐标施加 mask。输出是新分配的 contiguous `[N,M]` 矩阵，这里做的是物理转置，不是仅仅返回一个具有交换 stride 的 view。

### 不要把这个 Triton 版本描述成手写 shared memory

Triton 的逻辑 tile 和 CUDA 的 `__shared__ float tile[32][33]` 不是同一层抽象。这里没有手工声明 shared memory，也没有显式指定 33 的 padding。`tl.trans` 描述轴交换，lane layout 和所需转换由编译器决定。

因此，讲性能时应说“这个版本提供二维分块的数据流，接下来检查编译器生成的布局和访存”，不能仅凭看到 `tl.trans` 就声称读和写一定同时达到最优合并访问，更不能凭源码断言一定不存在 bank conflict。[`tl.trans` 官方语义](https://triton-lang.org/main/python-api/generated/triton.language.trans.html)

### 复杂度与指标

工作量 O(MN)，新输出空间 O(MN)。转置几乎没有有意义的浮点运算量，用 TFLOP/s 比较通常没有帮助。若每元素占 s Byte，一个直接的有效带宽指标为：

\[
BW_{\text{effective}}=\frac{2MN\,s}{t}.
\]

分子代表一次逻辑读取和一次逻辑写入；它不等于 profiler 测到的所有内存事务。非合并访问、缓存和写入策略都可能造成二者差异。

### 面试追问与回答

**问 1：为什么不能直接返回 `x.T`？**

答：`x.T` 通常只是改变 shape／stride 的 view，不一定搬动元素。这道题要求产生连续布局的转置结果，要与 `x.T.contiguous()` 这一类实际物化操作比较。拿零拷贝 view 与复制 kernel 计时，会比较两个不同的任务。

**问 2：只把索引换成二维，就一定解决访存问题吗？**

答：不一定。二维逻辑块让编译器有机会选择适合读取和写入的 layout，但物理 lane 到元素的映射才决定访存合并程度。我会检查生成代码／布局转换，再结合 memory throughput 与事务指标判断。正确性是由索引和 mask 保证的，性能还需要验证。

**问 3：CUDA 里常见的 32×33 padding 在 Triton 里怎么理解？**

答：CUDA 中 padding 是程序员直接控制 shared-memory 行跨度、改变 bank 映射的一种手段。标准 Triton 里主要通过 tile 形状、访问模式和编译器布局完成同类优化，不能把一个 logical tensor 的 shape 改成 32×33 就认为效果等价。是否需要更底层的布局控制，要看编译结果和瓶颈。

**问 4：怎么继续优化？**

答：先比较 16×16、32×32 等配置以及 num_warps，覆盖方阵和长条矩阵。若末端消费者接受转置 stride，最好评估是否能把转置融合进消费者的地址计算，从整个流程中消除一次物化；这比单独加速一次复制可能更有价值。

**问 5：为什么测试不能只用 1024×1024？**

答：这个尺寸不会检验二维尾块，而且方阵下某些错误索引更难发现。使用 31×33、1003×769，可以同时检查 M/N 没写反、输出行跨度正确、load/store 掩码都正确。转置只是搬运有限值，正常情况下应逐元素完全一致，不需要宽松的数值误差。
<!-- exercise:4:end -->

<!-- exercise:5:start -->
## 第 5 题的 Triton 版本：从 naive GEMM 到 tiled GEMM

### 朴素版本：先把地址写对

`matmul_naive(a,b)` 的输入是 A[M,K]、B[K,N]，输出 C[M,N]。每个 program 负责一段扁平输出位置：

\[
row=i//N,\quad col=i\bmod N,\quad
C_{row,col}=\sum_{k=0}^{K-1} A_{row,k}B_{k,col}.
\]

代码沿 K 顺序循环，A 地址为 `row*K+k`，B 地址为 `k*N+col`，使用 FP32 FMA 累加。它没有手工管理输入块的复用，适合当作正确性基线和面试第一版。

代码层面能看到重复加载，不代表每次都会落到 HBM：warp 广播、L1/L2 缓存与编译器优化会影响真实流量。因此描述时应说“缺少显式的 tile 复用结构”，不要把静态 load 条数直接写成实测 HBM 次数。

### 分块版本：一个 program 计算一块 C

`matmul_tiled(a,b)` 使用 `BM=BN=BK=32`。每个 program 对应一个 C 的 `[BM,BN]` 块；循环遍历 K：

1. 加载 `[BM,BK]` 的 A 块。
2. 加载 `[BK,BN]` 的 B 块。
3. `tl.dot(a_tile,b_tile,acc, input_precision="ieee")`。
4. 继续下一个 K 块，最后一次性写回 C。

在一个 K 块中，一个 A 元素可以影响 BN 个输出，一个 B 元素可以影响 BM 个输出，增加了读取数据后的计算复用。A 的边界 mask 同时检查 `row<M` 与 `k<K`；B 同时检查 `k<K` 与 `col<N`；不足的 K 部分补 0，不影响数学结果。

本代码没有使用 `%M`／`%N` 回绕边界，而是直接使用显式 load mask，方便解释所有地址的有效范围。M=0 或 N=0 在 wrapper 中返回空输出；K=0 返回初始化为零的输出，符合空内积定义。

### FP32、TF32、Tensor Core，口径必须分清

官方 `tl.dot` 文档说明，FP32 输入默认精度可能是 TF32；本版本显式使用 `input_precision="ieee"`。FP16/BF16 输入则保持其输入精度、在 FP32 中累加，最后将 C 转回输入 dtype。

不要说“用了 `tl.dot`，所以 FP32 一定使用 Tensor Core 并且更快”。严格 FP32 点积与 TF32 路径有不同的性能和精度；这里首先选择容易说明的精度合同。若做性能扩展，可增加一个显式参数比较 IEEE、TF32 等模式，但必须分别报告误差与吞吐，保持 PyTorch／cuBLAS baseline 的精度设置一致。[`tl.dot` 精度参数](https://triton-lang.org/main/python-api/generated/triton.language.dot.html)

### 为什么 tile 改变算术强度？

只考虑一次 K 分块的 A/B 读取、暂不计算输出写入和缓存，一个 tile 做约 `2*BM*BN*BK` FLOPs，加载约 `s*(BM*BK+BK*BN)` Byte：

\[
I_{\text{tile}}\approx\frac{2BM\,BN}{s(BM+BN)}.
\]

在这个简化模型里，扩大 BM/BN 可以增加数据复用，但也扩大输出累加器；BK 在分式里约掉，不意味着 BK 对实际速度没有作用，它仍影响循环次数、流水线、寄存器／shared memory 使用和尾块浪费。

完整 GEMM 的总工作量 O(MNK)，输出空间 O(MN)。tile 内还需要 A、B 和 accumulator 的片上工作集；这个资源占用不应忽略。

### 面试追问与回答

**问 1：分块为什么能加速？**

答：同样一次读取的 A/B 元素能够参与多个输出的计算，增加数据复用；`tl.dot` 也让编译器选择适合矩阵运算的指令和布局。收益取决于 tile 是否合适、精度路径、问题规模和资源使用，不能保证任意形状都变快。

**问 2：BM、BN 越大越好吗？**

答：越大可以增加复用，但 accumulator 有 BM×BN 个元素，寄存器压力会增长；资源占用过高可能减少同时驻留的 programs，甚至产生 spill。M/N 较小时，大 tile 还会做大量 masked 的无效工作。因此我会先根据矩阵长宽和硬件限制选一小组候选配置，再测量。

**问 3：为什么 BK 在你刚才的算术强度公式中消失了？**

答：因为这个简化式只比较一个 K tile 的计算量与 A/B 输入量，两者都正比于 BK。实际 kernel 还受循环、流水线、片上存储、指令形状和尾块影响，BK 仍然重要。算术强度解释的是一部分瓶颈，不是完整的性能预测器。

**问 4：small-M GEMM 怎么优化？**

答：先检查是否只有少量输出 tiles，导致 GPU 并行度不足，以及 N/K 数据搬运是否占主导。可以选择更小的 BM、适合该形状的 BN，必要时考虑 split-K 来增加并行任务。split-K 会引入 partial outputs 和归约，必须比较新增开销；不是看到 M 小就一律打开。

**问 5：split-K 为什么需要额外注意正确性？**

答：多个 programs 计算同一个 C 块的不同 K 分段，必须在最后合并。如果用 atomicAdd，要先清零输出并处理累加顺序；如果用临时数组，需要第二阶段归约。每个分段独立写完整 C 会覆盖别人的结果。精度也可能因合并顺序改变。

**问 6：如何让 A/B tiles 更容易命中 L2？**

答：可以调整 program 顺序，让相近时间执行的输出 tiles 复用相同的 A 或 B 区域，例如分组遍历多个 M tiles 后再移动 N。官方矩乘教程有 grouped ordering 的例子。但具体是否受益取决于 L2 容量、并发执行和形状，本版本先保留简单、易核对的一维行优先顺序。[官方 Matmul 教程](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)

**问 7：怎样做一个公平的性能比较？**

答：相同 M/N/K、dtype、布局、输出精度；明确是否允许 TF32；预热并排除 JIT；正确处理异步执行；不要把某一方的数据拷贝混进 kernel 时间却排除另一方的拷贝。先检验输出误差，再报告 latency、由 `2MNK/t` 得到的吞吐，以及代表性形状范围。

### 建议的优化实验顺序

先比较 naive 与固定 tile，确认差异来自明确的数据复用；随后只改变一个因素，例如 BM/BN/BK；再比较 num_warps、num_stages；最后加入分组遍历、epilogue fusion 或针对 small-M 的算法。每一步保留正确性验证和有收益／无收益的形状，面试时才能解释“为什么这样改”。
<!-- exercise:5:end -->

<!-- exercise:8:start -->
## 第 8 题的 Triton 扩展：LayerNorm 与 RMSNorm

### 公式、接口与数据布局

`layer_norm(x,weight,bias,eps)` 和 `rms_norm(x,weight,eps)` 都沿最后一个维度 N 归一化。X 可以是 `[N]`、`[M,N]`、`[B,T,N]` 等，只要 contiguous；前导维展平成 rows。weight／bias 为 `[N]`，与 x 同 dtype 和 device。

LayerNorm：

\[
\mu=\frac1N\sum_j x_j,\quad
\sigma^2=\frac1N\sum_j(x_j-\mu)^2,\quad
y_j=(x_j-\mu)(\sigma^2+\epsilon)^{-1/2}\gamma_j+\beta_j.
\]

RMSNorm：

\[
r=\left(\frac1N\sum_jx_j^2+\epsilon\right)^{-1/2},\quad
y_j=x_jr\gamma_j.
\]

本版本 epsilon 默认 `1e-5`，要求它在转换成 FP32 后仍为正有限数。仅检查 Python float 正有限还不够：`1e-300` 会在 FP32 中变成 0，`1e300` 会溢出。wrapper 用 host 端的 IEEE FP32 pack/unpack 检查并转换，不读取 GPU 数据；相应 CPU 参数测试在 `tests/test_triton_host.py`。比较框架实现时也应显式传同样的 epsilon，不能依赖双方可能不同的默认值。

### 为什么 LayerNorm 的 mask 要特别小心？

N=33、BLOCK=64 时，load 对无效位置填 0。均值用 `sum(x)/N`，这样没有问题。但是随后如果对全部 64 个位置直接计算 `(x-mean)^2`，后 31 个占位 0 会变成 `mean^2`，污染方差。

因此源码专门写成：

```python
centered = tl.where(valid, x - mean, 0.0)
variance = tl.sum(centered * centered, axis=0) / N
```

分母仍是实际 N，不是 BLOCK。这个 bug 在平均值接近 0 的随机数据上可能不明显，因此测试还使用 `100 + randn(3,33)`，让错误的 padding 贡献变得显著。

### 两遍统计，与两次读 HBM，是两回事

本 LayerNorm 先求均值，再求中心化平方和，属于两遍统计。但 x 在 program 内的逻辑块中被复用，源码只进行一次 X 的 global load；不是要求把整个 X 从 HBM 再读一遍。实际寄存器压力与 spill 要看编译器和硬件。

与 `E[x²]-E[x]²` 相比，直接归约 `(x-mean)²` 避免把两个可能很接近的大数相减。在均值很大而方差很小时，这个差分更容易损失有效位。

如果要跨很多块归并统计，可以考虑 Welford。对两个统计组 `(n_A,mu_A,M2_A)`、`(n_B,mu_B,M2_B)`，令：

\[
\delta=\mu_B-\mu_A,\quad n=n_A+n_B,
\]

\[
\mu=\mu_A+\delta\frac{n_B}{n},\qquad
M2=M2_A+M2_B+\delta^2\frac{n_A n_B}{n}.
\]

其中 M2 是相对本组均值的平方偏差和，最终方差为 M2/n。这种状态可分块合并，但比单一 sum 状态更复杂，有额外算术和通信。对当前中等行宽、一次加载整行的教育实现，先用两遍中心化统计更容易写对和解释；是否换 Welford，需要由稳定性需求、并行布局与性能证据决定。

### 面试追问与回答

**问 1：LayerNorm 的方差分母是 N 还是 N−1？**

答：这里是 N，即总体／biased variance；它是归一化 layer 所使用的定义。若用 `torch.var` 写参考，要明确选择 `correction=0` 或适用版本中的 `unbiased=False`，不能套用估计样本方差的 N−1。

**问 2：epsilon 为什么放在平方根内部？**

答：题目定义是除以 `sqrt(variance+eps)`。`sqrt(variance)+eps` 是另一个函数，尤其在方差接近 0 时差异很大。RMSNorm 同理，必须与参考实现保持相同位置与数值。

**问 3：RMSNorm 为什么少了一个步骤？**

答：它不减均值，直接使用均方根归一化。因此只需要平方和归约；LayerNorm 还要估计均值并中心化。这是两个不同的变换，不能只因为 RMSNorm 更容易融合，就在没有模型依据时随意替换。

**问 4：FP32 统计就一定数值稳定了吗？**

答：它可以降低半精度累加误差，但不能解决所有问题。输入已经量化的信息无法恢复，极端值平方仍可能溢出，最后转回低精度也会舍入。我会明确有限输入的数值范围，使用常数行、较大均值加小扰动、不同 N 和 dtype 测试；必要时再考虑更稳定的算法。

**问 5：这个 kernel 可以直接训练吗？**

答：不可以。raw Triton forward 不会自动生成正确的 autograd 链，本模块在 grad mode 下对 requires_grad 输入报错。需要另写 backward，并把梯度规则注册到 PyTorch，再验证输入、weight、bias 梯度。关闭这个检查并不会让 backward 自动正确。

**问 6：如果面试官继续问输入梯度，可以怎样回答？**

答：设 `g=dY*gamma`，LayerNorm 的 `u=(x-mu)*r`、`r=1/sqrt(var+eps)`，则一行的输入梯度可以写为：

\[
dX=r\,[g-\operatorname{mean}(g)-u\operatorname{mean}(g u)].
\]

RMSNorm 则为 `u=x*r`：

\[
dX=r\,[g-u\operatorname{mean}(g u)].
\]

weight 梯度需要沿所有前导维归约，LayerNorm 的 bias 梯度是对应 dY 的归约。除了公式，还应考虑前向保存什么统计量、反向是否重算、多个 programs 如何合并参数梯度；这些属于后续实现，当前 Triton 文件没有伪装成已经支持。

**问 7：怎样进一步加速 Norm？**

答：先按 N 与 rows 调整一 program 多行或一行多 warps；接着看寄存器、spill 和占用。若模型结构允许，可以融合 residual add、bias 或后续逐元素操作，减少中间读写。融合越多，资源占用和编译复杂度也会增加，因此仍要检查端到端收益。

官方实现与 forward／backward 思路可参考 [Triton Layer Normalization 教程](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)。
<!-- exercise:8:end -->

<!-- exercise:9:start -->
## 第 9 题的 Triton 扩展：分块 online-softmax attention forward

这是一个完整的 attention 核心前向实现。Q/K/V 的线性投影与多头拆分属于前面的 PyTorch MHA 练习；本 kernel 接收已经拆成 heads 的 Q/K/V，重点解释如何避免物化 `[L,S]` 的分数和概率矩阵。

### 接口与精度合同

```python
out = triton_ops.attention(
    q, k, v,                    # q[B,H,L,D], k/v[B,H,S,D]
    causal=True,
    query_start=0,
    attn_mask=None,              # 可选 bool [L,S] 或精确 [B,H,L,S]
)
```

- S>0，D 仅支持 16、32、64、128；B/H/L 可以使输出为空。
- Q/K/V 同 dtype 和 device、contiguous；输出 dtype 与 Q 相同。
- 布尔 mask 的 True 表示“可见”，与 causal 条件取 AND。支持 `[L,S]` 在所有 B/H 间共享，或精确 `[B,H,L,S]`；不做其他隐式广播。
- 不支持 additive mask、dropout、GQA、任意 stride、用户自定义 scale 或 backward；scale 固定为 `1/sqrt(D)`。
- q/k 点积使用 FP32 累加；FP32 输入明确 `input_precision="ieee"`。概率、中间最大值、分母、输出累加器均保留 FP32。
- P×V 也使用 FP32 点积，没有把概率偷偷转成 FP16/BF16 来获取吞吐。这是便于分析精度的教学选择，可能明显慢于生产 attention kernel。
- 全 mask 行的输出约定为零。普通 softmax 在空集合上没有自然归一化分布，这里是接口主动定义的行为，不能说它是普通公式自动得出的结果。

### 先解释为什么不能直接“每块各算一个 softmax”

对单个 query，目标为：

\[
O=\frac{\sum_j e^{s_j}V_j}{\sum_j e^{s_j}},\qquad
s_j=\frac{QK_j^\top}{\sqrt D}.
\]

如果每个 key 块单独归一化，它们的分母不同。把每个块的输出直接相加或平均，会改变不同块的权重。正确方法是保留能跨块合并的分母和加权分子，同时处理数值缩放。

### 每个 query 的三个状态

处理过一个 key 前缀后，保存：

\[
m=\max_{j\in\text{seen}}s_j,
\]

\[
l=\sum_{j\in\text{seen}}e^{s_j-m},
\]

\[
a=\sum_{j\in\text{seen}}e^{s_j-m}V_j.
\]

m 和 l 是每个 query 的标量；a 是长度 D 的向量。注意 a 是尚未除以分母的加权和，不是已经归一化的输出。初始状态为 `m=-inf, l=0, a=0`。

### 看见一个新 key tile，怎样更新？

新块的分数为 `s_new`，令：

\[
m'=\max(m,\max(s_{new})),\qquad \alpha=e^{m-m'}.
\]

旧分子和旧分母原来相对于旧最大值 m 缩放，现在要统一到新最大值 m'，所以都乘同一个 alpha：

\[
P_{new}=e^{s_{new}-m'},
\]

\[
l'=\alpha l+\sum P_{new},
\]

\[
a'=\alpha a+P_{new}V_{new}.
\]

所有 key 块处理完后才计算 `O=a/l`。新旧块都采用同一个指数基准，因此合并前后的数学定义一致。这个缩放推导比背住三行代码更重要：只更新 denominator、忘了更新 numerator，会得到错误的加权结果。

### 用两个元素手算一遍

设两个 score 为 `[0, log(2)]`，对应标量 value 为 `[2,5]`，每块只有一个 key。

第一块后：`m=0, l=1, a=2`。

第二块使最大值变成 `log(2)`，因此 `alpha=1/2`，新块指数为 1：

\[
l'=\tfrac12\times1+1=1.5,\qquad
a'=\tfrac12\times2+5=6.
\]

输出为 `6/1.5=4`，与直接计算 `(1*2+2*5)/(1+2)=4` 相同。这一例子适合面试时解释为什么旧状态必须一起缩放。

### 源码的数据块怎样对应这三个状态？

固定 BM=16、BN=32。一个 program 处理同一 batch/head 下的 16 个 queries，并沿 S 方向顺序扫描 keys：

| 逻辑 tensor | shape | 作用 |
|---|---|---|
| `q` | `[BM,D]` | 一次加载当前 query tile，跨 key tiles 复用 |
| `k` | `[D,BN]` | 构造成转置形式，便于 Q×K |
| `scores` | `[BM,BN]` | 当前 tile 的 logits，不写全局中间数组 |
| `v` | `[BN,D]` | 当前 values |
| `running_max` | `[BM]` | 每个 query 的 m |
| `denominator` | `[BM]` | 每个 query 的 l |
| `numerator` | `[BM,D]` | 每个 query 的 a |

整体 launch 的 program 编号同时编码 query block 和 batch/head：先用 `% q_blocks` 找 query block，再用 `// q_blocks` 找扁平 B/H。这样无须使用可能超过上限的二维 grid y 维。

### 空块、全 mask、尾块：如何避免 NaN？

如果尚未见过任何可见 key，`running_max=-inf`；当前块也全被屏蔽时，`new_max=-inf`。直接计算 `running_max-new_max` 就会产生 `-inf-(-inf)`。

代码仅在指数计算时建立安全参考值：

```python
safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
old_scale = tl.exp(running_max - safe_max)
probabilities = tl.exp(scores - safe_max[:, None])
```

此时 old_scale=0、所有 probabilities=0，l 和 a 仍为 0。真正的 `running_max` 继续保存 `new_max=-inf`；**不能为了避免 NaN，把状态本身永久改成 0**。否则第一次遇到极小但有限的 logits 时，它们会相对错误的最大值 0 计算指数，可能全部下溢。

如果以前已经有可见 key，而当前块完全不可见，那么 new_max 仍等于旧 max，old_scale=1，新 probabilities=0，旧状态完整保留。

循环结束后，对 denominator=0 的行，用 1 作为安全除数。由于 numerator 仍为 0，输出明确是 0。尾部无效 query 也采用这套安全计算，但最终 store mask 为假，不写越界位置。

测试特意覆盖三种不同情形：整行永远无可见 key；第一个 key tile 全 mask、第二个 tile 才出现可见 key；第一个 tile 有可见 key、后续 tiles 全 mask。它们分别检查初始化、从空状态进入非空状态、非空状态在空块中保持不变。

### causal offset：prefill 与 decode 不能混淆

key 位置是 `j=0..S-1`；query 的绝对位置定义为 `query_start+i`。可见条件是：

\[
j\le query\_start+i.
\]

完整 prefill 常用 L=S、query_start=0。缓存已经包含 S 个位置、只计算最后一个 token 的 query 时，L=1，通常应传 `query_start=S-1`。如果还传 0，唯一 query 只能看到第 0 个 key，虽然输出 shape 正确，语义却错了。

对一组新 token，query_start 必须对应它们在完整序列中的实际起点；不要无条件猜成 S−L，尤其在 cross-attention 或特殊缓存布局中。测试会比较完整 causal attention 最后一行与单 token decode 的结果。

### 计算量、存储量与当前版本的真实限制

QK 和 PV 两次矩乘的主项约为 `4*B*H*L*S*D` FLOPs，softmax 另有 O(BHLS) 的标量工作。算法没有把精确 dense attention 的主计算复杂度从平方降成线性；它主要改变中间数据存储和搬运。

当前 kernel 不分配完整 logits／probability 矩阵；工作 tile 的大小由 BM、BN、D 决定，Q/K/V/O 的全局存储随这些输入输出规模增长。如果你自己传入 `[B,H,L,S]` 的布尔 mask，这个 mask 本身仍然是 O(BHLS) 存储，不能再不加限定地称整个调用的所有内存为线性。

当前 causal 版本仍扫描所有 key tiles，并对不可见位置施加 mask；它没有跳过完全在未来的块。因此“有 causal mask”不等于代码自动节省了一半点积计算。可以根据 query block 的最大绝对位置缩短 key 循环，再对对角块进行逐元素 mask。

### 面试追问与回答

**问 1：这是不是 FlashAttention？**

答：它实现了同类的分块、在线归一化思想，用来解释如何不物化概率矩阵；我会称它为教育版 online-softmax attention forward。它没有完整生产实现的 backward、硬件流水线、精细调度和多种 shape／dtype 优化，不能用名字代替功能和性能验证。

**问 2：为什么 denominator 与 numerator 都要乘 alpha？**

答：它们原来都相对于旧 max 定义。新的 max 改变了指数的公共缩放基准，所以二者必须同时变换。只缩放其中一个，会改变分子与分母之间的比例，也就改变最终概率加权和。

**问 3：为什么不每处理一块就除一次 l？**

答：可以设计保存归一化输出的等价递推，但更新公式必须相应改变。本代码保存未归一化的 a，到最后只除一次，这样更容易核验。不能拿“保存已归一化 output”的某份实现公式，混进保存未归一化 numerator 的代码里。

**问 4：为什么 P×V 不把 P 转 half？**

答：生产实现常常需要在精度与吞吐之间做选择。本教育版把 P 保留 FP32，用于更直接地对照在线公式，避免无意引入一次概率量化。若要改用 FP16/BF16 点积，我会明确新增误差来源，比较不同 logits 范围、head dimension 和序列长度下的误差，再报告性能收益。

**问 5：如果序列很长，是否还会有数值误差？**

答：会。online softmax 避免直接指数溢出和保存整张矩阵，但仍有有限精度的点积、exp、分母和加权和累加。较小项可能下溢，累加顺序与普通 dense 实现也不同。我会用小尺寸 double precision 参考验证公式，再在长序列上检查误差随长度的变化。

**问 6：为什么 decode 可能需要不同优化？**

答：L=1 时，一个 batch/head 只有很少 query programs，沿 S 的循环很长，GPU 的并行度可能不足。可考虑 split-KV：多个 programs 处理不同 key 范围，各自输出 `(m,l,a)`，再用同样的状态合并公式归约。它增加临时状态和归并成本，但能增加并行度；是否有收益取决于 batch/head 数、S、D 和硬件。

**问 7：怎样合并两个 split-KV 结果？**

答：设两组是 `(m1,l1,a1)`、`(m2,l2,a2)`，令 `m=max(m1,m2)`，则：

\[
l=e^{m_1-m}l_1+e^{m_2-m}l_2,\qquad
a=e^{m_1-m}a_1+e^{m_2-m}a_2.
\]

最后返回 a/l；两组都为空时仍需要处理 `-inf-(-inf)`。这说明在线状态不仅能顺序更新，也能在并行分块后合并。

**问 8：怎么支持 GQA？**

答：需要区分 query head 数 Hq 与 KV head 数 Hkv，要求二者有合法的分组关系，再把每个 query head 映射到共享的 KV head；不能只把当前三个相同的 B/H 偏移照搬过去。还应关注共享 KV 是否在执行中真正复用、重复读取多少。当前版本明确拒绝不同 head 数，避免假装支持。

**问 9：怎么验证 causal mask 没写反？**

答：可以改变未来 token 的 K/V，确认较早位置输出不变；比较完整序列与分段／decode 的结果；再用 L≠S 和非零 query_start 检查 offset。仅测试方阵 L=S、query_start=0，无法覆盖缓存场景。

**问 10：下一步优化的顺序是什么？**

答：先确定主要 workload 是 prefill 还是 decode，并确认 dtype／精度目标；再调 BM/BN、num_warps 与流水线，查看寄存器和片上资源；对 causal prefill 跳过完全不可见块，对 decode 评估 split-KV；最后再考虑更低精度点积、布局和硬件专用能力。每步都保留全 mask、尾块和 offset 测试，因为性能修改不能改变这些语义。

原始思想与完整实现参考：[Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)、[FlashAttention 论文](https://arxiv.org/abs/2205.14135)、[Triton 官方 Fused Attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)。本章递推解释与代码是教学重写，不承诺复现论文或官方教程的加速数字。
<!-- exercise:9:end -->

## 怎样验证本模块，以及怎样报告结果

GPU 上运行：

```bash
python -m pytest tests/test_triton_ops.py -q
```

测试覆盖非整除 shape、dtype、矩阵 K=0、各算子的空维合同、Norm padding、causal offset、显式 mask 的空行与空块，以及非默认 CUDA device。缺少 PyTorch、Triton 或支持的 GPU 时，该测试模块明确 skip；**skip 不是通过 GPU 正确性测试**。

GPU 验证建议依次报告：实际 torch/triton/CUDA 版本和 GPU 型号；哪些测试执行、哪些 skip；随机／特殊输入正确性；再进行独立的性能测量。编译成功、Python 语法检查成功、GPU 上数值正确、目标模型端到端加速，是不同层面的证据，应分别说明。

源码对矩阵和 attention 的 FP32 `tl.dot` 精度已显式设置。若比较 PyTorch/cuBLAS 的 FP32 baseline，也应采用对应版本支持的精度设置，并记录是否允许 TF32；不要同时混用新旧互不兼容的 backend 配置接口。测试参考的矩乘与 attention 使用 double precision，避免把 TF32 默认设置意外带进正确性参考。

不建议从现有源码中只摘一个 kernel 函数、删除 wrapper 和测试就开始计时。wrapper 中的维度、dtype、device、空输入和 autograd 合同，正是让示例代码能够被可靠使用的一部分。

### 可选的 CPU 数学自检：使用 Triton 自己的解释器

包内另提供 `tools/check_triton_interpreter.py`，必须在一个新的进程中显式选择解释器：

```bash
TRITON_INTERPRET=1 python tools/check_triton_interpreter.py
```

它直接调用原始 Triton kernel，由 Triton 自带解释器执行，并与 PyTorch 高精度数学参考比较。它不会修改 GPU 入口，不会让正常 wrapper 在没有 GPU 时自动转到 CPU，也没有自造一个替代 emulator。

本次在独立的 torch 2.5.1+cpu、Triton 3.1.0、NumPy 1.26.4 环境中，34 个 FP32 数学检查通过，详见 `results/triton_interpreter.json`。覆盖非整除长度、两种 transpose/GEMM、K=0、Norm padding、attention 的不同 D、全 mask 行、前后空 tile 和非零 causal offset。

这份证据只支持高层索引、mask 与数学递推，**不验证 GPU 的 lane layout、线程同步、竞态、资源用量、实际浮点指令或性能**。GPU 测试仍需要在真实硬件运行。

另外，本次旧版 Triton 3.1.0 与 NumPy 2.3.5 的组合连独立的最小 load/store probe 都未通过；换到独立路径的 NumPy 1.26.4 后恢复。工具因此先检查一个与练习源码无关的内存读写 probe，失败时报告 `blocked_interpreter_memory_preflight`，避免把解释器环境异常误报为练习代码错误。若复现旧版解释器，建议为它单独准备兼容依赖；不要为了这项可选检查随意改动已经工作的 GPU 环境。
