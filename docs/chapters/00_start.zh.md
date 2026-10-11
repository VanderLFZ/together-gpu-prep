# 怎样使用这套材料

这套材料对应此前设计的 12 道核心练习，目的是准备 GPU programming / ML systems 面试。它是原创练习和参考答案，不代表 Together AI 公布了这 12 道原题，也不能用它推断所有面试轮次都固定不变。

你需要同时训练三种能力：从公式写出正确实现；把工作分配给线程、warp、block 或 Triton program；用实验判断优化是否有效。只记住最后一版代码会漏掉面试中最常见的追问：为什么这样索引、为什么这里需要同步、为什么另一种形状会变慢。

## 一、先分清材料中的四个层次

| 层次 | 代码所在位置 | 用途 | 是否训练可用 |
|---|---|---|---|
| PyTorch 公式实现 | gpu_prep/torch_layers.py | Linear、Norm、Attention、MHA、KV cache | 常规 forward 支持 autograd；cache 增量推理有明确限制 |
| CUDA C++ 教学 kernel | gpu_prep/csrc/kernels.cu | 看清线程、共享内存、同步和尾块处理 | 原始入口仅 forward |
| Triton 教学 kernel | gpu_prep/triton_ops.py | 看清 block 表达、mask、归约和矩阵 tile | 原始入口仅 forward |
| PyTorch 自定义算子接入 | gpu_prep/custom_ops.py | 注册算子、fake、autograd 与 compile 接入 | Bias+ReLU 示例提供 backward |

GPU 的基础入口刻意限制输入布局和 dtype，使你能先审查核心算法。CUDA C++ 使用 contiguous FP32；Triton 使用 contiguous FP32、FP16、BF16，并要求 NVIDIA compute capability 至少 8.0。它们没有悄悄替你搬设备、转 dtype 或复制 noncontiguous 输入。需要复制时，由调用者显式写 contiguous()，这样性能分析才不会把布局转换藏起来。

这些实现可以作为练习基线，不能仅凭“实现了 tiled”“用了 Triton”就认定优于 cuBLAS、PyTorch 或生产级 FlashAttention。源码中给定的 tile 是便于推导的起点，没有在当前无 GPU 环境中被实测为最佳配置。

## 二、运行顺序

先进入解压后的项目根目录。建议创建独立虚拟环境，随后按照 PyTorch 官方安装选择器安装与你的 NVIDIA 驱动、GPU 和 CUDA 运行时兼容的 PyTorch。不要在已有 CUDA 版 PyTorch 的环境中为了照抄命令而把它换成 CPU 版。

~~~bash
python -m venv .venv
source .venv/bin/activate
# 先安装与你机器匹配的 PyTorch，再安装本项目。
python -m pip install -e ".[test,cuda]"
# Linux / NVIDIA 上安装与 PyTorch、GPU 兼容的 Triton：
python -m pip install -e ".[triton]"

python -m gpu_prep.check_env
python -m pytest -q
python examples/torch_layers_demo.py
python examples/compile_demo.py --help
python -m gpu_prep.roofline
~~~

有真实 NVIDIA GPU 后再运行：

~~~bash
python -m gpu_prep.check_env --require-gpu
python examples/run_gpu.py --backend all --op all
python -m pytest tests/test_cuda_ops.py tests/test_triton_ops.py -q
python -m gpu_prep.benchmark --op matmul --shape 128,256,384 --backends torch,cuda,triton --output results/matmul_fp32.json
~~~

CUDA 扩展第一次调用会编译。GPU PyTorch wheel 提供运行所需的相关库，但不能据此假定你的机器已经有 nvcc；CUDA C++ 路径还需要完整的编译工具链、可用 C++ 编译器和 Ninja。Triton 路径不需要本项目的 CUDA 扩展编译。请保留第一次编译的错误信息，按缺失依赖修复；不要通过吞掉异常后换成 PyTorch 实现来假装自定义 kernel 已经运行。

FP16/BF16 对比只选支持的后端：

~~~bash
python -m gpu_prep.benchmark --op matmul --shape 256,512,768 --dtype float16 --backends torch,triton
~~~

形状参数的含义见每道题：GEMM 的 shape 顺序为 M,N,K，Attention 的顺序为 B,H,L,S,D。性能脚本只接受正维度；零维度与空张量属于正确性测试，避免把“没有实际工作”混入吞吐量结果。

## 三、每题按这个顺序准备

1. 不看答案，用 30 秒写出输入、输出、shape、dtype、layout 和数学公式。
2. 先写 PyTorch reference，至少准备一个很小、可以手工核对的例子。
3. 写直接实现；对 non-multiple、很短维度、很长维度和零尺寸约定做检查。
4. 解释每个线程或 program 负责哪一部分输出，以及它需要读哪些输入。
5. 确认同步与掩码，再讨论共享内存、寄存器、向量化、融合和 Tensor Core。
6. 先验证精度，再运行统一 benchmark。记录硬件、版本、形状、dtype 和计算精度策略。
7. 用“观察—假设—改动—验证—结果—适用范围”的顺序讲优化。

题目中的“答案示例”可以练口述，但不要把示例假设当成你真实做过的实验。没有测量时应说“我的假设是……，我会看……指标来验证”，而不是说某个优化必然快两倍。

## 四、全局正确性约定

### 4.1 容差不等于随意放宽

FP32 的归约顺序、FMA 与不同实现的 exp 近似会影响最后几位。比较浮点结果一般使用绝对误差和相对误差的组合，而不是要求逐 bit 相同。接近零时相对误差会失真，因此不能只报 max_relative_error。矩阵元素很多时，还应检查是否存在少数离群错误，而不是只看平均误差。

低精度输入的内部累加通常用 FP32，输出再转回输入 dtype。FP16/BF16 的输入量化误差仍然存在，FP32 累加不能恢复输入中已经丢失的有效位。基准对照还必须统一 FP32/TF32、累加精度以及算子语义。

### 4.2 尾块与空输入是两类问题

N=1003、BLOCK=256 时，最后一个 program 的一部分 lane 越界，这是尾块，应该用 load/store mask。N=0 时根本没有有效工作，通常由 host wrapper 提前返回。对于 row_max、softmax 这类沿空维度没有本题定义输出的操作，wrapper 明确拒绝 N=0。

GPU 行归约和 Norm 允许行数为零，但要求被归约的最后一维大于零。GEMM 的 K=0 有清楚的数学意义：非空输出为零；M=0 或 N=0 则返回空输出。Attention 的 GPU 练习要求 S>0，PyTorch 公式实现对 S=0 有单独定义。完整行为由源码的检查和 tests 共同给出。

### 4.3 Mask 与 NaN

普通 softmax 的数学定义不包含一整行负无穷的归一化：减去最大值时会出现负无穷减负无穷。普通 softmax 练习约定有限输入；Attention 单独处理 boolean mask，并把全遮蔽行定义成零输出。不能把两个合同混为一谈。

同样，测试有限随机数不能证明 NaN/Inf 的传播与框架完全一致。CUDA 的 bias_relu 专门保留 NaN，而其他基础 kernel 不承诺所有特殊值兼容。代码不会为每次调用扫描整个张量来检查有限性，那本身会增加一次完整读和同步；输入约定由调用者负责。

### 4.4 Forward、backward 与 compile 是三个检查点

一个 forward kernel 返回正确数值，不等于它能训练；注册了 autograd，不等于导数公式正确；能被 torch.compile 捕获，也不等于 Inductor 已经把它和邻近算子融合。第 11 题分别演示 forward reference、gradcheck、fake、opcheck 和 compile 检查，避免用一个“测试通过”覆盖几个不同结论。

## 五、源码阅读地图

| 题号 | 要先读的函数/文件 | 要背熟的核心句 |
|---|---|---|
| 1 | vector_add、bias_relu | 先确认连续访存，再判断融合节省了什么 |
| 2 | row_sum、row_max | 归约的 identity 和同步范围必须正确 |
| 3 | softmax | 减最大值保稳定，FP32 做统计与归一化 |
| 4 | transpose_naive、transpose_tiled | 物理转置需要同时改善读和写的访问模式 |
| 5 | matmul_naive、matmul_tiled | Tile 的价值来自输入复用，代价是资源占用 |
| 6 | roofline.py | 用一致单位算 FLOP/byte，再判断模型里的瓶颈 |
| 7 | ManualLinear、split_qkv_heads | Shape 正确还要检查 stride 与是否复制 |
| 8 | ManualLayerNorm、ManualRMSNorm | 统计维度、方差分母、eps 位置和精度要一致 |
| 9 | Attention、MHA、KVCache | Prefill 和 decode 的 query 绝对位置不同 |
| 10 | compile_demo.py | 分开 graph break、guard/recompile、首次编译和稳态执行 |
| 11 | custom_ops.py | schema、fake、autograd、compile 各有责任 |
| 12 | benchmark.py | 先定义测什么，再对正确且可比的实现计时 |
