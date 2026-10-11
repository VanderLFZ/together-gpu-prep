<!-- exercise:6:start -->
## 第 6 题：Roofline 计算、解释与答案

### 题目与完整实现

给定硬件峰值算力 P、可持续内存带宽 B，以及程序的浮点运算量 F 和数据搬运量 Q，估计算术强度、性能上界和执行时间下界。为 vector add、square GEMM 和 skinny GEMM 分别计算，并解释为什么同样叫 GEMM，瓶颈可能不同。

完整代码是 gpu_prep/roofline.py。它提供 Workload、vector_add_workload、gemm_workload、roofline_estimate 和 amdahl_speedup；既能直接调用，也能作为命令行运行。对应 tests/test_roofline.py 验证单位、数值、空维度约定和非法参数。

这道题以计算与分析为主，不需要额外写一个 CUDA 或 Triton 的 roofline kernel。输入性能模型的是前面实际 kernel 的工作量和测量数据，模型本身用 Python 即可清楚表达。

### 1. 公式要从单位解释

算术强度 AI = F / Q，单位 FLOP/byte。

计算所需时间下界 t_compute = F / P。

内存所需时间下界 t_memory = Q / B。

理想模型中的时间下界 t_min = max(t_compute, t_memory)。

对应的性能上界 perf_max = min(P, B × AI)。

转折点 AI_ridge = P / B。AI 比它低时，模型的带宽屋顶更低；AI 比它高时，模型的算力屋顶更低。

这里用十进制单位：1 TFLOP/s = 10^12 FLOP/s，1 TB/s = 10^12 byte/s，1 GB/s = 10^9 byte/s。GiB 则是 2^30 byte，不能在数字计算时把 GB 与 GiB 混用。一次 fused multiply-add 做一次乘法和一次加法，按通常性能报告约定计 2 FLOPs。

“下界”与“预测”必须区分。这个模型没有直接包含 launch latency、调度、访存延迟、依赖链、寄存器 spill、额外 tensor layout conversion、各层缓存流量和同步。算出 5 微秒，不代表真实实现就会在 5 微秒完成。

### 2. 手算例一：FP32 vector add

设 z=x+y，有 N 个元素。每个元素做 1 次加法；理想情况下读 x、读 y、写 z，各 4 byte，因此 F=N，Q=12N，AI=1/12≈0.0833 FLOP/byte。

假设一块“教学用虚构 GPU”有 P=100 TFLOP/s、B=2 TB/s。它不是对任何真实型号的规格声明。

此时 ridge=100/2=50 FLOP/byte，vector add 明显落在低 AI 一侧。模型里的性能上界是 2×(1/12)=0.1667 TFLOP/s，而不是 100 TFLOP/s。N=1,000,000 时，数据量为 12,000,000 byte，带宽时间下界为 6 微秒；计算时间下界只有 0.01 微秒。

**可复述答案：**“Vector add 的每个输出只做一次加法，却要读两个数、写一个数。按 FP32 理想流量算，AI 只有 1/12，所以我优先检查访存连续性、有效带宽和 launch 开销。对大数组，共享内存没有重复使用的数据可供缓存，加入 shared memory 很可能增加操作和同步；对很小数组，即使 AI 很低，实际也可能主要受 launch latency 影响。”

### 3. 手算例二：Square GEMM

A 是 M×K，B 是 K×N，C=AB 为 M×N，beta=0，不需要读取旧 C。按通常约定，F≈2MNK。算法层面的理想最小流量 Q=s(MK+KN+MN)，其中 s 为一个元素的 byte 数。

对于 FP32、M=N=K=1024：

- F=2,147,483,648 FLOPs。
- Q=12,582,912 byte。
- AI≈170.67 FLOP/byte。
- 在上面的虚构 GPU 上，t_compute≈21.475 微秒，t_memory≈6.291 微秒。
- 因而模型的计算屋顶更低，时间下界约 21.475 微秒。

但这个流量模型假设输入得到了理想复用。朴素实现每个线程独立循环 K，会产生大量重复访问。L1/L2 可能缓解部分 DRAM 流量，却不会消除每条 load 指令的执行成本或自动保证所有复用理想。不能仅凭算法 AI 很高就宣称一个没有测量的 naive kernel 已经 compute-bound。

**可复述答案：**“我会先区分算法的最小流量与该实现实际产生的流量。GEMM 的算法 AI 可以很高，但 naive 实现未必达到。Tiling 的作用是让加载到 shared memory 或寄存器的数据服务多个 FMA，从而降低实际搬运开销并提高执行效率；它没有改变 C=AB 的数学 FLOP 数。”

### 4. 手算例三：Skinny GEMM

取 M=1、N=K=4096，FP32：

- F=33,554,432 FLOPs。
- Q=67,141,632 byte。
- AI≈0.49976 FLOP/byte。
- 在虚构 GPU 上，计算时间约 0.336 微秒，带宽时间约 33.571 微秒。
- 模型里的带宽屋顶更低。

直觉是 B 很大，M=1 时 B 的每个元素只服务很少的输出行，跨行复用不足。即使使用相同 dtype、同一块 GPU，“大 square GEMM”与“single-token decode 中的线性层”也可能表现出很不同的资源利用。实际模型还要考虑 batch、权重是否被缓存、量化、并发请求和算子融合。

**可复述答案：**“我不会直接把大方阵 GEMM 的最佳 tile 搬到 M=1 的场景。M 很小时，输出 tile 的浪费和并行度不足可能突出。我会按实际 M/N/K 分桶，比较较小 M tile、适合 GEMV 的路径，以及必要时 split-K 的收益和归约成本。”

### 5. 实测结果怎样接入

假设某次真实测量得到 t 秒，则 achieved_FLOPs=F/t。若使用逻辑流量 Q，则 Q/t 只能称为 logical effective bandwidth。要声称实际 HBM 吞吐率，需要对应层级的 profiler 计数器；缓存命中可能让逻辑有效带宽超过硬件标称 HBM 带宽，这本身不说明计时一定错误。

还必须选对 P：FP32 CUDA Core 峰值、TF32 Tensor Core 峰值、FP16/BF16 Tensor Core 峰值，以及是否假设结构化稀疏，分别是不同的屋顶。拿 FP32 SIMT kernel 去除以一个稀疏 FP16 峰值，得到的利用率没有可解释性。

硬件峰值是规格上限，可持续带宽通常低于标称带宽。做实用诊断时，可以同时展示规格 roofline 与在类似条件下测得的 sustained roofline，但需要标明各自来源。Nsight Compute 的 roofline 还可按内存层级分析，不能把 L1、L2、HBM 的 byte 计数混为同一个 Q。

### 6. 高频追问与完整回答

**问：AI 低就一定是 bandwidth-bound 吗？**

“在简化 roofline 模型里，低 AI 意味着带宽上界更低；但真实 kernel 可能远没有饱和带宽，而是受 launch latency、访存延迟、并行度不足或指令依赖限制。我会结合数据规模和 profiler 看实际 DRAM 吞吐率、active warps、stall 原因与 kernel 时间，避免只用公式下最终结论。”

**问：为什么提高 occupancy 后反而变慢？**

“Occupancy 描述并发驻留资源的一部分，不是目标函数。为了提高它而减小寄存器，可能引入 spill；减小 tile 可能降低数据复用；过多 block 也可能改变缓存行为。我要验证的是延迟或吞吐量是否变好，以及这个变化是不是在相同语义和精度下获得的。”

**问：Shared memory 一定能增加 AI 吗？**

“共享内存可以减少重复的高层内存访问，但不是凭空增加数学运算。对有复用的 GEMM tile 很有价值；对读一次就写一次的 vector add，通常没有该复用。分析时我会说清 AI 的 Q 是算法理想流量、DRAM 实测流量还是 shared-memory 流量。”

**问：一个实现的 TFLOP/s 很低，怎样决定优化什么？**

“先定位规模和正确的计算屋顶，再看是否达到了对应内存层级的带宽。若 kernel 很短，我会先看 launch/调度成本；若带宽高且 AI 低，优先减少 bytes 或融合；若带宽和算力都低，检查并行度、访存模式、依赖、同步、layout conversion 与资源占用。最后只改一个主要变量，重新验证。”

**问：Transpose 的 FLOP/s 有意义吗？**

“物理转置主要搬运数据，没有有意义的大规模浮点算术。我会报告延迟和基于读写量计算的 GB/s，并用 profiler 判断真实内存行为，而不会为了套表强行给它一个计算利用率。”

**问：Softmax 能不能简单按几个 FLOPs 算？**

“可以建立一个粗模型，但 exp、除法、归约、比较与普通 FMA 的吞吐特性不同，只给一个 FLOP 数容易隐藏重要信息。我会先分析读写次数、归约结构和行宽，再结合具体指令与 profiler。源码的通用 benchmark 因而不为 softmax 伪造精确的 FLOP/s。”

**问：量化为什么可能让 decode 更快？**

“如果权重读取是主要成本，减小权重 byte 数可以提高有效 AI，降低传输量。但要同时计入反量化、scale 读取、重排、额外 kernel 与精度约束。是否更快必须按 batch、shape、缓存条件和量化实现测量，不能只按位宽比例预测速度。”

<!-- exercise:6:end -->

<!-- exercise:12:start -->
## 第 12 题：性能实验、诊断与项目答辩

### 题目与完整实现

选择前面的一个算子，对 PyTorch、CUDA 和 Triton 的实现做数值检查、稳态计时与瓶颈分析。提交硬件和软件版本、输入分布、shape、dtype、计算精度、测量范围、原始样本、优化前后结果及适用边界。不能只交一个 speedup 数字。

完整入口是 gpu_prep/benchmark.py。统一正确性演示是 examples/run_gpu.py。它们不在没有 GPU 时自动退回 CPU 并继续生成“GPU 性能”表。CPU 单元测试只检查模型计算、公式 reference 和计时统计的定义。

### 1. 先定义比较对象

“比较一个 kernel”可以指三种不同对象：

| 对象 | 计入什么 | 适合回答什么 |
|---|---|---|
| 单个 device kernel | 该 kernel 的 GPU 执行；通常由 profiler 或预分配 launch 测量 | 线程映射、访存、指令改进是否有效 |
| Resident-input 算子 wrapper | 输入已在 GPU；包括 wrapper 发出的工作与输出分配 | 自定义算子 API 替换后能节省多少 |
| 应用端到端 | 实际请求中的调度、复制、模型计算、排队等选定阶段 | 用户可见延迟或吞吐是否改善 |

本项目统一脚本测第二类，同时报告 GPU event 和同步后的 host wall 时间。每个 wrapper 自己分配输出，PyTorch 基线也遵循同样的算子语义。没有把 Python 内层循环、编译器耗时和服务器排队拼到某个没有标注的“kernel latency”里。

每次 wrapper 调用中的 shape、dtype、device 等 metadata 检查也在该测量范围内；排除的是独立的数值正确性比较。FP32 matmul 明确关闭 TF32；FP16/BF16 matmul 明确关闭 PyTorch 的 reduced-precision reduction 开关，避免把不同累加策略混为同一种精度。环境 JSON 会记录这些策略。Attention 的 PyTorch 基线仍可由 SDPA 选择可用 fused backend，不承诺与教学实现拥有逐步相同的运算和舍入顺序。

物理转置的 PyTorch baseline 是 x.T.contiguous()。只测 x.T 通常只是视图/stride 变化，做的工作与写出转置结果的 CUDA kernel 不一样。Bias+ReLU 的 PyTorch eager baseline 包含其张量运算过程，融合 kernel 则可能减少中间张量；若又比较 torch.compile，需要把它列为独立实现并单独预热。

### 2. 为什么普通 Python 计时会错

GPU launch 通常异步。若写 start=time.perf_counter()、调用 kernel、立即读取 end，测到的主要可能是 CPU 提交时间，而 GPU 尚未完成。

正确的 host wall 测量需要在开始前同步清掉前序工作，在结束前同步等待本次工作完成。正确的 CUDA event 测量需要在被测工作所属的 stream 上 record start 和 end，并等待 end 完成后读取 elapsed_time。多 stream 程序必须建立相应依赖；单个 stream 的 end event 不会神奇地覆盖没有依赖的其他 stream 工作。

本脚本的函数合同是“所有工作在所选 device 的当前 stream 上发起”。CUDA 入口使用 PyTorch current stream，Triton 在当前 device/stream 下 launch。测试还包含非默认 stream 的情形。若你扩展为异步 H2D、计算与 D2H 多 stream pipeline，请同时修改计时边界。

### 3. 首次执行、预热与稳态

第一次调用可能触发 Python import、CUDA context 初始化、扩展编译、Triton JIT、torch.compile、autotune 或 allocator 初始化。这些对生产系统有时很重要，但不应混成一个没有说明的稳态 kernel 时间。

脚本先构建数据和 reference，再对所有后端分别做正确性检查，因此可能出现的 JIT 已经发生在计时前。随后 warmup，再做多个 repeat，每个 repeat 连续调用多次，将批量 elapsed time 除以调用次数。这样能降低短 kernel 的计时噪声。

这也带来一个明确限制：结果属于重复使用相同输入的 warm-cache 模式。工作集很小时，L2 复用可能显著。若实际线上每次访问不同数据，应设计轮换输入或真实 workload；不要只给脚本加一句“已经 warmup”就认为数据访问分布匹配。

### 4. 统计数字怎样解释

脚本保存每个 repeat 的平均每次耗时，给出 median、min 和“repeat 平均值的 p95”。它没有把这个 p95 冒称为线上单请求 p95，更不是在模型服务排队条件下测到的 tail latency。

例如 7 个 repeat，每个 100 次调用，得到的是 7 个平均耗时样本。它适合粗看稳定性，不能推断 700 个单次请求的完整分布。若需要单请求 p95/p99，应按真实请求边界采样足够多的请求，说明并发、到达过程、queueing、batching、warm/cold state。

报告 speedup 时写清分子分母：speedup = baseline_time / candidate_time。不要把“延迟减少 20%”与“吞吐增加 20%”等同。基线 10 ms，优化后 8 ms，延迟减少 20%，同样串行工作可得 1.25× 速度，也就是理想处理率增加 25%。

### 5. 可直接执行的实验矩阵

先小后大，每轮保留日志和 JSON。以下参数是建议的实验设计，不是已经跑出的性能数据。

~~~bash
# 非整除尺寸，先确认边界正确
python examples/run_gpu.py --backend all --op all

# launch/调度成本明显的小 vector 与较大 vector
python -m gpu_prep.benchmark --op vector_add --shape 1024 --output results/add_small.json
python -m gpu_prep.benchmark --op vector_add --shape 16777216 --output results/add_large.json

# 不同归约行宽，观察单行策略的限制
python -m gpu_prep.benchmark --op softmax --shape 1024,127 --output results/softmax_127.json
python -m gpu_prep.benchmark --op softmax --shape 1024,4097 --output results/softmax_4097.json

# 同一 shape 下比较 naive 与 tiled
python -m gpu_prep.benchmark --op transpose --shape 1023,769 --variant naive --output results/trans_naive.json
python -m gpu_prep.benchmark --op transpose --shape 1023,769 --variant tiled --output results/trans_tiled.json
python -m gpu_prep.benchmark --op matmul --shape 127,65,93 --variant naive --output results/gemm_naive.json
python -m gpu_prep.benchmark --op matmul --shape 127,65,93 --variant tiled --output results/gemm_tiled.json

# skinny 与 square 分开；大型 naive GEMM 可能很慢，先用 tiled
python -m gpu_prep.benchmark --op matmul --shape 1,4096,4096 --output results/gemm_skinny.json
python -m gpu_prep.benchmark --op matmul --shape 1024,1024,1024 --output results/gemm_square.json

# 严格 FP32 与低精度分别比较
python -m gpu_prep.benchmark --op matmul --shape 512,512,512 --dtype float16 --backends torch,triton --output results/gemm_fp16.json

# 长度非整除的 noncausal attention；causal/decode 正确性另由 tests 覆盖
python -m gpu_prep.benchmark --op attention --shape 1,4,127,257,64 --output results/attention.json
~~~

执行前确认显存足够；尤其是 PyTorch float64 reference 会使用额外显存。脚本创建 reference 是为了正确性检查，不代表生产实现需要保留该 reference。只计时前向 wrapper，生成 reference 和复制输入不在计时范围。

### 6. Profiler 诊断顺序

第一步看应用时间线，确认目标算子在实际临界路径上的占比。第二步只捕获代表性 kernel，检查 grid/block、寄存器、shared memory、occupancy、DRAM/L2 吞吐与 stall。第三步提出一个可检验假设，修改一个主要因素。第四步重新跑同样的正确性与计时实验。

Nsight Systems 适合观察 CPU/GPU 时间线、launch gap、同步和重叠；Nsight Compute 适合深入单个 kernel 的硬件指标。性能分析工具可能重放 kernel 或显著改变运行过程，所以 profiler 下的总 wall time 不能直接当线上性能。先在普通运行里得到基准，再用 profiler 解释原因。

可作为起点的命令如下；不同版本支持的选项以已安装工具的 --help 为准：

~~~bash
nsys profile -o results/gemm_timeline python -m gpu_prep.benchmark --op matmul --shape 512,512,512 --backends triton
ncu --set full -o results/gemm_kernel python examples/run_gpu.py --backend triton --op matmul
~~~

实际 profiling 时应缩小捕获范围并选择目标 kernel，避免让大量 reference/JIT/预热工作淹没报告。若 GPU counter 权限受限，先使用能获得的时间线和计时数据，并如实说明缺失；不要把“没有拿到计数器”解释成“带宽利用率为零”。

### 7. 面试追问与可直接练习的回答

**问：你让 softmax 快了 2×，模型端到端能快多少？**

“要看它在临界路径上的占比。假设原来只占 5%，且没有重叠关系变化和额外开销，Amdahl 模型给出的整体速度是 1/(0.95+0.05/2)≈1.02564×。对应总延迟减少 2.5%，处理率增加约 2.564%。如果 kernel 与其他工作重叠，就还要用真实时间线重算，不能直接相加各 kernel 的耗时。”

**问：为什么你的 fused bias+ReLU 比两个 PyTorch 运算快？**

“我的预期是减少一次中间结果写回和读取，且可能减少一次 launch。但我会按张量大小验证：小张量可能主要受 launch 影响，大张量可能受流量影响。若使用 torch.compile 作为基线，它也可能已经融合，所以我会把 eager 与 compiled baseline 分开报告。最终还要看融合后的寄存器压力是否抵消收益。”

**问：你怎么证明一个优化有效，而不是噪声？**

“先在相同输入语义、dtype、精度策略和版本下检查正确性，排除首次编译，再交替或打乱测量顺序，多次采样并保存原始数据。我会在目标 shape 分布上看稳定收益；若差异接近运行波动，增加有目的的重复或控制时钟/负载，再决定是否接受，而不只挑最漂亮的一次。”

**问：为何 Triton tiled 版本比 naive 慢？**

“Tiled 只描述组织方式，不保证适合所有 shape。小矩阵可能被 launch 成本支配，固定 tile 可能产生很多 masked work，layout conversion 和寄存器压力也可能增大。严格 FP32 路径未必使用 Tensor Core。我要检查具体 generated code 和 profiler，确认实际执行路径，再决定改 tile、dtype/精度合同还是调度。”

**问：GPU event 时间为什么比 CPU wall 时间短？**

“事件反映所选 stream 时间区间，host wall 还包含 Python 调度、CPU 工作和同步成本。不过不能机械地把二者之差解释成某一个固定的 launch overhead，因为异步提交、GPU idle gap 和流水重叠会影响二者。这个脚本把它们分开报告，避免混淆；需要更精细归因时用时间线。”

**问：你是否考虑了编译和内存分配？**

“我把第一次编译单独看，稳态测试中先完成 JIT 与预热。当前 wrapper benchmark 包含实现内部的输出分配；如果要分析纯 device kernel，我会改成预分配输入输出并直接 launch，明确这是另一个测量对象。生产端到端还应看模型加载、动态 shape 的重新编译、allocator 行为和服务生命周期。”

**问：Baseline 应该选你自己的 naive 实现还是厂商库？**

“两者回答不同问题。Naive baseline 能说明某个教学优化为何有效，但要判断生产价值还需对照成熟实现，如同语义的 PyTorch/cuBLAS/SDPA，并确保精度、layout 和功能一致。我会同时列出最初版本与强基线，避免只证明自己超过了一个刻意低效的起点。”

**问：你做了一个只对少数 shape 很快的 kernel，怎么落地？**

“先确认这些 shape 在真实分布里的频率和耗时占比，再设置清晰的 dispatch 条件，并保留其他 shape 的正确 fallback。测试覆盖条件边界，记录编译与缓存成本。只有加权端到端收益能够抵过维护复杂度和回归风险时，专用路径才有价值。”

### 8. 项目答辩模板：把自己的真实材料填进去

“我的场景是【训练/推理、prefill/decode、batch 和 shape 分布】。原始测量显示【明确的时间线或指标】，其中【目标工作】占【真实比例】。我据此提出【具体瓶颈假设】，实现了【具体改动】。我用【reference、边界 case、梯度检查】验证正确性，用【硬件、软件版本、计时范围】测性能。结果在【适用范围】为【实测数字】，但在【其他形状/条件】存在【限制】。下一步我会验证【一个仍未证实的假设】。”

如果某部分来自仿真、推算或模拟器，直接说清：“这里是模型估计/仿真结果，尚未在目标硬件验证。”如果 GPU 型号、驱动、功耗条件或真实日志缺失，不要补一个看似合理的数字。面试中能清楚说明证据的边界，比无法复现的大幅 speedup 更容易让对方判断你的工程方法。

### 9. 两小时模拟面试的建议分配

| 时间 | 练习内容 | 通过条件 |
|---|---|---|
| 0–15 分钟 | Vector add 或 bias+ReLU | 写出合法索引、尾块 mask、正确调用 |
| 15–35 分钟 | Reduction 或 softmax | 解释 identity、同步和稳定性，处理非整除 |
| 35–60 分钟 | Transpose 或 GEMM | 先正确基线，再推导 tile 的读写与同步 |
| 60–80 分钟 | Linear/QKV 与 Norm | Shape、stride、统计轴、dtype 不出错 |
| 80–100 分钟 | Attention 与 decode mask | 能说明非方阵 causal 位置与 KV cache |
| 100–120 分钟 | Roofline、compile、实验答辩 | 给出模型假设与下一步可执行验证 |

首次练习可以边查边写；第二次去掉实现只看函数签名；第三次限时并口述每个关键选择。最终检查标准是你能解释并修改代码，而不是把所有源文件背下来。

<!-- exercise:12:end -->
