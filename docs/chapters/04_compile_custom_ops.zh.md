# torch.compile 与自定义算子接入

本章使用 `examples/compile_demo.py` 和 `gpu_prep/custom_ops.py`，采用在 PyTorch 2.5 已有的 `torch.library.custom_op`、`register_fake`、`register_autograd`、`opcheck` 接口。提供 CUDA C++ 与 Triton 两条 GPU 接入路径；两者都有 CPU 数学参考，以便没有 GPU 时检查注册和梯度。CPU fallback 不代表 GPU kernel 已经通过运行验证。

<!-- exercise:10:start -->
## 第 10 题：解释 graph break，修复数据依赖分支，并公平评估编译

### 原始题目

分析这段函数，说明调用 `torch.compile` 后可能发生什么：

```python
def bad_python_branch(x):
    if x.sum().item() > 0:
        return x.sin()
    return x.cos()
```

然后实现能处理正、负输入的可捕获版本；分别测量首次调用和稳态；解释一次 `fullgraph=True` 成功能证明什么。

本包的 CPU 检查命令是：

```bash
python -m examples.compile_demo --device cpu --backend eager --include-cond
```

GPU 上实际使用 Inductor 的实验命令是：

```bash
python -m examples.compile_demo --device cuda --backend inductor --include-cond
```

后者需要匹配的 PyTorch/CUDA/Triton 与可用 GPU；如果也选 CUDA C++ custom op，还需要相应 extension 编译工具链。当前生成环境完成的是 CPU capture/数学检查，GPU 速度需在你的机器上实际测量。

### 把编译链路说到足够准确

`torch.compile` 是用户入口。TorchDynamo 分析 Python 执行，抽取可捕获的 tensor 运算为 FX 图，并生成保证图适用的条件检查，也就是 guards。后端收到图之后再决定如何实现计算。

默认后端 Inductor 进行图级转换、调度和代码生成；在 GPU 路径中会利用 Triton 或已有库调用等实现。训练场景还涉及 AOTAutograd 对前向、反向及其保存状态的处理。只做推理、没有 backward 的实验，不应叙述成“实际跑过了训练反向编译”。官方 [torch.compiler 介绍](https://docs.pytorch.org/docs/stable/torch.compiler.html)与 [编译 FAQ](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_faq.html)区分了这些部分。

`backend="eager"` 则让捕获后的图仍以 eager 运算执行。它特别适合先分离问题：“Dynamo 是否能表示这段代码？”如果能，再检查后端代码生成和数值。它的计时不能拿来证明 Inductor 优化，也不能根据 CPU 上通过了测试就声称产生了正确的 GPU kernel。

### 原始函数为什么容易 graph break？

`x.sum()` 是 tensor 计算；`.item()` 把结果变成 Python 标量；`if` 又用这个数据值决定走哪条 Python 分支。该决定取决于本次输入内容，不能简单把第一组样本选中的分支永久固定。

默认允许 graph break 时，编译器可能捕获前一段运算、回到 Python 执行分支，再尝试捕获后续片段。使用 `fullgraph=True` 时，这类无法捕获的部分通常会直接暴露为错误，便于诊断。更改 scalar capture 配置有时可以把 `.item()` 本身纳入图，但不等于自动解决所有数据依赖 Python 控制流。[Common Graph Breaks](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/programming_model.common_graph_breaks.html)

如果 x 在 GPU 上，读取其值以驱动 CPU Python 逻辑还可能要求等待 GPU 结果。这与编译问题相关，但概念上不同：一个是图能否捕获，一个是 host/device 执行依赖是否造成同步。

不要把所有 `if` 都删掉。按 Tensor shape 或固定配置分支，可能通过符号形状或 guard 特化处理；按实际 tensor 数据分支，则需要合适的图内控制流或算法改写。应分析分支条件依赖什么。

### 修复方案一：`torch.where`

```python
def tensor_where(x):
    return torch.where(x.sum() > 0, x.sin(), x.cos())
```

条件保留为标量布尔 Tensor，并广播到输出形状。对这个例子，正输入选择 sin，非正输入选择 cos，与原函数语义一致。测试必须覆盖两个方向；只用全正输入通过，不足以说明对另一个分支也正确。

重要细节：Python 会先求值 `x.sin()` 和 `x.cos()`，再把两份结果交给 `where`。所以它是选择结果，不是“必然只执行被选中的表达式”。如果某个分支包含非常昂贵的计算、不可定义的数值操作或副作用，不能不分析就机械改成 `where`。

例如写 `torch.where(x != 0, 1/x, 0)` 时，不能据此认为 `1/x` 在零位置从未被求值；涉及梯度和异常数值时仍需单独验证。

### 修复方案二：`torch.cond`

```python
def tensor_cond(x):
    return torch.cond(
        x.sum() > 0,
        lambda t: t.sin(),
        lambda t: t.cos(),
        (x,),
    )
```

这是把条件执行明确表示为可捕获的结构化操作。分支要遵循该 API 的限制，例如接收约定一致的 operands，并返回匹配 shape/dtype 等元数据的结果；不能任意修改输入和全局状态。具体支持范围随 PyTorch 版本演进。本包的简单 sin/cos 例子已在 CPU PyTorch 2.5.1 的 capture 路径检查；不借此推断复杂分支或所有训练情况都支持。[官方 Control Flow - Cond](https://docs.pytorch.org/docs/main/cond.html)

回答方案选择时可以说：“计算轻、两边都定义良好时，where 简单；需要保留真实条件计算时，优先考虑 cond，并验证目标版本的限制。若必须执行外部 Python 逻辑，也可以明确接受局部 eager 区域，判断它是否处在性能关键路径。”

### graph break、重新编译、CUDA Graph 分别是什么？

| 概念 | 发生的层次 | 一个例子 | 应检查什么 |
|---|---|---|---|
| Graph break | 捕获 Python/tensor 程序时分段 | 数据值驱动的 Python if | break 原因、片段数量、是否处于热点循环 |
| Recompilation | 已有图的适用条件不满足 | shape、stride、dtype、配置等变化触发 guard 失效 | guard 日志、输入分布、缓存变体数量 |
| CUDA Graph capture/replay | GPU 工作提交和依赖重放 | 重复运行较稳定的 GPU 工作序列 | 地址/生命周期、捕获限制、额外内存、实际 host launch 开销 |

一个程序可以没有 graph break，却因为输入 shape 不断变化而反复编译。另一个程序也可能有几个可接受的图段，但稳态执行仍有明显加速。`torch.compile` 与 CUDA Graph 不是同一个开关；某些编译模式可能配合使用它们，但应分别解释它们解决的问题。

`fullgraph=True` 保证的是这次路径可被捕获为单个图，或明确报错。它不保证输出只由一个 GPU kernel 计算，也不保证无重新编译，更不保证性能必然提高。[fullgraph 官方说明](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.fullgraph_true.html)

### 实际排查顺序

先运行 eager，确认错误不是普通 shape/device/算法错误。然后缩小为可复现函数，用 `fullgraph=True` 暴露第一个 break。必要时打开日志：

```bash
TORCH_LOGS="graph_breaks,recompiles" python -m examples.compile_demo --device cpu --backend eager
```

解决一个 break 后再次检查；但不要把“零 break”作为所有任务的唯一优化目标。如果剩余 Python 区域不在热点，继续改写可能增加复杂度而没有收益。进入后端问题时，分别判断是编译失败、数值差异还是实际变慢，三者需要不同证据。

### 首次调用与稳态：为什么不能混成一个数字？

创建 `compiled=torch.compile(fn)` 通常只是建立包装，许多工作会在第一次实际调用时发生，包括图捕获、编译、autotuning、缓存初始化等。第一次执行有较大开销，并不直接代表之后每次运行的开销。

`compile_demo.py` 单独报告首次 compiled 调用时间，再 warm-up，最后报告多次稳态 wall time 中位数。在 CUDA 情况下计时前后同步，避免只测到 CPU 发出命令就结束；这是一种端到端调用计时，包含相应 host 和同步开销。专门测 kernel 时间时，还应使用 CUDA Events，并清楚说明是否包含分配和传输。

这里的“首次调用”不自动等同于完全冷启动：运行时可能已有 GPU context，磁盘可能已有编译缓存。若要报告服务冷启动，应明确是否包含进程启动、模型加载、context、编译缓存和首个请求。所有这些状态都会改变测量含义。

比较时固定：同一模型、输入、dtype、训练/推理模式、数值精度选项、shape 分布、计时范围与硬件负载。不能让基线包含数据搬运、编译和分配，而优化版本只测已驻留的 kernel；也不能在 warm-up 后只测一个特别适合优化版本的 shape。

### “编译值得吗”还可以算一个回本点

设额外编译成本为 C，eager 单次延迟为 `t_e`，compiled 稳态为 `t_c`，且 `t_e>t_c`。需要大约：

\[
N>\frac{C}{t_e-t_c}
\]

次调用才能在累计时间上抵消编译开销。这个式子假设调用形状能够复用同一编译结果；频繁重新编译或在线请求对首个响应极敏感时，结论会改变。回答时要把工作负载和服务目标放进去，而不只是宣称“steady state 更快”。

### 面试追问：可直接复述的回答

**问：`torch.compile` 的优化主要来自哪里？**

答：“一部分来自看到多步运算后做融合、减少中间读写与调度；另一部分来自对具体形状、布局和后端实现进行选择或专门化。收益取决于工作负载。如果已经是大型优化 GEMM 主导，或被 Python 数据依赖频繁打断，改善可能有限。”

**问：为什么会重新编译？我不是已经 compile 过了吗？**

答：“编译结果有适用条件。新输入的形状、布局、类型或相关 Python 状态不满足 guard 时，已有版本不能直接复用。我要先看触发的是哪些条件，再决定使用动态形状、按常见长度分桶、移动可变 Python 状态，还是接受少量专门化版本。”

**问：用了 dynamic shapes 就不会重编译了吗？**

答：“不能这样保证。动态形状扩大了一些维度的适用范围，但其他 guard、结构变化和不支持的路径仍可能触发重新编译。过度动态化还可能损失专门化优化。应针对实际请求分布做选择。”

**问：CPU `backend='eager'` 通过了，下一步是什么？**

答：“它说明这个输入范围内的 Dynamo 捕获和参考数学能成立。接下来检查实际目标后端的编译、前向与反向数值，再测目标 GPU 的 latency、吞吐和显存。不能把前端验证当作 GPU 性能结果。”

**问：看到第一次很慢，是不是直接放弃？**

答：“先拆分首次调用和稳态，检查是否有重复编译。离线重复训练和长时间运行服务可能摊薄一次性成本；短生命周期任务或首请求敏感服务则要把启动成本计入决策。是否预热或预先编译，要基于真实使用方式。”

### 本题的完成标准

能解释给定函数为何难以捕获，写出 where 和 cond 两种等价例子并说明取舍；清楚区分 graph break、guard 重编译、CUDA Graph；知道 eager backend 验证范围；能够拿出首次调用与稳态分开、同条件比较的测量方案。
<!-- exercise:10:end -->

<!-- exercise:11:start -->
## 第 11 题：把 CUDA 与 Triton bias+ReLU 接入 PyTorch 编译和训练

### 题目与交付的两条路径

本题固定一个易理解、容易验证的函数：

\[
y_{m,n}=\max(x_{m,n}+b_n,0).
\]

在 `gpu_prep/custom_ops.py` 中注册了两个独立名字：

| Python 入口 | Dispatcher 名字 | CUDA 张量的实现 | CPU 张量的实现 |
|---|---|---|---|
| `bias_relu(x,bias)` | `gpu_prep::bias_relu` | `gpu_prep.cuda_ops.bias_relu` | `torch.relu(x+bias)` |
| `bias_relu_triton(x,bias)` | `gpu_prep::bias_relu_triton` | `gpu_prep.triton_ops.bias_relu` | `torch.relu(x+bias)` |

它们共用同一 metadata 检查、fake 规则与 backward，但调用不同 GPU 实现。GPU 分支都惰性导入；仅导入 `gpu_prep.custom_ops` 不会编译 C++/CUDA，也不要求 CPU 用户先安装一个可运行的 GPU 编译环境。

### 先明确接口合同

这里支持连续的 `x[M,N]` 和连续的 `bias[N]`；同 dtype、同 device；返回新分配的连续 `y[M,N]`，不修改输入，也不与输入共享输出存储。M 或 N 为零都允许，结果自然为空。

CPU 参考支持 FP16、BF16、FP32、FP64；本练习 GPU 接入层明确限制 FP32，以对齐 CUDA 基础实现。FP64 CPU 路径的一个用途是做数值梯度检查，不是用来代替 GPU 测试。数学 backward 的验证使用有限输入，在 ReLU 的零点采用导数为 0 的约定。

如果想接受非连续输入，有两种清楚的扩展：修改 kernel 按 stride 寻址；或在 wrapper 中先 `.contiguous()`，把复制成本纳入接口和性能测量。本实现选择主动报错，避免底层按连续地址访问而默默算错。

`mutates_args=()` 是对 dispatcher 的正式承诺：输入不会被原地修改。它不只是装饰器的格式要求。编译器、functionalization 和 autograd 会依赖这个信息推理程序行为。[Custom Python Operators](https://docs.pytorch.org/tutorials/advanced/python_custom_ops)介绍了 schema、aliasing 与 mutation 合同。

### 为什么“一个 Python 函数能调用 CUDA”还不够？

普通 Python wrapper 能在 eager 中启动 kernel，只解决了“怎么执行”这一层。框架还需要知道：输出什么 shape/dtype、是否修改输入、没有真实数据时怎么推理、反向怎么计算、编译图中用什么算子节点表示。

`torch.library.custom_op` 给这个计算一个稳定的 dispatcher 身份。在编译器看来，它可以是一个已知输入输出合同的操作，而不是必须继续分析每一行 Python、JIT build 逻辑和底层外部调用。CUDA/C++ 接入的总体组件也可以在 [PyTorch 官方 C++/CUDA custom op 教程](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)中核对。

### Fake kernel：不算答案，只给出正确的输出元数据

代码注册的 fake 实现只做 metadata 检查，并返回：

```python
torch.empty(x.shape, dtype=x.dtype, device=x.device)
```

在 FakeTensor 模式下，这代表一个形状、类型、设备和布局正确的符号输出，不是申请真实 GPU 数据来执行融合运算。fake 函数不能用 `.item()` 读取数据，不能把输入拷到 CPU 检查元素，不能调用实际 CUDA/Triton kernel。

为什么不随便 `empty_like(x)`？在这个合同里输入已要求连续，因此通常也能匹配；但更一般的 custom op 可能总是返回连续输出，而 `empty_like` 可能沿用某种输入布局。fake 必须与真实输出的 stride、aliasing 等行为一致，而不只是 shape 看起来相同。本代码直接声明一个新连续张量，使合同更直观。[Functional Custom Operators](https://docs.pytorch.org/tutorials/advanced/python_custom_ops_functional.html)

### 训练接入：forward-only kernel 怎样有 backward？

底层 CUDA/Triton 练习明确只实现 forward。如果直接拿它们处理需要梯度的输入，会主动拒绝，避免静默丢梯度。custom op 的 GPU 分支在 `torch.no_grad()` 下调用 forward kernel，然后由注册的 autograd 公式负责整个操作的求导。

这不是把训练关掉：`no_grad` 只包住不打算让 PyTorch 逐行追踪的底层执行；外层 dispatcher 仍根据 `register_autograd` 提供的规则建立梯度关系。

设 `z=x+b`，上游梯度为 G。对有限输入并约定 ReLU 在 0 的导数为 0：

\[
dX_{m,n}=G_{m,n}\,\mathbf{1}[z_{m,n}>0],
\]

\[
dB_n=\sum_{m=0}^{M-1}dX_{m,n}.
\]

代码的 setup_context 保存 x 和 bias；backward 用可被 PyTorch 理解的 tensor 操作重算符号 mask，再分别返回 x 与 bias 的梯度。bias 只有一维，但 forward 把它广播到了每一行，反向必须把所有行的贡献相加。M=0 时该和应为零，不能留下未初始化的梯度。

上面的数学公式用指示函数相乘表达导数；实际浮点代码使用 `torch.where(active, grad_output, torch.zeros_like(grad_output))`。原因是无效位置的上游梯度即使为 Inf 或 NaN，也应该像 PyTorch ReLU 一样返回零，而 `0*Inf` 或 `0*NaN` 会生成 NaN。针对 `x=[[-1,0,1]]`、零 bias、上游梯度 `[[Inf,NaN,1]]`，两条 custom op 都有回归测试，确认 x/bias 梯度为 `[0,0,1]`。

另一种实现可以保存输出或激活 mask，避免反向再做 x+b。选择时比较保存状态的显存、重算成本、mask 表示和实现复杂度。这里先用容易读懂的版本。官方 [Adding Training and Other Registrations](https://docs.pytorch.org/tutorials/advanced/python_custom_ops_registrations.html)提供了 `setup_context` 与 `register_autograd` 的接口结构。

### `opcheck` 与 `gradcheck` 分别证明什么？

| 检查 | 本包做了什么 | 能证明什么 | 不能代替什么 |
|---|---|---|---|
| 数值前向比较 | 与 `torch.relu(x+bias)` 比较 | 给定输入的数学结果和输出 metadata 一致 | 未覆盖输入范围、GPU 性能 |
| backward 参考比较 | 比较 x/bias 的梯度 | 给定输入和上游梯度的求导结果一致 | 一般注册合同 |
| `torch.library.opcheck` | 检查 schema、autograd 注册、FakeTensor、AOT dispatch 等 | 框架注册使用符合相关规则 | 梯度公式的数学正确性 |
| `torch.autograd.gradcheck` | 在小 FP64 输入上做有限差分 | 被检查输入附近的解析梯度与数值梯度一致 | GPU dtype/stream/内存错误 |
| `torch.compile(fullgraph=True,backend='eager')` | 捕获调用 custom op 的图并比较结果/梯度 | 该路径可被 Dynamo 捕获并正确执行 | Inductor codegen、GPU 运行或加速 |

两个注册入口都跑了这些 CPU 检查。选择数值梯度输入时，刻意让 `x+bias` 远离 0：ReLU 在 0 不可导，中心差分与我们指定的导数 0 不一致，可能产生假失败。零点的行为通过另外一条离散断言检查，而不强迫有限差分在不可导点通过。

不能这样回答：“opcheck 全通过，所以我的 backward 肯定正确。”正确回答是：“opcheck 验证接入规则；我另外用 gradcheck 和参考梯度比较验证公式。”

### CUDA 接入为什么必须处理 device 与 current stream？

Python wrapper 已确认两个输入在同一设备，但底层 launch 还必须切到输入设备对应的上下文，并使用 PyTorch 当前 stream。否则在用户切换设备或采用非默认 stream 时，kernel 可能在错误设备访问指针，或忽略与前后操作的执行依赖。

面试时可以回答：“我不会把 stream 0 写死。输入 tensor 的 device 决定设备 guard，PyTorch 当前 stream 决定 launch stream。kernel 保持异步；调试或 benchmark 的同步放在调用方合适的位置，而不是每个 forward 内无条件 `cudaDeviceSynchronize()`。”

跨 stream 使用资源还要考虑生产者/消费者依赖与内存生命周期。一个算子在默认 stream 的随机输入测试通过，不能证明并发情况下正确。PyTorch 的 [CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda)说明了 stream 顺序、跨 stream 同步和内存使用注意事项。

Triton 的 kernel launch 也应遵守当前 PyTorch CUDA 执行环境；自建 stream 并不会自动变快。两个路径都需要在 GPU 上用独立的非默认 stream 测试和实际 profiler 验证，CPU 注册测试覆盖不到这里。

### custom op 与编译融合：边界本身也有成本

`custom_op` 在本基线中是一个对编译器不透明的调用。fake 让编译器知道结果的元数据，autograd 注册让它知道反向规则，但它并不会因此自动看见 CUDA/Triton kernel 内部每个算子。

因此有两种不同层次的融合：kernel 内部的 `x+bias+relu` 已经融合；它与外部残差加法、norm 或其他操作是否还能融合，是另一回事。如果把本来可被编译器完整看到的一小段纯 PyTorch 运算强行包成 opaque custom op，反而可能减少优化机会。

本练习为了练接入、调用外部 CUDA/Triton 实现，明确使用 custom op。真实项目若只是 `torch.relu(x+bias)`，通常先让编译器处理原始组合更简单。是否需要 custom op，应由外部 kernel、硬件能力或明确接口需求驱动。

### Triton 的另一种较新接入方式

PyTorch 2.6 增加了 `torch.library.triton_op`。与本包使用的 opaque `custom_op` 相比，它允许编译器进入 Triton-backed 实现，并通过 `torch.library.wrap_triton` 表达 kernel 调用，同时继续提供框架注册能力。官方 [Using User-Defined Triton Kernels with torch.compile](https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html)列出了两种接入方式的差异。

本包以实际验证过的 PyTorch 2.5.1 为基线，所以可运行代码选择 `custom_op`，没有直接导入较新的 API。升级时可以单独试验 `triton_op` 路径，并重新检查 CPU fallback、fake/autograd 行为及目标版本限制。不能把 `triton_op` 文档中的可见性优点自动归给本包的 2.5 custom-op wrapper。

一个面试回答是：“如果只需要一个 Triton 函数在已支持的 compile 场景运行，直接集成可能更简单；如果还需要统一 dispatcher 身份、CPU fallback 和其他子系统行为，再选择适当的注册 API。基于版本和需求决定，不会只因为名字更新就替换。”

### 怎样在自己的 GPU 上进一步验收？

先运行底层 CUDA/Triton 的数值测试，确认 launch、尾部和 NaN 等约定。再运行本题 custom op 的 GPU 测试，使外层注册、底层实现和 backward 连起来。最后做实际 Inductor 的 forward/backward 数值和性能实验。

可以分别选择一个实现，避免把 CUDA extension 的首次编译与 Triton 首次 JIT 混在同一个性能比较中：

```bash
python -m examples.compile_demo --device cuda --backend inductor --operator-backend cuda
python -m examples.compile_demo --device cuda --backend inductor --operator-backend triton
```

脚本在 capture 实验之前先调用一次底层路径，触发该路径所需的初始化。首次构建 extension、Triton JIT、Dynamo/Inductor 编译和 kernel 稳态时间仍是不同指标。如果要比较纯 kernel 延迟，应使用包中专门的 benchmark，并统一是否包括 wrapper、分配与同步。

### 面试追问：可直接复述的回答

**问：为什么 fake kernel 不能直接算一次真实结果？**

答：“编译阶段可能只有符号形状和 fake tensor，没有实际数据。fake 的职责是描述输出元数据。读取元素会把它变成数据依赖，还可能启动不应该发生的 GPU 工作。我只用输入的 shape、dtype、device、stride 和 schema 来描述输出。”

**问：为什么你不直接返回输入 x，当作 ReLU 的输出？**

答：“这违反了我声明的 fresh functional 输出合同，也改变了 aliasing。如果要设计原地算子，就应明确声明修改哪个参数，并解决 autograd 和 functionalization 的对应规则。不能在 `mutates_args=()` 下偷偷原地写入。”

**问：为什么 CPU 测试使用 double，CUDA 用 float？**

答：“double 的 CPU fallback 用来检查数学梯度；基础 CUDA kernel 只实现了 float32，因此接口明确限制。我要分别证明注册和公式正确、GPU实现正确，再根据需要扩展 dtype。用 double CPU gradcheck 并不能证明 half/float 的 GPU 数值表现。”

**问：注册了 autograd，就支持所有训练方式了吗？**

答：“它支持我们给出的 VJP 和被测试的训练路径。AMP、vmap、forward-mode、二阶求导、export 或其他后端如果是需求，要分别确认和注册。这里 backward 由 PyTorch 表达式组成便于组合，但我不会据此默认所有子系统都已验证。”

**问：bias 梯度为什么是 sum，不是 mean？**

答：“链式法则要求把每个使用点对同一个参数的贡献累加。loss 是否取 mean 已经反映在上游梯度 G 中；在 backward 里再无条件取 mean 会重复缩放，产生错误。”

**问：自定义 kernel 更快，整体为什么可能没更快？**

答：“要检查它是否减少了实际热点时间，以及是否引入 contiguous 拷贝、额外分配、同步或打断周围融合。custom op 边界是实现能力，也可能带来优化限制。我会同时测底层 kernel、算子调用和实际模型路径。”

**问：如果面试只让我接一个算子，最少需要交代什么？**

答：“先说明 shape/device/dtype/stride 和是否修改输入；给出底层调用与正确 stream；补 fake 元数据；如果要求训练，再给 backward 和广播归约；最后分别用前向比较、梯度检查和 opcheck 验证。每项的验证范围说清楚。”

### 本题的完成标准

能够解释并运行两条注册入口；知道导入阶段与首次执行的区别；理解 fake 的 metadata 合同；推导 bias+ReLU 的 backward；区分 opcheck 和 gradcheck；解释 current stream、opaque boundary 和版本相关 API；对没有 GPU 的环境只报告实际完成的 CPU 验证。
<!-- exercise:11:end -->
