# PyTorch 层：从形状、数值到训练与 KV cache

本章的实现位于 `gpu_prep/torch_layers.py`；演示位于 `examples/torch_layers_demo.py`；有独立参考实现、梯度和缓存等价性检查的测试位于 `tests/test_torch_layers.py`。这些是可读、可验证的教学实现。它们使用 PyTorch 已有算子组成计算图，所以可在 CPU 上检查数学与 autograd，在 GPU 上执行；它们不会自动等同于一个融合 CUDA kernel。

<!-- exercise:7:start -->
## 第 7 题：手写 Linear，正确拆分 Q／K／V

### 题目与边界

实现 `ManualLinear(in_features, out_features, bias=True)`，不能在实现内部调用 `nn.Linear` 或 `F.linear`。输入形状可以是 `[D_in]`、`[B,D_in]`、`[B,T,D_in]`，统一得到末维为 `D_out` 的输出。使用真正的可训练参数，保留自动求导。外层 batch 或 sequence 维可以为零；本实现要求特征维为正。

第二部分：给定一次合并投影产生的 `qkv[B,T,3D]`，把它拆成三个 `[B,H,T,d]` 张量，满足 `D=H*d`。明确约定投影末维依次排列全部 Q 特征、全部 K 特征、全部 V 特征，不要在没有说明布局的情况下任意 reshape。

### 先把矩阵式讲清楚

对一个输入向量，输出第 j 项是：

\[
y_j=\sum_{i=0}^{D_{in}-1}x_i W_{ji}+b_j.
\]

因此 `weight` 的形状选择为 `[D_out,D_in]`，输入的最后一维与 `weight.T` 的倒数第二维相乘：

```python
out = x @ self.weight.transpose(0, 1)
if self.bias is not None:
    out = out + self.bias
```

`[B,T,D_in] @ [D_in,D_out]` 得到 `[B,T,D_out]`。`bias[D_out]` 从右侧对齐维度，相当于对所有 batch、所有 token 加同一组偏置。没有必要先把输入手工改成二维再逐行循环。

代码把 `weight` 和 `bias` 声明为 `nn.Parameter`，并挂在 `nn.Module` 属性上。这样优化器可以通过 `model.parameters()` 找到它们。普通 Tensor 属性不会因为“看起来像权重”自动完成参数注册。[PyTorch Parameter 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html)说明了这一注册行为。

### 一个贯穿全题的形状例子

取 `B=2，T=5，D=8，H=2，d=4`：

| 步骤 | 形状 | 这一步到底改变了什么 |
|---|---|---|
| 输入 X | `[2,5,8]` | 2 个序列，每个 5 个 token，每 token 8 个特征 |
| 合并投影 | `[2,5,24]` | 每个 token 产生 Q、K、V 各 8 个特征 |
| `reshape(B,T,3,H,d)` | `[2,5,3,2,4]` | 将末维显式拆成“QKV 类型、head、head 内特征” |
| `permute(2,0,3,1,4)` | `[3,2,2,5,4]` | 把 QKV 类型移动到最前，把 head 放到 sequence 前面 |
| `unbind(0)` | 3 个 `[2,2,5,4]` | 分别选出 Q、K、V |

不能直接把 `[B,T,3D]` 改成 `[B,H,T,3d]` 然后声称已完成多头拆分。这通常改变了维度解释，并未按照正确的 token/head 分组完成重排。一个很好的测试办法是使用递增整数，让每个 Q/K/V 区段的值可辨认，而不只检查最终 shape。

### `view`、`reshape`、`permute`、`contiguous` 为什么容易出错？

Tensor 的“形状”不是全部信息。访问元素时，还要用 stride 决定每个维度前进一步，需要在底层存储中移动多少个元素。

例如连续的 `[2,3]` 张量，其 stride 是 `(3,1)`；转置后变成 `[3,2]`，stride 是 `(1,3)`。转置通常只改变索引规则，并没有把数据按新行序复制一遍。此时想按新形状展平，可能无法继续只靠一个视图表达。

| 操作 | 语义 | 是否保证不复制 |
|---|---|---|
| `permute` / `transpose` | 重排维度及对应 stride | 返回视图；不承诺结果连续 |
| `view` | 在当前存储布局允许的前提下重新解释 shape | 不复制；布局不兼容就报错 |
| `reshape` | 得到指定 shape | 尽量返回视图，必要时复制 |
| `contiguous` | 得到指定连续布局 | 已经满足时可以直接返回；否则复制 |

这里的代码在合并 attention heads 时使用：

```python
attended.transpose(1, 2).reshape(B, T, D)
```

它先把 `[B,H,T,d]` 改成 `[B,T,H,d]`，再把 `H,d` 合并。`reshape` 让语义正确；性能分析时还需要确认它是否产生了复制。不要把“写起来没有报错”解释成“没有额外访存”。官方 [Tensor.view 文档](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.view.html)给出了按 shape/stride 判断能否合并维度的条件；[Tensor Views](https://docs.pytorch.org/docs/main/tensor_view.html)解释了视图共享存储。

还有两个常见混淆：`clone()` 是数据复制，但并不会自动切断 autograd；`detach()` 切断梯度关系，但通常仍共享存储。为了得到一份独立的测试输入，测试里使用 `x.detach().clone().requires_grad_()`。

### Linear 的反向传播，怎样口头推导？

把所有前导维暂时合成 `R` 行，写成 `Y=XW^T+b`。设上游梯度为 `G=dL/dY`，则：

\[
\frac{\partial L}{\partial X}=GW,
\qquad
\frac{\partial L}{\partial W}=G^T X,
\qquad
\frac{\partial L}{\partial b}=\sum_{r=0}^{R-1}G_{r,:}.
\]

可以这样解释偏置梯度：“同一个 bias 元素被广播给所有样本位置。每个位置对它的贡献都要累加，所以沿所有广播出来的前导维求和，最后只保留 `D_out`。”

代码没有另写 backward，因为 `matmul`、转置与加法已经有 autograd 规则。考试要求手写层，不一定要求手写 `autograd.Function`；先确认要求，再选择最小而完整的实现。

### 面试追问：可直接复述的回答

**问：为什么不用普通的 `torch.randn` 作为 `self.weight`？**

答：“计算本身可以用普通 Tensor，但如果它只是 Module 的普通属性，优化器通过 `parameters()` 不一定能发现它。我的实现用 `nn.Parameter` 明确登记训练状态，权重初始化在初始化阶段完成，forward 不重新生成权重。”

**问：你的代码能处理不同 batch size 吗？**

答：“可以。矩阵乘法只要求最后一维与输入特征数匹配，前面的维度代表独立位置。我不把 B 和 T 写死，并且测试了单向量、多维输入和外层为空的情况。特征维为零不在本实现的约定内，会明确报错。”

**问：为什么合并 QKV 投影可能更快？**

答：“三个线性映射数学上仍独立。合并后可以将三次较小的 GEMM 变成一次更大的 GEMM，减少调度，并可能改善矩阵尺寸和输入复用。但还要考虑拆分后的布局是否导致复制，不能只根据调用次数保证加速。”

**问：调用 `model.eval()` 是否会停止计算梯度？**

答：“不会。`eval()` 改变依赖训练状态的层行为，例如 dropout；是否记录梯度要用 `no_grad()`、`inference_mode()` 或相应的 requires_grad 设置。这两个维度应分别处理。”

**问：非连续输入是否必须先 `.contiguous()`？**

答：“纯 PyTorch 运算通常能处理 stride，因此这个参考实现不先强制复制。自写 CUDA 如果只按连续下标访问，就必须声明只接受连续输入，或者显式处理 stride。是否复制应当作为接口和性能成本的一部分说明。”

### 验收标准

实现能够与 `F.linear` 对比前向和输入/参数梯度；QKV 数值位置而非仅 shape 正确；能够解释 bias 梯度和 reshape 复制；导入模块不会创建 GPU 资源或编译 CUDA。对应测试已经写在 `test_torch_layers.py` 中。
<!-- exercise:7:end -->

<!-- exercise:8:start -->
## 第 8 题：LayerNorm 与 RMSNorm，包括完整反向推导

### 题目与实现位置

本题的类为 `ManualLayerNorm` 和 `ManualRMSNorm`。要求沿 `normalized_shape` 指定的末尾维度做归一化，显式使用 `eps=1e-5`，支持可训练缩放；LayerNorm 可加偏置。实现内部不调用现成的 norm 层。

FP16/BF16 输入的统计量在 FP32 中计算，最终恢复输入 dtype；FP64 输入保留 FP64，供数值梯度检查使用。不能无条件 `.float()`，否则 double 输入也被降成 FP32，有限差分的精度基础会丢失。特征归约维必须为正，外层没有样本的张量可以自然返回空结果。

epsilon 还必须能用实际累加 dtype 表示为正有限值。例如 Python 中 `1e-300` 为正，在 FP32 中却会舍入为零，零输入就可能重新产生除零问题；在 FP64 中这个值仍然合法。两种 Norm 在 forward 按累加 dtype 检查这一点，使用与 FP32 转换等价的纯 host 边界比较，不创建 GPU 标量或调用 `.item()`。它没有把“可表示”错误地收紧为“必须是正规数”；正的可表示 subnormal 仍可通过。

### 两个公式放在一起看

把一次归一化的所有特征展成长度 D 的向量，LayerNorm 是：

\[
\mu=\frac1D\sum_i x_i,
\quad
v=\frac1D\sum_i(x_i-\mu)^2,
\quad
r=(v+\epsilon)^{-1/2},
\]

\[
z_i=(x_i-\mu)r,
\qquad y_i=\gamma_i z_i+\beta_i.
\]

RMSNorm 是：

\[
m_2=\frac1D\sum_i x_i^2,
\quad r=(m_2+\epsilon)^{-1/2},
\quad z_i=x_i r,
\quad y_i=\gamma_i z_i.
\]

区别不只是少写一行代码：LayerNorm 先消除均值，再控制尺度；RMSNorm 用均方根控制尺度，不要求输出均值为零。这两种模型组件不能在未经验证时互换。PyTorch 的 [LayerNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)采用分母 D 对应的总体方差，即 `correction=0`；[RMSNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html)的默认 epsilon 规则可能不同，因此参考比较时要显式传相同的 `eps`。

### 用一个小向量算一次

设 `x=[1,2,3]`、缩放全为 1、偏置为 0，暂时忽略很小的 epsilon：

- LayerNorm：均值 2，方差 `2/3`，输出约 `[-1.225,0,1.225]`。
- RMSNorm：均方为 `14/3`，输出约 `[0.463,0.926,1.389]`。

后一个输出并不以零为中心，这是公式的预期行为。加入 epsilon 后，未经过 affine 的输出方差或均方也不严格等于 1。例如 LayerNorm 的归一化方差为 `v/(v+eps)`。测试若机械断言方差“精确等于 1”，会把正确实现误判为错误。

### 为什么减均值后再求平方？

可以把方差写成 `E[x^2]-E[x]^2`，但当 x 的均值很大、波动很小时，这是两个相近大数相减，可能损失精度。参考实现使用两遍思路：先求均值，再求偏差的平方均值，容易解释和验证。

进一步优化时可以考虑 Welford 形式，它把统计量表示成样本数、均值和偏差平方和，支持稳定地组合部分结果。面试时不能只说“Welford 更好”，应指出它在并行归约中的组合能力，以及指令数、数值稳定性和数据读取次数之间的取舍。

### `normalized_shape=(3,4)` 到底是什么意思？

输入 `[B,3,4]`，指定 `(3,4)` 表示每个样本的 12 个元素共同计算一组统计量，gamma/beta 的形状也为 `[3,4]`。指定 `4` 则表示每个 `[4]` 行分别归一化，一共有 `B*3` 组统计量。

代码使用末尾的多个轴归约，并保留 `keepdim=True`。保留长度为 1 的维度后，均值或 reciprocal standard deviation 可以按原位置广播，避免减法误对齐。其数学仍可按“每行 D 个特征”推导，D 是归一化维度大小的乘积。

### LayerNorm 反向：从中间量推到可实现的公式

设 `g_i=dL/dy_i`，定义经过缩放参数后的梯度：

\[
a_i=g_i\gamma_i.
\]

前向归一化是 `z_i=(x_i-mu)r`。反向有三条依赖：x 直接影响自己的偏差，x 影响均值，x 也通过方差影响所有元素使用的 r。合并这些贡献，可以得到：

\[
\frac{\partial L}{\partial x_i}
=r\left[a_i-\operatorname{mean}(a)-z_i\operatorname{mean}(a z)\right].
\]

这个式子包含两次行内归约：`mean(a)` 和 `mean(a*z)`。不能把后两项省掉，只写 `dx=dy*gamma*r`，因为均值和方差也依赖 x。

推导的中间步骤是：

1. `dz_i = a_i`。
2. `dr = sum_i a_i * (x_i-mu)`。
3. 因为 `r=(v+eps)^(-1/2)`，所以 `dv=-0.5*r^3*dr`。
4. `v=mean((x-mu)^2)` 对每个偏差贡献 `2*(x_i-mu)/D`。
5. 再合并 `mu=mean(x)` 带来的所有路径，得到上面的紧凑公式。

参数梯度是对所有外层样本位置求和：

\[
\frac{\partial L}{\partial\gamma_i}=\sum_{rows}g_i z_i,
\qquad
\frac{\partial L}{\partial\beta_i}=\sum_{rows}g_i.
\]

注意它们的归约方向与输入梯度不同：`dx` 需要一行内部的统计量；`d_gamma/d_beta` 需要跨行累积。真正写 GPU backward 时，这种差别会影响并行划分、临时缓冲和是否需要两阶段归约。

### RMSNorm 反向：少了哪一项？

仍令 `a_i=g_i*gamma_i`、`r=(mean(x^2)+eps)^(-1/2)`、`z_i=x_i*r`。因为没有均值中心化，不再需要 `-mean(a)`：

\[
\frac{\partial L}{\partial x_i}
=r a_i-r^3x_i\operatorname{mean}(a x)
=r\left[a_i-z_i\operatorname{mean}(a z)\right].
\]

参数梯度仍是：

\[
\frac{\partial L}{\partial\gamma_i}=\sum_{rows}g_i z_i.
\]

这也是一个实用的记忆方法：先理解“均值中心化”引入了什么依赖，再理解 RMSNorm 为什么没有对应项。不要把两个 backward 公式混着背。

### 如何验证梯度，而不被 ReLU 或低精度干扰？

本题本身在正 epsilon 下是光滑的，适合对很小的 double 输入做 `gradcheck`。测试使用 `torch.func.functional_call` 把 gamma/beta 也作为显式输入，因此可以检查输入和参数的数学导数。普通的前向比较则使用 PyTorch norm API 作为独立参考。

有限差分大致使用 `[f(x+h)-f(x-h)]/(2h)`。h 太大会有截断误差，太小会有舍入误差，所以先在 FP64、小尺寸下检查公式，再在目标 GPU dtype 下检查允许误差。PyTorch 的 [gradcheck 文档](https://docs.pytorch.org/docs/main/generated/torch.autograd.gradcheck.gradcheck.html)明确提醒，默认容差针对 double，且不可导点可能使数值与解析梯度不一致。

### 优化思路应如何回答？

**问：为什么值得融合 norm？**

答：“参考写法包含求均值、减均值、平方、归约、rsqrt、scale/bias 等多个逻辑操作。Eager 可能为中间结果分配张量并启动多个 kernel。融合可以把一行的数据与中间统计量尽量留在寄存器或共享存储，减少 HBM 读写和启动次数。具体是否只读一遍，取决于一行能否驻留以及实现方式。”

**问：每行一个 block 总是最好吗？**

答：“这是便于理解的起点。行很短时，一个 block 可能资源利用不足，可以一块处理多行；行很长时，寄存器需求和归约长度增加，可能需要分段归约或多阶段计算。行数很少时还要考虑并行度不足。我会按行长度和行数选择不同方案，而不是只调一个 block size。”

**问：能用原子操作累加 gamma 的梯度吗？**

答：“可以作为简单版本，因为多行会写同一 gamma 位置。但要考虑竞争、累加顺序和可重复性。另一种方案是先让不同 block 写各自的部分和，再做第二次归约。哪种更合适，要结合行数、特征数和精度要求测量。”

**问：RMSNorm 是否一定比 LayerNorm 快？**

答：“它减少了中心化相关工作，算法上更简单。但实际速度还受是否已经融合、内存带宽、launch 和形状影响。不能从少一次 mean 就直接承诺端到端加速，更不能不经模型验证就替换归一化类型。”

**问：为什么把 epsilon 放到平方根里面？**

答：“这是我们定义和参考实现的公式：`rsqrt(statistic+eps)`。改成 `1/(sqrt(statistic)+eps)` 是另一个函数，尤其在统计量很小时前向和梯度都不同。epsilon 也不应只在一端默认，让比较的两边采用不同数值。”

**问：能不能把所有张量都转成 FP32 保证稳定？**

答：“低精度输入的统计量升为 FP32 是合理起点，但不应无条件把 double 降成 FP32。另一方面，FP32 也不是对所有极端输入都不会溢出。还需要声明支持的数值范围，并单独检查输出误差和模型效果。”

### 本题的完成标准

你应能在没有参考代码时写出两种 forward；知道方差分母与 epsilon 位置；能画出归约轴；能推导并解释两种 `dx` 公式；能说出 GPU norm 的并行与融合选择。代码中的 autograd 与测试用于验证你的推导，不能代替你解释它。
<!-- exercise:8:end -->

<!-- exercise:9:start -->
## 第 9 题：手写多头 causal attention，并加入可靠的 KV cache

### 接口先讲清楚，后面才不会混淆 mask

底层函数 `scaled_dot_product_attention` 使用：

| 参数 | 形状／约定 |
|---|---|
| q | `[B,H,L,d]`，L 是本次 query 数 |
| k | `[B,H,S,d]`，S 是可访问 key 数 |
| v | `[B,H,S,dv]`，value 的特征维可以与 d 不同 |
| attn_mask | 布尔张量，可广播到 `[B,H,L,S]`，`True` 表示允许访问 |
| is_causal | 为真时再与 causal 条件相交 |
| query_start | 本次 query 第 0 项的绝对位置；key 位置从 0 开始 |
| dropout_p / training | 仅在 training 为真且概率非零时应用 attention dropout |

同批次、同 head 数的 q/k/v 必须具有相同 dtype 和 device。本实现不隐式广播 batch/head，也没有把 GQA 藏进接口。padding 常见原始形状是 `[B,S]`，必须显式变成 `padding[:,None,None,:]`，否则一个二维 mask 会从最右侧对齐，可能误把 batch 当 query 维。

低精度 q/k/v 的 attention 参考计算转到 FP32；矩阵乘法处还显式关闭外层 autocast，避免“已经 `.float()`，又被 AMP 自动降回低精度”的情况。最后输出转回 q 的 dtype，返回的 probabilities 保留累加 dtype。这里的 FP32 tensor 规则与 GPU 是否采用 TF32 等底层精度选项仍是不同问题，性能对比时应固定相关设置。

PyTorch SDPA 也使用“布尔 True 允许参加 attention”的约定，但 `nn.MultiheadAttention` 的 `key_padding_mask` 使用相反语义；非方阵 `is_causal=True` 的默认对齐也未必等于你想要的 decode 偏移。因此测试使用显式 mask 对照，而不依赖默认行为。[官方 SDPA 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)

### 完整 forward 从哪几步组成？

`MultiHeadSelfAttention` 的 forward 有四步：

1. `X[B,T,D]` 经合并 Linear 得到 `[B,T,3D]`。
2. 拆出 Q/K/V，形状均为 `[B,H,T,d]`。
3. 计算每个 head 的 attention，得到 `[B,H,T,d]`。
4. 转回 `[B,T,H,d]`，合并 heads 为 `[B,T,D]`，再做输出投影。

对一个 head：

\[
A=QK^T/\sqrt d,
\quad P=\operatorname{softmax}(A+mask),
\quad O=PV.
\]

`QK^T` 的输出 `[L,S]` 中，第 i 行是 query i 对全部 keys 的分数，所以 softmax 必须沿最后一维 S。把 softmax 放在 L 维会变成“不同 query 争夺一个 key”的分配，语义已经改变。

缩放 `1/sqrt(d)` 的常见解释是假设 Q/K 分量独立且具有单位量级方差，其点积方差随 d 增长，除以 sqrt(d) 可控制 logits 尺度，减轻 softmax 过早饱和。这是设计动机，不是对所有实际训练分布的严格统计保证。

### 数值稳定：先减最大值，再处理“整行不可访问”

对至少有一个有效且有限 logit 的行，先求有效位置最大值 m，再计算 `exp(score-m)`。最大的指数为 1，其他有效指数不超过 1，避免直接对很大的正分数取指数。

但如果整行全部被屏蔽，直接填成 `-inf` 后求 softmax，会出现 `-inf-(-inf)`。不能先制造 NaN，再仅用 `nan_to_num` 修饰输出；前向看起来变成零，并不保证反向没有经过含 NaN 的中间量。

本实现把全 masked 行的 softmax 输入先改成安全的全零行，然后把被屏蔽位置的指数强制设为零。分母为零时用 1 代替，明确得到全零 probabilities 和全零输出。该路径对有限输入保留有限零梯度。

“全 masked 输出为零”这里指 attention 核心输出。完整 MHA 如果输出 Linear 带非零 bias，那么 `out_proj(0)` 会等于 bias；这不是 mask 失败。若业务要求 padding query 的最终整行也为零，应在输出投影后另乘 query-valid mask。

本 PyTorch 参考层还明确支持 `S=0`：用空维矩阵乘法得到零输出，并保留对 q/k/v 的零梯度连接。包内 GPU 教学 attention 的支持域较窄，会拒绝空 key 维；不要把不同实现的边界约定混为一谈。

### causal offset：KV cache 面试里最值得主动解释的细节

整序列计算时，query i 和 key j 都从位置 0 开始，条件是 `j<=i`。缓存已有 p 个 token、本次又输入 L 个 token 时，本次 query i 的绝对位置是 `p+i`，正确条件是：

\[
j\le p+i.
\]

例如已缓存 3 个 token，再输入 2 个 token，新 queries 位于 3、4，总 keys 位于 0…4：

| 本次 query | 绝对位置 | 可访问 keys |
|---|---:|---|
| 第 0 个 | 3 | 0、1、2、3 |
| 第 1 个 | 4 | 0、1、2、3、4 |

如果错误地使用一个 `[2,5]` 的普通左上角下三角矩阵，第 0 行只能看到 key 0，第 1 行只能看到 keys 0、1，几乎整个历史都被屏蔽。对于单 token decode，`L=1`，这种错误尤其隐蔽：程序完全能运行，输出却只看到了开头的位置。

代码在追加 cache 前记录 `query_start=cache.length`；追加新 K/V 后，再用这个旧长度生成 causal 条件。记录长度的前后顺序必须正确。

### KVCache 的实现和复杂度

`KVCache` 一次分配 `[B,H,T_max,d]` 的 key 和 value 存储，并维护当前有效长度。追加时只把新的 K/V 写入 `[length:length+new_T]`，随后返回有效前缀视图。

完整预分配存储是连续的，但 `cache.get()` 返回的前缀切片在未填满容量时通常不是连续 Tensor：相邻 head 的 stride 仍按 `T_max*d` 跨越，逻辑上的有效长度却只有 `length`。PyTorch reference matmul 可以处理这个 stride；接入本包要求连续输入的 GPU attention 时，需要选择支持 stride 的 kernel 或合适的缓存/读取布局。不要每个 token 都盲目把整个有效前缀 `.contiguous()`，否则会重新引入随历史增长的整段复制，抵消预分配的主要收益。

与每步 `torch.cat([old_k,new_k],dim=-2)` 相比，预分配避免为每个 token 分配并复制全部历史缓存。对于一次单 token 追加，新增写入量约为 `O(B*H*d)`；但 **attention 本身仍要读取历史 keys/values**，其单步成本仍随历史长度增长。不能因此把整个 decode 宣称为 O(1)。

代码选择一个清晰的教学边界：同一 batch 的序列共享一个长度；一个 cache 属于一个 attention layer 和一个独立会话批次。它不实现不同请求长度、分页、回收、prefix sharing、RoPE 或并发调度。真实模型如果使用 RoPE，应保证缓存中的 K 已按正确绝对位置处理；本练习是没有位置编码的单层 attention 等价性验证。

缓存路径会修改内部状态，因此只支持推理。`forward_step` 要求同时满足：

```python
model.eval()
with torch.inference_mode():
    y = model.forward_step(x, cache)
```

`eval()` 关闭本层训练 dropout；inference/no-grad 关闭梯度记录。代码不默默吞掉训练梯度：条件不满足时主动抛错。超过容量也先报错再写入，避免验证失败后留下半截状态。

如果在外层开启 autocast，cache 的 dtype 应匹配 QKV 投影后实际得到的 K/V dtype，而不能只看原始 x 的 dtype。预分配也不自动保证适合编译或 CUDA Graph：本教学 cache 用 Python 整数维护 length，切片形状随长度增长；这些状态变化可能触发 guard 或新的捕获。真实部署应另外设计长度表达、固定容量 mask、分桶和图复用策略，并验证性能。

### 怎样证明 prefill + decode 与整序列一致？

测试采用固定权重、同一输入和关闭 dropout 的模型：

1. 一次性计算完整 causal attention，保存全部位置的输出。
2. 重新建立空 cache，先喂前 3 个 token。
3. 继续追加一个 token，再追加一个多 token chunk，最后追加剩余 token。
4. 把各次新输出拼回原顺序，与第 1 步逐元素比较。

加入多 token chunk 很关键：只测试逐 token decode，有时发现不了 chunk 内未来 token 泄漏。测试还修改了输入中未来 token 的内容，验证过去位置的输出保持不变。

有些优化会改变浮点累加顺序，所以“等价”通常按明确的 dtype 与容差判断，不要求所有 GPU 路径按 bit 完全一致。这里小尺寸 FP64 参考测试采用较严格容差；低精度 GPU kernel 应另设符合精度的阈值。

### Attention 反向的完整骨架

忽略 dropout，令上游梯度为 `G=dL/dO`。因为 `O=PV`：

\[
dV=P^T G,
\qquad dP=GV^T.
\]

对一行 softmax，有：

\[
\frac{\partial p_i}{\partial a_j}=p_i(\delta_{ij}-p_j),
\]

所以不必显式构造 S×S Jacobian，向量形式为：

\[
dA=P\odot\left(dP-\operatorname{sum}(dP\odot P,\ dim=-1,\ keepdim=True)\right).
\]

最后 `A=QK^T/sqrt(d)`：

\[
dQ=dA K/\sqrt d,
\qquad dK=dA^T Q/\sqrt d.
\]

被屏蔽位置的概率与分数梯度均为零。若训练时加入 dropout，必须把 dropout 的实际 mask 与缩放应用到相应 backward，不能直接复用“没有 dropout”的梯度公式。

本参考实现依靠 autograd 计算这些链式导数；测试既与 SDPA 的前向/梯度比较，也对部分屏蔽与全屏蔽行做 double gradcheck。你应能把这些矩阵形状写出来，而不只是背四个公式。

### 优化方向：从普通 attention 到 FlashAttention

普通实现显式生成 `[B,H,T,T]` 分数和概率。单个这样的 FP32 张量就占 `4*B*H*T*T` 字节，实际峰值还包括多个中间结果和训练保存状态。将 Q/K/V 改为半精度，并不能让这个参考代码自动摆脱 T² 级中间存储。

分块 attention 的核心是按 K/V 块更新每行 softmax 的统计量和未归一化输出，不把全部分数写回 HBM。若已有统计量 `(m,l,U)`，新块分数为 S，则可以更新：

\[
m'=\max(m,\max S),\qquad \alpha=\exp(m-m'),
\]

\[
\widetilde P=\exp(S-m'),\quad
l'=\alpha l+\sum\widetilde P,\quad
U'=\alpha U+\widetilde P V_{block}.
\]

最终 `O=U/l`。这里旧的分母和输出都要乘同一个 rescale 系数 alpha；只更新最大值、不重新缩放旧累积量会出错。处理起始状态或一个块没有任何有效 key 时，还必须避免 `-inf-(-inf)`。这与本题全 masked 行的处理属于同一类数值边界。

面试回答可以是：“FlashAttention 的重要改进是减少 attention 中间结果的 HBM 读写，并安排分块与在线 softmax；精确 attention 的乘加规模不会因此普遍变成线性。对于不同 sequence 长度、head dim、prefill/decode，还需要不同并行和调度策略。”参考 [FlashAttention 项目](https://github.com/Dao-AILab/flash-attention)与 [Triton fused attention 教程](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)。

### MHA、MQA、GQA 和 KV cache 的关系

MHA 中每个 query head 都有对应 K/V head，即 `H_q=H_kv`；MQA 使用一个 K/V head；GQA 让一组 query heads 共享一个 K/V head。若每个 head 维度为 d，每个元素 s 字节，每层缓存约为：

\[
\text{KV bytes}=2 B T H_{kv} d s.
\]

因此减少 K/V head 数可降低缓存容量和读写量。实际支持 GQA 还要明确 q head 到 kv head 的映射；在演示中 `repeat_interleave` 可以表达数学，但若物化重复 K/V，会抵消一部分内存收益。这个包的 MHA 参考层没有隐式扩展 GQA，面试扩展时应新增明确接口并验证。

### 面试追问：可直接复述的回答

**问：为什么 prefill 和 decode 可能需要不同 kernel？**

答：“Prefill 通常同时处理多个 query，QK 和 PV 有较大的矩阵维度，数据复用与 Tensor Core 利用有较多空间。单 token decode 的 query 长度很小，却需要访问长历史 K/V；此时缓存读取、并行划分和低 batch 下的利用率更突出。是否受带宽或算力限制，需要结合实际形状与测量判断。”

**问：你的 cache 省掉了哪些计算？**

答：“它保存已处理 token 的 K 和 V，后续只对新 token 做 Q/K/V 投影，并用新的 Q 查询全部有效缓存。不再重新计算历史 K/V，但仍然需要把新 Q 与历史 K 做匹配，再用权重读 V。”

**问：用了 causal mask，为什么还要检查 cache offset？**

答：“causal 是绝对位置关系。缓存输入里 query 长度和 key 长度往往不同，本地 query 下标不能直接当绝对位置。我记录旧 cache 长度 p，让 query i 只访问 `j<=p+i`，并用 prefill+chunk+decode 对完整结果做验证。”

**问：能不能直接用 `softmax(masked_fill(-inf))`？**

答：“如果保证每行至少有一个有效有限分数，这样可以。如果接口支持全 masked 行，我会在 softmax 之前建立安全分支，并明确返回零，保证 backward 也没有 NaN。我不会只把最后输出的 NaN 改掉就认为问题解决。”

**问：返回 attention weights 会有什么影响？**

答：“完整 weights 本身是 L×S 规模。如果业务必须拿到它们，就不能同时宣称完全避免存储这张矩阵。生产推理若只需要 attention 输出，应让接口允许不返回权重；本参考代码主要服务可读性和检查，仍显式构建概率。”

**问：这个手写版本能替代生产中的 SDPA 吗？**

答：“它是透明的数学与边界参考，有训练支持和明确测试。生产选择还要比较融合程度、内存峰值、硬件指令、shape 覆盖和维护成本。我会用它验证新的 CUDA/Triton kernel，再测实际 workload，而不是因为代码能跑就替换已优化的实现。”

### 本题的完成标准

能手写完整 QKV→attention→output projection；解释每一步 shape；正确区分布尔 mask 语义；处理全 masked 和非方阵因果关系；理解并验证 cache 等价性；能推导 softmax/attention 梯度，并解释融合、GQA 与推理阶段的性能取舍。
<!-- exercise:9:end -->
