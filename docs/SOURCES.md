# 参考资料与来源边界

查阅日期：2026-10-10。下列链接用于核对 API 语义、数学背景和进一步学习。项目源码为针对这套练习编写的原创教学实现；没有复制某篇教程的完整 kernel 并声称是自己的性能成果。

## CUDA 与性能分析

1. [NVIDIA CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)：访存、共享内存、正确性与优化方法。
2. [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)：线程协作、同步、执行与内存模型。
3. [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)：roofline、内存层级与 kernel 指标。
4. [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)：CPU/GPU 时间线和并发观察。
5. [PyTorch C++ Extension API](https://docs.pytorch.org/docs/stable/cpp_extension.html)：load、构建工具链与 JIT extension。
6. [PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html)：设备、stream 与异步执行。

## Triton 官方教程与 API

1. [Vector Addition](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)：program、arange、load/store mask。
2. [Fused Softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)：融合归约与资源取舍。
3. [Matrix Multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)：矩阵 tile 与 dot。
4. [Layer Normalization](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)：统计量、前向与反向的组织。
5. [Fused Attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)：分块 attention 的更完整实现背景。
6. [tl.dot](https://triton-lang.org/main/python-api/generated/triton.language.dot.html)：dtype、accumulator 与 input_precision。
7. [tl.load](https://triton-lang.org/main/python-api/generated/triton.language.load.html)：mask 与 other。
8. [tl.trans](https://triton-lang.org/main/python-api/generated/triton.language.trans.html)：block 轴交换；不等同于手动声明 CUDA shared-memory tile。
9. [Triton 项目与支持平台](https://github.com/triton-lang/triton)：安装和硬件支持说明。

官方最新文档可能描述比本次验证的 Triton 3.1.0 更多的参数和硬件路径。源码仅使用本次编译检查实际支持的功能，不因为新版文档列出了某种模式，就假定旧版也支持。

## PyTorch 层、编译与注册

1. [Linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html)：权重布局与输入末维约定。
2. [LayerNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)：normalized_shape、population variance 与 affine。
3. [RMSNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html)：均方根归一化与 eps。
4. [Scaled Dot Product Attention](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html)：boolean mask、dropout、非方阵 causal 对齐与 backend 选择。
5. [Tensor Views](https://docs.pytorch.org/docs/stable/tensor_view.html)：view、stride、contiguous 与复制。
6. [torch.compile tutorial](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)：图捕获与编译入门。
7. [Common Graph Breaks](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/programming_model.common_graph_breaks.html)：数据依赖和捕获问题。
8. [torch.cond](https://docs.pytorch.org/docs/main/cond.html)：结构化控制流及限制。
9. [Custom C++ and CUDA Operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)：fake、autograd、opcheck 与扩展接入。
10. [User-Defined Triton Kernels with torch.compile](https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html)：Triton kernel 接入与 triton_op 的版本相关差异。
11. [Numerical Accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)：归约顺序、TF32、低精度累加与不同实现的误差。
12. [Gradcheck Mechanics](https://docs.pytorch.org/docs/stable/notes/gradcheck.html)：数值梯度检查的假设和用途。

本包把 opcheck 用于注册合同，把 double gradcheck 用于导数数学验证。CPU backend=eager 的 fullgraph 通过只证明相应图捕获和结果检查，不证明 GPU Inductor 性能。

## Attention 数学原始论文

1. [Online normalizer calculation for softmax — Milakov & Gimelshein](https://arxiv.org/abs/1805.02867)：在线最大值/分母更新。
2. [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness — Dao et al.](https://arxiv.org/abs/2205.14135)：分块重排 attention 的 IO 视角。

“数学上等价”意味着在实数算术模型中等价，不意味着有限精度下逐 bit 一样。教材的 Triton attention 保留 FP32 probabilities 做第二个 dot，便于解释误差；它没有实现生产 FlashAttention 的所有 pipeline、反向或硬件专用优化。

## 岗位背景与进一步阅读

- [Together AI: Inside the Together AI Kernels Team](https://www.together.ai/blog/inside-the-together-ai-kernels-team)：岗位所处的 GPU kernel 与系统方向背景。
- [CUDA GEMM Worklog — Simon Boehm](https://siboehm.com/articles/22/CUDA-MMM)：作者的逐步 GEMM 实验，可用于对照优化方法；本文不复用其硬件性能数字。
- [GPU MODE Lectures](https://github.com/gpu-mode/lectures)：延伸课程与讲座。
- [ThunderKittens](https://github.com/HazyResearch/ThunderKittens)：进一步研究 tile 级抽象与更复杂 kernel。

这些材料不能证明某公司必考某题。面试准备应以真实岗位职责、你的项目和收到的面试说明为主；本教材用公开技术知识构造练习和问答。
