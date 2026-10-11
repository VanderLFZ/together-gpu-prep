# 验证记录与适用范围

记录日期：2026-10-10。完整原始记录在 results 目录。下表区分数学/注册检查、编译检查和真实 GPU 运行；这些结论不能互相替代。

## 已完成的实际检查

| 检查 | 结果 | 能支持的结论 |
|---|---|---|
| 最终 pytest 全套收集与执行 | **119 passed、330 skipped、0 failures、0 errors；共 449 项** | CPU 公式、梯度、参数合同、注册与 capture 的对应测试通过；GPU 项明确跳过 |
| PyTorch 层与 custom-op 测试子集 | **78 passed、2 skipped** | Linear/Norm/attention/cache，fake、opcheck、gradcheck、CPU fullgraph 等检查 |
| Triton host 参数检查 | **17 passed** | EPS 的 FP32 可表示范围与类型合同，不需 GPU |
| Roofline/benchmark 参考与统计检查 | **24 passed** | 数学单位、流量模型、reference 语义和统计定义 |
| Triton 自带 interpreter | **34/34 FP32 检查通过** | 原 kernel 的高层算术、索引和 mask；不等于 GPU 执行 |
| Triton 离线编译 | **48/48 特化成功生成 PTX 和 cubin** | 三种 dtype、多种 shape/mask/head dimension 的编译路径成立，目标 sm_80 |
| 完整 CUDA kernels.cu 离线编译 | **NVCC 13.0.48 编译成功** | 12 个 kernel entry 经 ptxas，生成 411,864-byte sm_80 object；未链接 |
| CUDA bindings.cpp host 语法 | **g++ -fsyntax-only 通过** | 与本次 PyTorch headers 对应的绑定声明和 C++ 语法检查 |
| torch_layers_demo.py | **执行成功** | 手写层参考一致、QKV 布局输出、prefill + decode 等价示例 |
| compile_demo.py，CPU / backend=eager | **执行成功** | 预期 graph break、where/cond 捕获、两种注册路径与梯度检查 |

330 个 skipped 的组成是：CUDA C++ GPU 测试 87 项、Triton GPU 测试 241 项、custom-op 的 GPU 接入测试 2 项。它们已经写好并被测试框架收集；“跳过”不代表数值通过。没有使用 CPU 时间代替 GPU 性能。

## 环境

| 项目 | 本次验证环境 |
|---|---|
| Python | 3.12.14 |
| PyTorch | 2.5.1+cpu |
| Triton | 3.1.0 |
| pytest | 9.1.1 |
| 主环境 NumPy | 2.3.5 |
| Interpreter 独立进程 NumPy | 1.26.4 |
| NVIDIA GPU | 当前环境没有可用设备 |
| 离线 CUDA compiler | 官方 NVIDIA pip 组件 NVCC 13.0.48 |
| 离线编译目标 | sm_80 / compute capability 8.0 |

NVCC 通过独立安装路径调用，没有加入主环境 PATH，因此 check_env 中 nvcc 字段可能为空；离线编译的版本、命令与结果另有完整记录。当前 CPU wheel 没有 CUDA runtime，不能据此检查扩展实际链接或在 GPU 上运行。

这组版本用于记录实际做过的检查，不是对所有 GPU 的安装推荐。新硬件应选择支持它的 PyTorch、CUDA toolkit、驱动和 Triton，重新运行实际 GPU 测试。

## 测试重点

CPU 检查覆盖 Linear 的参数注册和梯度、QKV 形状/stride、Norm 的前向/反向及 FP32/FP64 统计策略、autocast、不同 normalized_shape、causal offset、全遮蔽行的输出与梯度、空 attention key 域的约定，以及 cache 的单 token/多 token chunk 等价。

两种 custom_op 同时检查 schema/fake/autograd/compile 的集成合同与 double 数值梯度。新增的定向测试检查 inactive ReLU 如何阻断上游 Inf/NaN：实际 backward 使用 where，避免用乘法产生 0×Inf。Norm 还拒绝在 FP32 accumulator 中舍入为 0 或 Inf 的 eps，同时保留 FP64 中合法的极小 eps。

GPU 测试文件覆盖尾块、非整除尺寸、空张量、K=0、Norm、attention offset 和 mask、非默认 stream 等。CUDA 文件额外包含需要两块 GPU 的 device guard 检查。实际运行时需要相应硬件条件。

## Interpreter 的边界与复现

本次直接调用原 Triton kernel，由 Triton 自带 interpreter 在 CPU 上执行。没有改写 kernel 为另一套模拟实现，也没有让公开 CUDA-only wrapper 自动退回 CPU。

Triton 3.1.0 在本次 NumPy 2.3.5 环境里，最小独立 load/store probe 就不能正确读写。将 NumPy 1.26.4 安装到独立 target 目录，并只在解释器进程内设置 PYTHONPATH 后，该 probe 与全部 34 项数学检查通过。工具保留这项 preflight，遇到不兼容环境会报告 blocked，不会把环境错误当成练习通过。

需要复现时，建议创建单独环境并安装兼容版本，例如 CPU PyTorch、Triton 3.1.0、NumPy 1.26.4，再执行：

~~~bash
TRITON_INTERPRET=1 python tools/check_triton_interpreter.py
~~~

该检查不覆盖 GPU lane/layout lowering、真实并行同步、竞态、寄存器/shared-memory 资源限制、FP16/BF16 运行或性能。正式 GPU benchmark 和 demo 会拒绝在 TRITON_INTERPRET=1 下运行 Triton 路径，以防计时对象被改变。

## 离线编译的边界与复现

Triton 检查入口：

~~~bash
python tools/compile_triton_offline.py --arch 80 --output results/triton_offline_compile.json
~~~

该工具使用的 ASTSource/GPUTarget 接口是在 Triton 3.1.0 上验证的内部编译接口，新版本可能需要调整。它验证 AST→IR→PTX→cubin 的编译路径，没有分配 GPU tensor，也没有运行 device code。

CUDA 离线检查使用完整 kernels.cu；具体编译参数保存于 results/cuda_offline_compile_command.json。CPU Torch headers 不包含 CUDA 生成宏文件，因此仅在离线命令中使用头文件支持的 C10_CUDA_NO_CMAKE_CONFIGURE_FILE 开关。NVIDIA pip compiler 的 NVVM 工具目录还做了一处本地路径修正，详细记录于 results/cuda_offline_environment.txt。交付的 CUDA 源码与正常 JIT loader 没有加入这些环境修正。

## 尚未由本环境验证的项目

- CUDA 扩展在用户实际 CUDA/PyTorch 组合上的完整链接。
- NVIDIA GPU 上 CUDA 和 Triton 的数值结果、真实线程同步、stream 与设备切换。
- GPU FP16/BF16 的运行数值与实际 Tensor Core 路径。
- GPU Inductor 编译后的前后向运行。
- GPU 延迟、吞吐、显存和任何优化 speedup。

教材中的 roofline 图和时间数字明确属于假设硬件上的模型计算。compile demo 日志中的 CPU 计时用于展示“首次/稳态”测量方法；没有把它们写成 GPU 优化结论。

## 用户机器上的下一组检查

~~~bash
python -m gpu_prep.check_env --require-gpu
python -m pytest tests/test_cuda_ops.py tests/test_triton_ops.py tests/test_custom_ops.py -q
python examples/run_gpu.py --backend all --op all
python examples/compile_demo.py --device cuda --backend inductor --include-cond
python -m gpu_prep.benchmark --op matmul --shape 128,256,384 --backends torch,cuda,triton --output results/my_gpu_matmul.json
~~~

准备好 GPU 和编译工具链后，这些命令会真正调用相应实现。某个 GPU-only case 被跳过时，应查看其 skip 原因，而不是只看进程退出码。
