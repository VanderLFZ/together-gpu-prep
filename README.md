# GPU Programming 面试：12 题完整实现与中文答案

[![CPU correctness](https://github.com/VanderLFZ/together-gpu-prep/actions/workflows/ci.yml/badge.svg)](https://github.com/VanderLFZ/together-gpu-prep/actions/workflows/ci.yml)

适用于准备 CUDA / Triton / PyTorch / ML systems 编程面试。对应此前设计的 12 道练习，包含完整工程、公式推导、常见错误、优化路线、逐题追问答案、测试与计时脚本。

这是一套原创准备材料，不把练习列表宣称为 Together AI 的官方原题。

## 从这里开始

- **完整阅读：** [GPU_Interview_12_Exercises_ZH.html](GPU_Interview_12_Exercises_ZH.html)，单文件、离线阅读、目录导航、可复制代码。
- **完整文本：** [GPU_Interview_12_Exercises_ZH.md](GPU_Interview_12_Exercises_ZH.md)，所有讲解及源码均完整收录。
- **PDF 阅读：** [GPU_Interview_12_Exercises_ZH.pdf](output/pdf/GPU_Interview_12_Exercises_ZH.pdf)，A4 排版、可搜索文本、目录链接和 PDF 书签，包含完整源码附录。
- **实际验证范围：** [VALIDATION.md](VALIDATION.md)，区分 CPU 数值/梯度、离线编译与尚需 GPU 运行的测试。
- **官方参考：** [docs/SOURCES.md](docs/SOURCES.md)。

## 12 题与实现位置

| 题号 | 内容 | CUDA C++ | Triton | PyTorch / Python |
|---|---|---|---|---|
| 1 | Vector Add、Bias+ReLU | 有 | 有 | Reference、接入与 backward |
| 2 | Row Sum、Row Max | 有 | 有 | Reference |
| 3 | 稳定 Softmax | 有 | 有 | Reference |
| 4 | 物理 Transpose | Naive / tiled | Naive / tiled | contiguous 对照 |
| 5 | GEMM | Naive / tiled | Naive / tiled | matmul 对照 |
| 6 | Roofline | 使用前题工作量 | 使用前题工作量 | 完整计算模型 |
| 7 | Linear、QKV、stride | 使用 GEMM 的知识 | 使用 GEMM 的知识 | 完整可训练层 |
| 8 | LayerNorm、RMSNorm | Forward | Forward | 完整可训练层、反向推导 |
| 9 | MHA、causal mask、KV cache | Dense forward | Online softmax forward | 完整 MHA 与 cache |
| 10 | torch.compile | 作为后端接入 | 作为后端接入 | graph break / where / cond |
| 11 | Custom operator | CUDA custom_op | Triton custom_op | fake / autograd / opcheck |
| 12 | 实验与答辩 | 统一验证、计时 | 统一验证、计时 | reference / roofline / Amdahl |

## 安装

建议 Linux + NVIDIA GPU。先按照 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)安装与硬件、驱动和 CUDA 运行时兼容的 PyTorch，然后在项目根目录运行：

~~~bash
python -m pip install -e ".[test,cuda]"
python -m pip install -e ".[triton]"
python -m gpu_prep.check_env
~~~

纯 CPU 学习 PyTorch 公式时，不必安装 Triton 或 CUDA toolkit，只需要匹配的 CPU PyTorch 和 test 依赖。CUDA C++ 路径需要 nvcc、C++ 编译器、Ninja 与 CUDA 支持的 PyTorch；第一次调用会编译。Import 不会触发扩展构建。

本次 CPU 验证使用 PyTorch 2.5.1+cpu，Triton 离线编译使用 3.1.0。这个组合是本次检查基线，不是对新 GPU 的通用安装推荐。较新硬件请安装支持它的版本。命令行工具和 compiler 内部 API 随版本演进，实际 GPU 验证应在最终运行环境执行。

## 运行

~~~bash
# CPU 即可检查公式、梯度、注册、Dynamo 图捕获
python -m pytest -q
python examples/torch_layers_demo.py --device cpu
python examples/compile_demo.py --device cpu --backend eager --include-cond
python -m gpu_prep.roofline

# 真 GPU 数值测试与稳态 benchmark
python -m gpu_prep.check_env --require-gpu
python examples/run_gpu.py --backend all --op all
python -m pytest tests/test_cuda_ops.py tests/test_triton_ops.py -q
python -m gpu_prep.benchmark --op matmul --shape 128,256,384 --output results/matmul.json

# FP16/BF16 使用支持它们的后端
python -m gpu_prep.benchmark --op matmul --shape 256,512,768 --dtype float16 --backends torch,triton
~~~

Benchmark 的 GEMM shape 顺序是 M,N,K，Attention 是 B,H,L,S,D。它测输入已经在 GPU 的算子 wrapper，包括 metadata 检查、必要输出分配和 launch；首次 JIT 和独立数值检查不计入稳态。GPU event 与同步 host wall 时间分开报告。没有 GPU 时会清楚报错，不生成替代性的“GPU 性能”数字。

## 接口限制

CUDA C++ 原始入口要求同设备、contiguous、FP32、有限值和无中间溢出；只提供 forward。Triton 原始入口支持 contiguous FP32/FP16/BF16、FP32 累加、NVIDIA compute capability 8.0+，同样只提供 forward。

Triton 的单行 reduction/softmax/norm 将行宽限制在 1–8192；CUDA 这些行运算可循环处理更大宽度。CUDA attention 限制 key 长度 1–4096；Triton attention 支持 head dimension 16/32/64/128。二者均要求 S>0。Triton attention 额外支持规定形状的 boolean mask。具体检查见各函数 docstring 和测试。

第 11 题注册的两种 bias+ReLU custom_op 提供完整 PyTorch backward，并把 GPU 接入固定为 FP32，方便保持共同合同。其他原始 GPU kernel 不能直接宣称可训练；第 7–9 题的 PyTorch 实现使用普通 tensor 运算，支持其文档约定内的 autograd。

## 教材重建

完整教材已随包生成，阅读不需要这些构建依赖。如果修改源码后希望重建：

~~~bash
python -m pip install -e ".[docs]"
npm install --prefix .book_deps katex@0.16.22 @fontsource/noto-sans-sc@5.3.0 --ignore-scripts
python tools/make_roofline_figure.py
python tools/build_book.py --katex-dir .book_deps/node_modules/katex --cjk-font-dir .book_deps/node_modules/@fontsource/noto-sans-sc
~~~

HTML 将数学公式预先排版，中文/数学字体与图形嵌入单文件，不依赖 CDN。KaTeX 的 MIT 许可与 Noto Sans SC 的 OFL 许可随包提供。源码章节来自 docs/chapters；完整源文件会自动写入教材附录，减少讲解与工程不同步的风险。

生成 PDF 时，先保证 HTML 对应最新教材，再运行：

~~~bash
node tools/export_pdf.cjs
~~~

PDF 导出需要 Node.js 22+ 和 Chrome/Chromium，无须安装 npm 依赖。macOS 会自动寻找已安装的 Google Chrome，Linux 会寻找 PATH 中的 Chrome/Chromium；也可通过 `--chrome` 传入浏览器可执行文件路径。默认 PDF 写入 `output/pdf/`。源码、章节、离线 HTML 和交付 PDF 均保存在本仓库中。

## GitHub 自动检查

GitHub Actions 使用 Linux、Python 3.12、PyTorch 2.5.1 CPU 和 Triton 3.1.0 执行正确性测试与 CPU 示例。GPU 测试按测试合同跳过，实际 GPU 验证和性能实验请在 NVIDIA GPU 上运行。
