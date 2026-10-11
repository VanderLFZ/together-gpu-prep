"""Build a self-contained Chinese textbook from chapters and exact source files."""
from __future__ import annotations

import argparse
import ast
import base64
import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import markdown
from pygments.formatters import HtmlFormatter


ROOT = Path(__file__).resolve().parents[1]
BOOK_NAME = "GPU_Interview_12_Exercises_ZH"
FENCE = chr(96) * 3
CHAPTERS = [
    ("01_cuda.zh.md", "CUDA C++"),
    ("02_triton.zh.md", "Triton"),
    ("03_torch_layers.zh.md", "PyTorch"),
    ("04_compile_custom_ops.zh.md", "编译与算子注册"),
    ("05_roofline_benchmark.zh.md", "模型与实验"),
]
TITLES = {
    1: "Vector Add 与 Bias + ReLU 融合",
    2: "Row Sum 与 Row Max 归约",
    3: "数值稳定的行 Softmax",
    4: "Naive 与 Tiled 物理转置",
    5: "Naive 与 Tiled GEMM",
    6: "Roofline：工作量、流量与瓶颈模型",
    7: "Linear、QKV、Shape 与 Stride",
    8: "LayerNorm 与 RMSNorm：前向、反向与 GPU 实现",
    9: "Attention、MHA、Causal Mask 与 KV Cache",
    10: "torch.compile：Graph Break、Guard 与稳态性能",
    11: "自定义 CUDA / Triton 算子：Fake、Autograd 与编译",
    12: "性能实验、诊断与项目答辩",
}
CODE_MAP = {
    1: {
        "gpu_prep/csrc/kernels.cu": ["vector_add_kernel", "bias_relu_kernel", "vector_add_cuda", "bias_relu_cuda"],
        "gpu_prep/triton_ops.py": ["_vector_add_kernel", "vector_add", "_bias_relu_kernel", "bias_relu"],
    },
    2: {
        "gpu_prep/csrc/kernels.cu": ["block_reduce", "row_reduce_kernel", "row_reduce_impl", "row_sum_cuda", "row_max_cuda"],
        "gpu_prep/triton_ops.py": ["_row_reduce_kernel", "_row_reduce", "row_sum", "row_max"],
    },
    3: {
        "gpu_prep/csrc/kernels.cu": ["softmax_kernel", "softmax_cuda"],
        "gpu_prep/triton_ops.py": ["_softmax_kernel", "softmax"],
    },
    4: {
        "gpu_prep/csrc/kernels.cu": ["transpose_naive_kernel", "transpose_tiled_kernel", "transpose_impl", "transpose_naive_cuda", "transpose_tiled_cuda"],
        "gpu_prep/triton_ops.py": ["_transpose_naive_kernel", "_transpose_tiled_kernel", "_transpose", "transpose_naive", "transpose_tiled"],
    },
    5: {
        "gpu_prep/csrc/kernels.cu": ["matmul_naive_kernel", "matmul_tiled_kernel", "matmul_impl", "matmul_naive_cuda", "matmul_tiled_cuda"],
        "gpu_prep/triton_ops.py": ["_matmul_naive_kernel", "_matmul_tiled_kernel", "_matmul", "matmul_naive", "matmul_tiled"],
    },
    6: {"gpu_prep/roofline.py": ["Workload", "vector_add_workload", "gemm_workload", "roofline_estimate", "amdahl_speedup"]},
    7: {"gpu_prep/torch_layers.py": ["ManualLinear", "split_qkv_heads"]},
    8: {
        "gpu_prep/torch_layers.py": ["ManualLayerNorm", "ManualRMSNorm"],
        "gpu_prep/csrc/kernels.cu": ["norm_kernel", "layer_norm_cuda", "rms_norm_cuda"],
        "gpu_prep/triton_ops.py": ["_norm_kernel", "_norm_shape", "layer_norm", "rms_norm"],
    },
    9: {
        "gpu_prep/torch_layers.py": ["scaled_dot_product_attention", "KVCache", "MultiHeadSelfAttention"],
        "gpu_prep/csrc/kernels.cu": ["attention_kernel", "attention_cuda"],
        "gpu_prep/triton_ops.py": ["_attention_kernel", "attention"],
    },
    10: {"examples/compile_demo.py": ["bad_python_branch", "tensor_where", "tensor_cond", "median_wall_ms", "main"]},
    11: {"gpu_prep/custom_ops.py": None},
    12: {"gpu_prep/benchmark.py": ["Case", "build_case", "make_callable", "check_output", "measure_cuda", "summarize_samples", "main"]},
}


def source_anchor(relative: str) -> str:
    return "source-" + re.sub(r"[^a-z0-9]+", "-", relative.lower()).strip("-")


def shift_headings(text: str, levels: int) -> str:
    lines, fenced = [], False
    for line in text.splitlines():
        if line.lstrip().startswith((FENCE, "~~~")):
            fenced = not fenced
        match = re.match(r"^(#{1,6})\s+(.*)$", line)
        if match and not fenced:
            line = "#" * min(6, len(match[1]) + levels) + " " + match[2]
        lines.append(line)
    return "\n".join(lines)


def extract_python(relative: str, names: list[str] | None) -> str:
    text = (ROOT / relative).read_text(encoding="utf-8")
    if names is None:
        return text
    nodes = {node.name: node for node in ast.parse(text).body
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    lines = text.splitlines()
    result = []
    for name in names:
        node = nodes[name]
        first = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
        result.append("\n".join(lines[first:node.end_lineno]))
    return "\n\n\n".join(result) + "\n"


def extract_cuda(relative: str, names: list[str]) -> str:
    text = (ROOT / relative).read_text(encoding="utf-8")
    results = []
    for name in names:
        # Kernel source uses brace-balanced functions without raw-string literals.
        pattern = r"(?m)^(?:template\s*<[^\n]+>\n)?(?:__global__ void|__device__ float|at::Tensor)\s+" + re.escape(name) + r"\s*\("
        match = re.search(pattern, text)
        if match is None:
            raise ValueError(f"Could not locate {relative}:{name}")
        opening = text.index("{", match.start())
        depth, position = 1, opening + 1
        # Strip comments/strings only for brace counting while preserving offsets.
        masked = list(text[opening + 1:])
        tail = text[opening + 1:]
        lexical = r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\''
        for token in re.finditer(lexical, tail, flags=re.S):
            for i in range(token.start(), token.end()):
                masked[i] = " "
        for character in masked:
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
            position += 1
            if depth == 0:
                break
        if depth:
            raise ValueError(f"Unbalanced braces: {name}")
        results.append(text[match.start():position])
    return "\n\n".join(results) + "\n"


def code_block(text: str, language: str) -> str:
    fence = FENCE
    while fence in text:
        fence += chr(96)
    return fence + language + "\n" + text.rstrip() + "\n" + fence + "\n"


def all_source_files() -> list[Path]:
    paths = []
    for folder in ("gpu_prep", "examples", "tests", "tools"):
        for path in (ROOT / folder).rglob("*"):
            if path.is_file() and path.suffix in (".py", ".cu", ".cpp", ".h", ".cjs") and "__pycache__" not in path.parts:
                paths.append(path)
    for name in ("pyproject.toml", "requirements.txt", "MANIFEST.in"):
        paths.append(ROOT / name)
    return sorted(paths, key=lambda p: p.relative_to(ROOT).as_posix())


def assemble_markdown() -> tuple[str, dict]:
    sections = {number: [] for number in TITLES}
    prefaces = []
    for filename, label in CHAPTERS:
        content = (ROOT / "docs" / "chapters" / filename).read_text(encoding="utf-8")
        first_marker = content.find("<!-- exercise:")
        prefix = content[:first_marker].strip() if first_marker >= 0 else ""
        if prefix:
            prefaces.append(shift_headings(prefix, 1))
        for number, body in re.findall(
            r"<!-- exercise:(\d+):start -->(.*?)<!-- exercise:\1:end -->", content, flags=re.S
        ):
            sections[int(number)].append((label, body.strip()))
    if any(not content for content in sections.values()):
        raise ValueError("Every exercise must have an explanation.")
    parts = [
        "# GPU Programming 面试：12 题完整代码与中文答案\n",
        "CUDA C++ · Triton · PyTorch · 数学推导 · 优化问答 · 测试与实验\n",
        "**阅读说明：** 每道题先给讲解和追问答案，再列出本题核心函数的完整代码。"
        "章节中的代码由工程源码自动提取，没有省略函数体；公共依赖、完整模块、测试与工具在文末逐文件全文收录。"
        "直接运行请使用代码包中的工程，避免只复制一段函数后遗漏 import、常量或公共校验函数。\n",
        "[TOC]\n",
        shift_headings((ROOT / "docs" / "chapters" / "00_start.zh.md").read_text(encoding="utf-8"), 1),
    ]
    if (ROOT / "VALIDATION.md").exists():
        parts.append(shift_headings((ROOT / "VALIDATION.md").read_text(encoding="utf-8"), 1))
    parts.append("## 各后端共享的工程约定\n\n" + "\n\n".join(prefaces))
    function_count = 0
    for number, title in TITLES.items():
        parts.append(f'<a id="exercise-{number}"></a>\n\n## 第 {number} 题｜{title}\n')
        links = [f'[{path}](#{source_anchor(path)})' for path in CODE_MAP[number]]
        parts.append("**完整源文件：** " + "；".join(links) + "。\n")
        for label, body in sections[number]:
            parts.append(shift_headings(body, 1))
        if number == 6:
            svg = ROOT / "docs" / "assets" / "roofline_model.svg"
            if svg.exists():
                image_data = base64.b64encode(svg.read_bytes()).decode()
                parts.append('### Roofline 模型图\n\n'
                             '下图使用假想的 100 TFLOP/s、2 TB/s；圆点表示数学屋顶上的位置，不是性能测量。\n\n'
                             f'![Roofline 模型，所有点均为计算上界而非 GPU 实测](data:image/svg+xml;base64,{image_data})\n')
        parts.append("### 本题核心实现：函数体全文\n\n"
                     "以下内容直接来自随包源码。import、常量、共用的检查/归约辅助函数及所有绑定，见上面的完整源文件链接和文末源码附录。\n")
        for relative, names in CODE_MAP[number].items():
            parts.append(f"#### {relative}\n")
            if relative.endswith(".cu"):
                extracted = extract_cuda(relative, names)
                language = "cuda"
            else:
                extracted = extract_python(relative, names)
                language = "python"
            function_count += len(names) if names else 1
            parts.append(code_block(extracted, language))
    parts.append(shift_headings((ROOT / "docs" / "SOURCES.md").read_text(encoding="utf-8"), 1))
    paths = all_source_files()
    parts.append("## 完整源码附录\n\n"
                 "下面逐文件给出工程源码、测试和构建工具的完整文本。文件之间存在 import/链接关系，"
                 "建议解压工程后直接运行；本附录用于离线阅读、查找和逐行对照。\n")
    source_lines = 0
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        language = {".py": "python", ".cu": "cuda", ".cpp": "cpp", ".h": "cpp", ".cjs": "javascript", ".toml": "toml"}.get(path.suffix, "text")
        text = path.read_text(encoding="utf-8")
        source_lines += len(text.splitlines())
        parts.append(f'<a id="{source_anchor(relative)}"></a>\n\n### {relative}\n')
        parts.append(code_block(text, language))
    stats = {
        "exercises": len(TITLES), "source_files_embedded_in_full": len(paths),
        "source_lines_embedded_in_full": source_lines, "chapter_code_sections": function_count,
        "explanation_blocks": {str(k): len(v) for k, v in sections.items()},
    }
    return "\n\n".join(parts).rstrip() + "\n", stats


def render_math(body: str, katex_dir: Path) -> tuple[str, int]:
    pattern = re.compile(
        r'<(?P<tag>span|div) class="arithmatex">\s*<(?:span|div) class="MathJax_Preview">.*?</(?:span|div)>'
        r'\s*<script type="math/tex(?P<display>; mode=display)?">(?P<tex>.*?)</script>\s*</(?P=tag)>',
        flags=re.S,
    )
    matches = list(pattern.finditer(body))
    data = [{"tex": html.unescape(m["tex"]), "display": bool(m["display"])} for m in matches]
    node = os.environ.get("CODEX_PRIMARY_RUNTIME_NODE") or shutil.which("node")
    if not node:
        raise RuntimeError("Node is required to typeset formulas at build time.")
    run = subprocess.run([node, str(ROOT / "tools" / "render_math.cjs"), str(katex_dir.resolve())],
                         input=json.dumps(data), text=True, capture_output=True, check=True)
    rendered = json.loads(run.stdout)
    failures = [{"tex": data[i]["tex"], "error": r["error"]} for i, r in enumerate(rendered) if r["error"]]
    if failures:
        raise RuntimeError("Math rendering failed: " + json.dumps(failures, ensure_ascii=False))
    iterator = iter(rendered)
    body = pattern.sub(lambda _: next(iterator)["html"], body)
    if '<script type="math/tex' in body:
        raise RuntimeError("An unrendered formula remains in the HTML.")
    return body, len(rendered)


def embedded_katex_css(katex_dir: Path) -> str:
    css = (katex_dir / "dist" / "katex.min.css").read_text(encoding="utf-8")
    def replace_src(match):
        urls = re.findall(r"url\(([^)]+)\)", match[0])
        name = next((u.strip("\"'") for u in urls if ".woff2" in u), None)
        if not name:
            return match[0]
        encoded = base64.b64encode((katex_dir / "dist" / name).read_bytes()).decode()
        return 'src:url(data:font/woff2;base64,' + encoded + ') format("woff2");'
    return re.sub(r"src:[^;]+;", replace_src, css)


def embedded_cjk_css(font_dir: Path, text: str) -> tuple[str, int]:
    """Keep required original Unicode-range font subsets; no font CDN at read time."""
    css = (font_dir / "400.css").read_text(encoding="utf-8")
    used = {ord(character) for character in text if ord(character) > 127}
    blocks = []
    for block in re.findall(r"@font-face\s*\{[^}]+\}", css):
        range_match = re.search(r"unicode-range:\s*([^;]+);", block)
        if range_match:
            intervals = [(int(low, 16), int(high or low, 16)) for low, high in
                         re.findall(r"U\+([0-9a-fA-F]+)(?:-([0-9a-fA-F]+))?", range_match[1])]
            if not any(low <= point <= high for point in used for low, high in intervals):
                continue
        font_path = re.search(r"url\(([^)]+\.woff2)\)", block)
        if not font_path:
            continue
        data = base64.b64encode((font_dir / font_path[1].strip("\"'")).read_bytes()).decode()
        block = re.sub(r"src:[^;]+;", "src:url(data:font/woff2;base64," + data + ") format('woff2');", block)
        blocks.append(block)
    return "\n".join(blocks), len(blocks)


STYLE = r"""
:root {--ink:#172e41;--muted:#536979;--line:#d9e3ea;--accent:#006e61;--paper:#ffffff;--back:#f4f7f9}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:24px}
body{margin:0;background:var(--back);color:var(--ink);font:16px/1.9 "Noto Sans SC",system-ui,-apple-system,"Microsoft YaHei",sans-serif}
.layout{display:grid;grid-template-columns:280px minmax(0,1fr);max-width:1560px;margin:auto}
nav{position:sticky;top:0;height:100vh;overflow:auto;padding:28px 24px;border-right:1px solid var(--line);font-size:13px;line-height:1.65}
.brand{font-size:18px;font-weight:750;color:var(--accent);margin-bottom:8px}
.navhint{color:var(--muted);font-size:12px;margin:0 0 20px}
nav ul{padding-left:15px}nav>div>ul{padding-left:0;list-style:none}nav li{margin:7px 0}nav a{color:var(--muted);text-decoration:none}nav a:hover{color:var(--accent)}
main{min-width:0;padding:44px 5.2vw 90px;background:var(--paper)}
article{max-width:1100px;margin:auto}
h1,h2,h3,h4,h5{line-height:1.4;letter-spacing:.005em}
h1{font-size:34px;margin:0 0 18px}h2{font-size:26px;margin:65px 0 24px;border-top:2px solid var(--line);padding-top:30px}
h3{font-size:21px;margin:35px 0 18px}h4{font-size:18px;margin:26px 0 14px;color:#1a655d}
p{margin:14px 0}a{color:var(--accent);text-underline-offset:3px;overflow-wrap:anywhere}
strong{font-weight:720}li{margin:8px 0}blockquote{border-left:4px solid var(--accent);background:#eef7f4;margin:22px 0;padding:4px 20px}
table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.6;margin:24px 0;display:block;overflow:auto}
th{background:#eaf1f5;text-align:left;white-space:normal}td,th{border:1px solid var(--line);padding:12px 14px;vertical-align:top;min-width:90px}
tr:nth-child(even) td{background:#f8fafb}
code{font-family:"KaTeX_Typewriter","SFMono-Regular",Consolas,"Liberation Mono","Noto Sans SC",monospace;font-size:.88em;background:#edf2f6;padding:2px 5px;border-radius:4px;overflow-wrap:anywhere}
pre{padding:23px 22px;margin:0;overflow:auto;font-family:"KaTeX_Typewriter","SFMono-Regular",Consolas,"Liberation Mono","Noto Sans SC",monospace;font-size:13px;line-height:1.65;tab-size:4}
pre code{background:none;padding:0;font-size:inherit;overflow-wrap:normal}
.codehilite{position:relative;background:#f7f9fc;border:1px solid #dce5ed;border-radius:9px;margin:22px 0}
.copy{position:sticky;float:right;right:8px;top:8px;margin:8px 8px -34px 0;border:1px solid #cedbe4;border-radius:5px;background:#fff;color:#426073;padding:4px 9px;cursor:pointer;font-family:"Noto Sans SC",sans-serif;font-size:11px;z-index:1}
.copy:hover{background:#e8f3ee}
img{max-width:100%;height:auto}.katex-display{overflow-x:auto;overflow-y:hidden;padding:10px 0}
.katex{font-size:1.06em}.katex .base{white-space:nowrap}
.tagline{font-size:12px;letter-spacing:.12em;color:var(--accent);font-weight:750;margin-bottom:20px}
.footer{margin-top:70px;padding-top:20px;border-top:1px solid var(--line);font-size:12px;color:var(--muted)}
@media(max-width:1000px){.layout{grid-template-columns:220px minmax(0,1fr)}nav{padding:22px 15px}main{padding:35px 28px 70px}h1{font-size:29px}}
@media(max-width:760px){.layout{display:block}nav{position:relative;height:auto;max-height:300px;border-bottom:1px solid var(--line);border-right:0}main{padding:30px 19px 60px}h1{font-size:26px}h2{font-size:23px}pre{font-size:12px;padding:20px 15px}table{font-size:13px}}
@media print{nav,.copy{display:none}.layout{display:block}main{padding:0}body{background:white;font-size:10pt}h2{break-before:page}h3,h4{break-after:avoid}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:8pt}table{display:table;font-size:8pt}a{color:inherit}img{max-height:18cm}.codehilite{break-inside:auto}.katex-display{overflow:visible}}
"""


SCRIPT = r"""
document.querySelectorAll('.codehilite').forEach(function(block) {
  var button=document.createElement('button');
  button.className='copy'; button.textContent='复制代码'; button.type='button';
  button.addEventListener('click',async function(){
    var text=block.querySelector('pre').innerText;
    try {
      if(navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
      } else {
        var field=document.createElement('textarea');
        field.value=text; field.style.position='fixed'; field.style.opacity='0';
        document.body.appendChild(field); field.select();
        if(!document.execCommand('copy')) throw new Error('copy unavailable');
        field.remove();
      }
      button.textContent='已复制';
    } catch(e) {button.textContent='请选中代码复制';}
    setTimeout(function(){button.textContent='复制代码';},1800);
  });
  block.prepend(button);
});
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--katex-dir", type=Path, required=True)
    parser.add_argument("--cjk-font-dir", type=Path, required=True)
    args = parser.parse_args()
    content, stats = assemble_markdown()
    markdown_path = ROOT / (BOOK_NAME + ".md")
    markdown_path.write_text(content, encoding="utf-8")
    converter = markdown.Markdown(
        extensions=["tables", "fenced_code", "codehilite", "toc", "pymdownx.arithmatex"],
        extension_configs={
            "toc": {"toc_depth": "2-3", "permalink": False},
            "codehilite": {"guess_lang": False, "css_class": "codehilite"},
            "pymdownx.arithmatex": {"generic": False},
        },
    )
    body = converter.convert(content)
    # The TOC is kept in the sidebar; do not duplicate it before the main text.
    toc = converter.toc
    body = body.replace(toc, "", 1)
    body, formula_count = render_math(body, args.katex_dir)
    cjk_css, font_subsets = embedded_cjk_css(args.cjk_font_dir, content)
    css = cjk_css + STYLE + HtmlFormatter(style="friendly").get_style_defs(".codehilite") + embedded_katex_css(args.katex_dir)
    document = (
        '<!doctype html>\n<html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>GPU Programming 面试 · 12 题完整代码与中文答案</title>'
        '<style>' + css + '</style></head><body><div class="layout">'
        '<nav aria-label="教材目录"><div class="brand">GPU Programming · 12 题</div>'
        '<p class="navhint">完整源码与答案 · 离线阅读<br>点击目录跳转；Ctrl / ⌘ + F 搜索全文。</p>' + toc + '</nav>'
        '<main><article><div class="tagline">CUDA · TRITON · PYTORCH · FULL SOURCE</div>' + body +
        '<div class="footer">原创教学材料。性能模型与实际测量分别标注；请参照验证记录。数学排版使用 KaTeX（MIT），中文字体使用 Noto Sans SC（OFL）。</div>'
        '</article></main></div><script>' + SCRIPT + '</script></body></html>\n'
    )
    html_path = ROOT / (BOOK_NAME + ".html")
    html_path.write_text(document, encoding="utf-8")
    license_dir = ROOT / "licenses"
    license_dir.mkdir(exist_ok=True)
    shutil.copyfile(args.katex_dir / "LICENSE", license_dir / "KaTeX_LICENSE.txt")
    shutil.copyfile(args.cjk_font_dir / "LICENSE", license_dir / "Noto_Sans_SC_OFL.txt")
    stats.update({
        "display_and_inline_formulas_rendered": formula_count,
        "markdown_characters": len(content), "markdown_bytes": markdown_path.stat().st_size,
        "html_bytes": html_path.stat().st_size, "code_blocks": body.count('<div class="codehilite">'),
        "embedded_cjk_font_subsets": font_subsets,
    })
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "book_build.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(html_path)
    print(markdown_path)


if __name__ == "__main__":
    main()
