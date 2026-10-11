"""Generate a scientific figure of an explicitly hypothetical roofline model."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gpu_prep.roofline import gemm_workload, vector_add_workload


def main():
    root = Path(__file__).resolve().parents[1]
    target = root / "docs" / "assets"
    target.mkdir(parents=True, exist_ok=True)
    intensity = np.logspace(-2, 3, 600)
    peak, bandwidth = 100.0, 2.0
    ceiling = np.minimum(peak, bandwidth * intensity)
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none",
    }):
        fig, ax = plt.subplots(figsize=(10.4, 5.1), constrained_layout=True)
        fig.patch.set_facecolor("#f7fafc")
        ax.set_facecolor("#f7fafc")
        ax.loglog(intensity, ceiling, color="#0b7466", linewidth=3, label="Model performance ceiling")
        ax.axvline(peak / bandwidth, color="#687785", linewidth=1.2, linestyle="--")
        cases = [
            ("Vector add, FP32", vector_add_workload(1_000_000), (15, 13)),
            ("GEMM 1 x 4096 x 4096, FP32", gemm_workload(1, 4096, 4096), (18, -23)),
            ("GEMM 1024 cubed, FP32", gemm_workload(1024, 1024, 1024), (-153, -33)),
        ]
        for label, workload, offset in cases:
            x = workload.arithmetic_intensity
            y = min(peak, bandwidth * x)
            ax.scatter([x], [y], color="#e49b32", edgecolors="#56320a", s=64, zorder=5)
            ax.annotate(label, (x, y), xytext=offset, textcoords="offset points", fontsize=9)
        ax.text(peak / bandwidth * 1.12, 0.034, "Ridge = 50 FLOP/byte", fontsize=9, color="#56616d")
        ax.set(xlabel="Arithmetic intensity (FLOP/byte; ideal logical traffic)",
               ylabel="Model ceiling (TFLOP/s)", xlim=(0.01, 1000), ylim=(0.02, 240))
        ax.grid(True, which="major", alpha=0.16)
        ax.set_title("Hypothetical GPU: 100 TFLOP/s and 2 TB/s\nAll points lie on a mathematical roof; none are measured kernel results.",
                     loc="left", fontsize=13, pad=17)
        fig.savefig(target / "roofline_model.svg")
        fig.savefig(target / "roofline_model.png", dpi=160)
        plt.close(fig)
    print(target / "roofline_model.svg")


if __name__ == "__main__":
    main()
