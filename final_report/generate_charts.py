"""
Generate all 4 charts for the Causal-MoE research paper.
Uses actual benchmark data from analysis_output.json.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Colour palette ───────────────────────────────────────────────────────────
C_BASE  = "#8c8c8c"   # neutral gray
C_V1    = "#2b5b84"   # deep academic blue
C_V2    = "#c94c4c"   # crimson red
C_BG    = "#FFFFFF"   # pure white background
C_PANEL = "#FFFFFF"   # pure white panel
C_TEXT  = "#000000"   # black text
C_GRID  = "#E0E0E0"   # light gray grid
C_ACCENT= "#006400"   # dark green accent

plt.rcParams.update({
    "figure.facecolor":  C_BG,
    "axes.facecolor":    C_PANEL,
    "axes.edgecolor":    "#333333",
    "axes.labelcolor":   C_TEXT,
    "axes.titlecolor":   C_TEXT,
    "xtick.color":       C_TEXT,
    "ytick.color":       C_TEXT,
    "grid.color":        C_GRID,
    "text.color":        C_TEXT,
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    14,
    "axes.labelsize":    12,
    "legend.facecolor":  C_PANEL,
    "legend.edgecolor":  "#CCCCCC",
    "legend.labelcolor": C_TEXT,
})

# ─── Data ─────────────────────────────────────────────────────────────────────
ACCURACY = {
    "CSQA":  {"Base": 82.06, "V1": 83.05, "V2": 78.38},
    "GSM8K": {"Base": 88.86, "V1": 68.08, "V2": 76.80},
}

TOKENS = {
    "CSQA Base":  155.4,
    "CSQA V1":     27.1,
    "CSQA V2":     33.4,
    "GSM8K Base": 208.9,
    "GSM8K V1":    55.0,
    "GSM8K V2":    75.8,
}

FLOPS = {"Base": 4.28, "V1": 7.67, "V2": 2.81}
RHO   = {"Base": 20.76, "V1": 8.87, "V2": 27.33}
GSM_ACC = {"Base": 88.86, "V1": 68.08, "V2": 76.80}

# ══════════════════════════════════════════════════════════════════════════════
# Chart A — Grouped Bar Chart: Accuracy comparison
# ══════════════════════════════════════════════════════════════════════════════
def chart_a():
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(C_BG)

    benchmarks = ["CommonsenseQA", "GSM8K"]
    models = ["Base", "V1", "V2"]
    colors = [C_BASE, C_V1, C_V2]

    x = np.arange(len(benchmarks))
    width = 0.24
    offsets = [-width, 0, width]

    for i, (model, color) in enumerate(zip(models, colors)):
        vals = [ACCURACY[b][model] for b in ["CSQA", "GSM8K"]]
        bars = ax.bar(x + offsets[i], vals, width,
                      label=model, color=color,
                      edgecolor="none", zorder=3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val:.1f}%",
                    ha="center", va="bottom",
                    fontsize=9.5, fontweight="bold", color=C_TEXT)

    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, fontsize=12)
    ax.set_ylim(55, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Chart A — Accuracy: Base vs Causal-MoE V1 vs V2", fontweight="bold", pad=14)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    legend = ax.legend(title="Model", loc="upper right", framealpha=0.85)
    legend.get_title().set_color(C_TEXT)

    # Annotate best V1 (CSQA) and best V2 (GSM8K)
    ax.annotate("V1 best on CSQA\n(+0.99 pp)",
                xy=(x[0] + offsets[1], ACCURACY["CSQA"]["V1"]),
                xytext=(x[0] + offsets[1] - 0.4, ACCURACY["CSQA"]["V1"] + 5),
                fontsize=8, color=C_V1,
                arrowprops=dict(arrowstyle="->", color=C_V1, lw=1.2))
    ax.annotate("V2 best MoE\non GSM8K",
                xy=(x[1] + offsets[2], ACCURACY["GSM8K"]["V2"]),
                xytext=(x[1] + offsets[2] + 0.15, ACCURACY["GSM8K"]["V2"] + 6),
                fontsize=8, color=C_V2,
                arrowprops=dict(arrowstyle="->", color=C_V2, lw=1.2))

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "chart_a.png")
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"Saved: {path}")

# ══════════════════════════════════════════════════════════════════════════════
# Chart B — Horizontal Bar Chart: Generated tokens
# ══════════════════════════════════════════════════════════════════════════════
def chart_b():
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(C_BG)

    labels = list(TOKENS.keys())
    vals   = list(TOKENS.values())
    bar_colors = [C_BASE, C_V1, C_V2, C_BASE, C_V1, C_V2]

    y = np.arange(len(labels))
    bars = ax.barh(y, vals, color=bar_colors, edgecolor="none", height=0.55, zorder=3)

    for bar, val, label in zip(bars, vals, labels):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f} tok", va="center", fontsize=10, fontweight="bold")
        # Add reduction % for V2 bars
        if "V2" in label:
            base_key = label.replace("V2", "Base")
            base_val = TOKENS[base_key]
            reduction = (base_val - val) / base_val * 100
            ax.text(bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
                    f"−{reduction:.1f}%", va="center", ha="center",
                    fontsize=9, color="white", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Avg. Generated Tokens per Query")
    ax.set_xlim(0, 260)
    ax.set_title("Chart B — Generated Token Count: MoE vs Base Model", fontweight="bold", pad=14)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    patches = [
        mpatches.Patch(color=C_BASE, label="Base"),
        mpatches.Patch(color=C_V1,   label="V1"),
        mpatches.Patch(color=C_V2,   label="V2"),
    ]
    ax.legend(handles=patches, loc="lower right", framealpha=0.85)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "chart_b.png")
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"Saved: {path}")

# ══════════════════════════════════════════════════════════════════════════════
# Chart C — Dual-axis line plot: FLOP Cost vs Accuracy (GSM8K)
# ══════════════════════════════════════════════════════════════════════════════
def chart_c():
    fig, ax1 = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(C_BG)
    ax2 = ax1.twinx()
    ax2.set_facecolor(C_PANEL)

    models_order = ["Base", "V1", "V2"]
    x_pos = [0, 1, 2]
    flops  = [FLOPS[m] for m in models_order]
    accs   = [GSM_ACC[m] for m in models_order]

    # Shade region where V2 FLOPs < Base FLOPs
    ax1.axhspan(0, FLOPS["Base"], alpha=0.08, color=C_ACCENT, zorder=0)
    ax1.axhline(FLOPS["Base"], color=C_ACCENT, linestyle=":", linewidth=1.2,
                label=f"Base FLOP baseline ({FLOPS['Base']}T)")

    # FLOP line
    ax1.plot(x_pos, flops, color=C_V2, marker="o", linewidth=2.5,
             markersize=9, zorder=4, label="Est. FLOPs (T)")
    for xi, yf, model in zip(x_pos, flops, models_order):
        ax1.annotate(f"{yf}T", (xi, yf), textcoords="offset points",
                     xytext=(0, 10), ha="center", color=C_V2, fontsize=10, fontweight="bold")

    # Accuracy line
    ax2.plot(x_pos, accs, color=C_V1, marker="s", linewidth=2.5,
             markersize=9, zorder=4, linestyle="--", label="GSM8K Accuracy (%)")
    for xi, ya, model in zip(x_pos, accs, models_order):
        ax2.annotate(f"{ya:.1f}%", (xi, ya), textcoords="offset points",
                     xytext=(0, -18), ha="center", color=C_V1, fontsize=10, fontweight="bold")

    # Green band label
    ax1.text(0.5, 2.3, "← V2 achieves 34% FLOP reduction\nbelow base model",
             color=C_ACCENT, fontsize=9, ha="center",
             bbox=dict(boxstyle="round,pad=0.4", facecolor=C_PANEL, edgecolor=C_ACCENT, alpha=0.8))

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(models_order, fontsize=12)
    ax1.set_ylabel("Est. Total FLOPs (Trillions)", color=C_V2)
    ax1.tick_params(axis="y", labelcolor=C_V2)
    ax1.set_ylim(0, 11)
    ax1.set_xlim(-0.4, 2.4)

    ax2.set_ylabel("GSM8K Accuracy (%)", color=C_V1)
    ax2.tick_params(axis="y", labelcolor=C_V1)
    ax2.set_ylim(40, 110)

    ax1.set_title("Chart C — FLOP Cost vs Accuracy Trade-off (GSM8K)", fontweight="bold", pad=14)
    ax1.xaxis.grid(True, linestyle="--", linewidth=0.6, zorder=0)
    ax1.set_axisbelow(True)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="upper center", framealpha=0.85, fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "chart_c.png")
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"Saved: {path}")

# ══════════════════════════════════════════════════════════════════════════════
# Chart D — Scatter/Bubble: Accuracy-FLOP Pareto Frontier
# ══════════════════════════════════════════════════════════════════════════════
def chart_d():
    fig, ax = plt.subplots(figsize=(9, 6.5))
    fig.patch.set_facecolor(C_BG)

    models = ["Base", "V1", "V2"]
    colors_map = {"Base": C_BASE, "V1": C_V1, "V2": C_V2}
    x_flops  = {m: FLOPS[m] for m in models}
    y_accs   = {m: GSM_ACC[m] for m in models}
    rho_vals = {m: RHO[m] for m in models}

    # Iso-efficiency line at ρ_R = 20.76 (base level)
    flop_range = np.linspace(2.0, 9.0, 200)
    iso_acc    = 20.76 * flop_range
    ax.plot(flop_range, iso_acc, color=C_BASE, linestyle="--", linewidth=1.2,
            alpha=0.6, label="Iso-efficiency (ρ_R = 20.76, Base level)")
    ax.text(7.2, 20.76 * 7.2 + 1.5, "ρ_R = 20.76", color=C_BASE,
            fontsize=8.5, alpha=0.8, rotation=22)

    for m in models:
        size = (rho_vals[m] / 8) ** 2 * 900  # bubble area proportional to ρ_R²
        sc = ax.scatter(x_flops[m], y_accs[m],
                        s=size, color=colors_map[m],
                        edgecolors="white", linewidths=1.5,
                        zorder=5, alpha=0.93)
        offsets = {"Base": (0.15, -4.5), "V1": (-0.45, 2.5), "V2": (0.15, 2.5)}
        dx, dy = offsets[m]
        ax.annotate(f"{m}\n({x_flops[m]:.2f}T FLOP, {y_accs[m]:.1f}%)\nρ_R = {rho_vals[m]:.2f}",
                    xy=(x_flops[m], y_accs[m]),
                    xytext=(x_flops[m] + dx, y_accs[m] + dy),
                    fontsize=9.5, color=colors_map[m], fontweight="bold",
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.35",
                              facecolor=C_PANEL, edgecolor=colors_map[m], alpha=0.85))

    # Arrow from Base to V2 showing direction of improvement
    ax.annotate("", xy=(x_flops["V2"], y_accs["V2"]),
                xytext=(x_flops["Base"], y_accs["Base"]),
                arrowprops=dict(arrowstyle="->", color=C_ACCENT,
                                lw=1.8, connectionstyle="arc3,rad=0.2"))
    ax.text(3.6, 84, "Higher ρ_R\n(better efficiency)", color=C_ACCENT,
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=C_PANEL, edgecolor=C_ACCENT, alpha=0.8))

    ax.set_xlabel("Est. Total FLOPs (Trillions)", fontsize=12)
    ax.set_ylabel("GSM8K Accuracy (%)", fontsize=12)
    ax.set_title("Chart D — Accuracy-FLOP Pareto Frontier (bubble size ∝ Reasoning Density ρ_R)",
                 fontweight="bold", pad=14)
    ax.set_xlim(1.5, 9.5)
    ax.set_ylim(55, 100)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.6, zorder=0)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    legend_handles = [
        mpatches.Patch(color=C_BASE, label=f"Base  (ρ_R = {RHO['Base']:.2f})"),
        mpatches.Patch(color=C_V1,   label=f"V1    (ρ_R = {RHO['V1']:.2f})"),
        mpatches.Patch(color=C_V2,   label=f"V2    (ρ_R = {RHO['V2']:.2f}) ★ best"),
        plt.Line2D([0], [0], color=C_BASE, linestyle="--", label="Iso-eff. line (Base ρ_R)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", framealpha=0.85, fontsize=9.5)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "chart_d.png")
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"Saved: {path}")

# ─── Run all ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    chart_a()
    chart_b()
    chart_c()
    chart_d()
    print("\nAll 4 charts generated successfully.")
