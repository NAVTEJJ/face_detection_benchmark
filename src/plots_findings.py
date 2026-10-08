"""Figures for the two findings the original plots don't cover.

Reads the saved results instead of retraining, so the figures always match the
numbers in outputs/metrics.json and outputs/gate_g0.json exactly.

    python -m src.plots_findings
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.plots import CNN_COLOR, QNN_COLOR

SURFACE = "#fcfcfb"
INK = "#1f1f1e"
MUTED = "#6b6b68"
GRID = "#e6e6e3"
OLD_COLOR = "#b5b5b1"   # "before" reads as faint, "after" as solid
NEW_COLOR = "#3f3f3c"   # not a model colour: blue/orange already mean CNN/QNN


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=10)
    ax.grid(axis="y", color=GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)


def plot_imbalance(metrics, save_path="outputs/fig_imbalance.png"):
    """Precision falls at the deployment ratio; ROC-AUC does not notice."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    fig.patch.set_facecolor(SURFACE)

    panels = [
        ("precision", "Precision", "drops when background outnumbers faces"),
        ("roc_auc", "ROC-AUC", "barely moves - it ignores class balance"),
    ]
    models = [("cnn", "CNN", CNN_COLOR), ("qnn", "QNN", QNN_COLOR)]
    xs = [0, 1]

    for ax, (key, title, sub) in zip(axes, panels):
        _style(ax)
        vals = {m: [metrics[m][key], metrics[f"{m}_imbalanced"][key]] for m, _, _ in models}
        # when both models sit on top of each other, push the labels apart
        nudge = [8 if abs(vals["cnn"][i] - vals["qnn"][i]) < 0.04 else 0 for i in xs]
        for m, name, color in models:
            ys = vals[m]
            sign = 1 if m == "cnn" else -1
            ax.plot(xs, ys, color=color, linewidth=2, solid_capstyle="round", zorder=2)
            ax.scatter(xs, ys, s=64, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
            ax.annotate(f"{name}  {ys[1]:.3f}", (1, ys[1]), xytext=(10, sign * nudge[1]),
                        textcoords="offset points", va="center", fontsize=10, color=INK)
            ax.annotate(f"{ys[0]:.3f}", (0, ys[0]), xytext=(-10, sign * nudge[0]),
                        textcoords="offset points", va="center", ha="right",
                        fontsize=10, color=MUTED)
        ax.set_xticks(xs, ["Balanced 1:1", "Deployment 1:10"])
        ax.set_xlim(-0.45, 1.6)
        ax.set_ylim(0.5, 1.02)
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=INK, pad=22)
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=10, color=MUTED, va="bottom")

    n_pos = metrics["qnn_imbalanced"]["n_pos"]
    n_neg = metrics["qnn_imbalanced"]["n_neg"]
    fp_c = metrics["cnn_imbalanced"]["false_positives"]
    fp_q = metrics["qnn_imbalanced"]["false_positives"]
    fig.text(0.01, 0.01,
             f"1:10 set: {n_pos} test faces + {n_neg} background images held out of training. "
             f"False positives at 1:10: CNN {fp_c}, QNN {fp_q}.",
             fontsize=9, color=MUTED)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=160, facecolor=SURFACE)
    print(f"saved {save_path}")
    return fig


def plot_gate(gate, save_path="outputs/fig_gate_g0.png"):
    """Can one number per image tell faces from non-faces? Old vs new negatives."""
    rows = [
        ("sharpness_auc", "Sharpness\n(Laplacian variance)"),
        ("intensity_auc", "Mean brightness"),
    ]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=1, linestyle="-")

    for i, (key, label) in enumerate(rows):
        y = len(rows) - 1 - i
        old, new = gate["old"][key], gate["new"][key]
        ax.plot([old, new], [y, y], color=GRID, linewidth=2, zorder=1)
        ax.scatter([old], [y], s=80, color=OLD_COLOR, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.scatter([new], [y], s=80, color=NEW_COLOR, edgecolor=SURFACE, linewidth=2, zorder=3)
        for v, is_old in ((old, True), (new, False)):
            ax.annotate(f"{v:.2f}", (v, y), xytext=(0, -18), textcoords="offset points",
                        ha="center", fontsize=9.5, color=MUTED if is_old else INK)

    ax.set_yticks(range(len(rows)), [r[1] for r in reversed(rows)])
    ax.set_ylim(-0.6, len(rows) - 0.3)
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("AUC from that single number", color=MUTED, fontsize=10)

    ax.axvline(0.5, color=MUTED, linewidth=1)
    ax.text(0.5, len(rows) - 0.35, " chance", color=MUTED, fontsize=9, va="top")
    ax.axvspan(0.9, 1.0, color="#f3e3df", zorder=0)
    ax.text(0.95, len(rows) - 0.35, "fail zone\n(AUC ≥ 0.90)", color=MUTED,
            fontsize=9, va="top", ha="center")

    ax.scatter([], [], s=80, color=OLD_COLOR, label="Old negatives (4 vehicle classes)")
    ax.scatter([], [], s=80, color=NEW_COLOR, label="New negatives (all 10 classes)")
    ax.legend(loc="lower right", frameon=False, fontsize=9.5, labelcolor=INK)

    ax.set_title("Gate G0: no single image statistic separates faces from non-faces",
                 loc="left", fontsize=12.5, fontweight="bold", color=INK, pad=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=160, facecolor=SURFACE)
    print(f"saved {save_path}")
    return fig


if __name__ == "__main__":
    with open("outputs/metrics.json") as f:
        metrics = json.load(f)
    with open("outputs/gate_g0.json") as f:
        gate = json.load(f)
    plot_imbalance(metrics)
    plot_gate(gate)
