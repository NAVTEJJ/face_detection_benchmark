import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve

CNN_COLOR = "#4A90D9"
QNN_COLOR = "#E8543A"


def plot_training_curves(cnn_history, qnn_history, save_path="outputs/training_curves.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for ax, key, ylabel in zip(axes, ["loss", "acc"], ["Loss", "Accuracy"]):
        ax.plot(cnn_history[key], color=CNN_COLOR, linewidth=2, label="Classical CNN")
        ax.plot(qnn_history[key], color=QNN_COLOR, linewidth=2, label="Quanvolutional QNN")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"Training {ylabel}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle("Training Progress: CNN vs. QNN", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Training curves saved to {save_path}")
    return fig


def plot_roc_curves(cnn_metrics, qnn_metrics, save_path="outputs/roc_curves.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))

    for metrics, color, label in [
        (cnn_metrics, CNN_COLOR, "Classical CNN"),
        (qnn_metrics, QNN_COLOR, "Quanvolutional QNN"),
    ]:
        fpr, tpr, _ = roc_curve(metrics["y_true"], metrics["y_prob"])
        auc = metrics["roc_auc"]
        ax.plot(fpr, tpr, color=color, linewidth=2.2,
                label=f"{label}  (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], color="#AAAAAA", linestyle="--", linewidth=1.2, label="Chance")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve Comparison: Classical CNN vs. Quanvolutional QNN",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10.5)
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"ROC curves saved to {save_path}")
    return fig


def plot_confusion_matrices(cnn_metrics, qnn_metrics, save_path="outputs/confusion_matrices.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, metrics, title in zip(axes,
                                   [cnn_metrics, qnn_metrics],
                                   ["Classical CNN", "Quanvolutional QNN"]):
        disp = ConfusionMatrixDisplay(
            confusion_matrix=metrics["conf_matrix"],
            display_labels=["Non-Face", "Face"]
        )
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(title, fontsize=12, fontweight="bold")

    plt.suptitle("Confusion Matrices", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Confusion matrices saved to {save_path}")
    return fig


def plot_misclassified(metrics, X_test, y_test, save_path="outputs/misclassified.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    y_pred = metrics["y_pred"]
    y_true = y_test

    fp_idx = np.where((y_pred == 1) & (y_true == 0))[0]
    fn_idx = np.where((y_pred == 0) & (y_true == 1))[0]

    fp_idx = fp_idx[:5]
    fn_idx = fn_idx[:5]

    fig, axes = plt.subplots(2, 5, figsize=(13, 5.5))
    fig.suptitle("Misclassified Examples — Error Analysis", fontsize=13, fontweight="bold")

    for col, idx in enumerate(fp_idx):
        img = X_test[idx, 0] if len(X_test.shape) == 4 else X_test[idx]
        axes[0, col].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title("True: Non-Face\nPred: Face", fontsize=7.5, color="#CC0000")
        axes[0, col].axis("off")

    for col in range(len(fp_idx), 5):
        axes[0, col].axis("off")

    fn_img = None
    for col, idx in enumerate(fn_idx):
        fn_img = X_test[idx, 0] if len(X_test.shape) == 4 else X_test[idx]
        axes[1, col].imshow(fn_img, cmap="gray", vmin=0, vmax=1)
        axes[1, col].set_title("True: Face\nPred: Non-Face", fontsize=7.5, color="#CC0000")
        axes[1, col].axis("off")

    for col in range(len(fn_idx), 5):
        axes[1, col].axis("off")

    fig.text(0.01, 0.72, "False\nPositives", va="center", fontsize=10,
             fontweight="bold", color="#CC0000")
    fig.text(0.01, 0.28, "False\nNegatives", va="center", fontsize=10,
             fontweight="bold", color="#CC0000")

    plt.tight_layout(rect=[0.04, 0, 1, 0.95])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Misclassified examples saved to {save_path}")
    return fig


def plot_comparison_table(cnn_metrics, qnn_metrics, save_path="outputs/comparison_table.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    rows = [
        ["Test Accuracy",                f"{cnn_metrics['accuracy']:.4f}",       f"{qnn_metrics['accuracy']:.4f}"],
        ["Precision",                    f"{cnn_metrics['precision']:.4f}",      f"{qnn_metrics['precision']:.4f}"],
        ["Recall",                       f"{cnn_metrics['recall']:.4f}",         f"{qnn_metrics['recall']:.4f}"],
        ["F1-Score",                     f"{cnn_metrics['f1']:.4f}",             f"{qnn_metrics['f1']:.4f}"],
        ["ROC-AUC",                      f"{cnn_metrics['roc_auc']:.4f}",        f"{qnn_metrics['roc_auc']:.4f}"],
        ["Total Trainable Params",       f"{cnn_metrics['n_params']:,}",         f"{qnn_metrics['n_params']:,}"],
        ["Quantum Params (frozen)",      "—",                                    f"{qnn_metrics['q_params']}"],
        ["Qubit Count",                  "—",                                    "4"],
        ["Circuit Depth",                "—",                                    f"{qnn_metrics.get('circuit_depth', '?')}"],
        ["Training Time (s)",            f"{cnn_metrics['train_time']:.1f}",     f"{qnn_metrics['train_time']:.1f}"],
        ["Inference Latency / patch (ms)", f"{cnn_metrics['latency_mean']:.2f} +/- {cnn_metrics['latency_std']:.2f}",
                                           f"{qnn_metrics['latency_mean']:.2f} +/- {qnn_metrics['latency_std']:.2f}"],
        ["Inference / full 32x32 (ms)", f"{cnn_metrics['latency_full']:.2f}",   f"{qnn_metrics['latency_full']:.2f}"],
    ]
    col_labels = ["Metric", "Classical 12-Net CNN", "Quanvolutional QNN"]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.65)

    for j in range(3):
        table[0, j].set_facecolor("#2C3E50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    for i in range(1, len(rows) + 1):
        bg = "#F7F9FC" if i % 2 == 0 else "white"
        for j in range(3):
            table[i, j].set_facecolor(bg)
        table[i, 0].set_text_props(fontweight="bold", color="#333")
        try:
            cv = float(rows[i-1][1].split("+/-")[0].replace(",","").replace("—","").strip())
            qv = float(rows[i-1][2].split("+/-")[0].replace(",","").replace("—","").strip())
            if cv > qv:
                table[i, 1].set_facecolor("#DFF5E1")
            elif qv > cv:
                table[i, 2].set_facecolor("#DFF5E1")
        except (ValueError, IndexError):
            pass

    ax.set_title("Benchmark Comparison: Classical CNN vs. Quanvolutional QNN",
                 fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Comparison table saved to {save_path}")
    return fig


CNN_BLOCKS = [
    ("Input",          "1 × 32 × 32",    "#D0D0D0", False),
    ("Conv2d 3×3\n16 filters", "→ 16×32×32",   "#4A90D9", True),
    ("ReLU",           "",                "#F5A623", False),
    ("MaxPool 3×3\nstride 2",  "→ 16×16×16",   "#7ED321", False),
    ("Flatten",        "→ 4096",          "#9B9B9B", False),
    ("Dense 16\nReLU", "→ 16",            "#BD10E0", False),
    ("Dense 2\n(Output)",      "→ 2 logits",   "#417505", False),
]

QNN_BLOCKS = [
    ("Input",          "1 × 32 × 32",    "#D0D0D0", False),
    ("4-Qubit\nQuanv 2×2", "→ 4×16×16", "#E8543A", True),
    ("ReLU",           "",               "#F5A623", False),
    ("MaxPool 3×3\nstride 2", "→ 4×8×8", "#7ED321", False),
    ("Flatten",        "→ 256",          "#9B9B9B", False),
    ("Dense 16\nReLU", "→ 16",           "#BD10E0", False),
    ("Dense 2\n(Output)",     "→ 2 logits",  "#417505", False),
]


def _draw_column(ax, blocks, x_center, cnn_params=None, qnn_info=None):
    block_w = 0.36
    block_heights = [0.055, 0.11, 0.055, 0.09, 0.055, 0.09, 0.07]
    gap = 0.025
    y = 0.93

    rects = []
    for i, ((label, shape, color, is_swap), bh) in enumerate(zip(blocks, block_heights)):
        left = x_center - block_w / 2
        rect = mpatches.FancyBboxPatch(
            (left, y - bh), block_w, bh,
            boxstyle="round,pad=0.008",
            linewidth=3.0 if is_swap else 1.0,
            edgecolor="#CC0000" if is_swap else "#555555",
            facecolor=color,
            zorder=3
        )
        ax.add_patch(rect)

        text_y = y - bh / 2
        ax.text(x_center, text_y + (0.012 if shape else 0),
                label, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color="white", zorder=4)
        if shape:
            ax.text(x_center, text_y - 0.018,
                    shape, ha="center", va="center",
                    fontsize=7.5, color="white", alpha=0.85, zorder=4)

        rects.append((left, y - bh, block_w, bh, is_swap, y - bh / 2))

        if i < len(blocks) - 1:
            ax.annotate("", xy=(x_center, y - bh - gap + 0.003),
                        xytext=(x_center, y - bh),
                        arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.2), zorder=2)
        y -= bh + gap

    if cnn_params:
        ax.text(x_center, y - 0.01, f"Params: {cnn_params:,}",
                ha="center", fontsize=8.5, color="#333", style="italic")
    if qnn_info:
        ax.text(x_center, y - 0.01, qnn_info,
                ha="center", fontsize=8, color="#333", style="italic")

    return rects


def generate_arch_diagram(cnn_params=None, qnn_params=None, circuit_depth=None, save_path="outputs/architecture_diagram.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.text(0.5, 0.985,
            "Controlled Experiment: Conv  →  Quantum Conv  Layer Swap",
            ha="center", va="top", fontsize=14, fontweight="bold", color="#1a1a1a")

    ax.text(0.27, 0.945, "Classical 12-Net CNN",
            ha="center", fontsize=11.5, fontweight="bold", color="#4A90D9")
    ax.text(0.73, 0.945, "Quanvolutional QNN",
            ha="center", fontsize=11.5, fontweight="bold", color="#E8543A")

    ax.axvline(0.5, ymin=0.03, ymax=0.95,
               color="#AAAAAA", linestyle="--", linewidth=1.2, alpha=0.7)

    qnn_info = None
    if qnn_params is not None:
        depth_str = str(circuit_depth) if circuit_depth is not None else "?"
        qnn_info = f"Params: {qnn_params:,}  |  Qubits: 4  |  Depth: {depth_str}  |  Frozen quantum: 24"

    cnn_rects = _draw_column(ax, CNN_BLOCKS, x_center=0.27, cnn_params=cnn_params)
    qnn_rects = _draw_column(ax, QNN_BLOCKS, x_center=0.73, qnn_info=qnn_info)

    cnn_swap = [r for r in cnn_rects if r[4]][0]
    qnn_swap = [r for r in qnn_rects if r[4]][0]

    cnn_cx = 0.27 + cnn_swap[2] / 2
    qnn_cx = 0.73 - qnn_swap[2] / 2

    ax.annotate(
        "", xy=(qnn_cx, qnn_swap[5]), xytext=(0.27 + cnn_swap[2] / 2, cnn_swap[5]),
        arrowprops=dict(
            arrowstyle="<->", color="#CC0000", lw=1.8,
            linestyle="dashed",
            connectionstyle="arc3,rad=0.0"
        ), zorder=5
    )
    ax.text(0.5, (cnn_swap[5] + qnn_swap[5]) / 2 + 0.025,
            "Layer Swap", ha="center", fontsize=9, color="#CC0000",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#CC0000", linewidth=1.2))

    legend_handles = [
        mpatches.Patch(facecolor="#4A90D9", label="Classical Conv (swapped out)"),
        mpatches.Patch(facecolor="#E8543A", label="Quantum Quanv (swapped in)"),
        mpatches.Patch(facecolor="#7ED321", label="Shared layers (identical)"),
    ]
    ax.legend(handles=legend_handles, loc="lower center",
              ncol=3, fontsize=8.5, frameon=True,
              bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Architecture diagram saved to {save_path}")
    return fig
