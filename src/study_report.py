"""Summarise outputs/study_runs.jsonl: mean and 95% CI per arm, paired contrasts.

    python -m src.study_report
"""

import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from src.plots import CNN_COLOR, QNN_COLOR
from src.plots_findings import GRID, INK, MUTED, SURFACE, _style

RUNS = "outputs/study_runs.jsonl"
OUT = "outputs/study_summary.json"

ARM_LABELS = {
    "cnn_original":       "CNN, original (16 ch, big head)",
    "cnn_matched":        "CNN, matched, trained",
    "cnn_matched_frozen": "CNN, matched, frozen random",
    "qnn_frozen":         "QNN, frozen random",
    "qnn_trained":        "QNN, trained",
}
ARM_ORDER = list(ARM_LABELS)

# The three questions, each a paired comparison across the same seeds/splits
CONTRASTS = [
    ("qnn_frozen", "cnn_matched_frozen",
     "Random quantum features vs random classical features"),
    ("qnn_trained", "cnn_matched",
     "Trained quantum layer vs trained classical layer (same shape)"),
    ("qnn_trained", "qnn_frozen",
     "Does training the circuit help?"),
]

METRICS = [
    ("balanced", "accuracy", "Balanced accuracy"),
    ("balanced", "roc_auc", "Balanced ROC-AUC"),
    ("imbalanced", "precision", "1:10 precision"),
    ("imbalanced", "avg_precision", "1:10 average precision"),
    ("imbalanced", "false_positives", "1:10 false positives"),
]


def ci(values):
    a = np.asarray(values, float)
    n = len(a)
    if n < 2:
        return float(a.mean()), float("nan")
    half = stats.t.ppf(0.975, n - 1) * a.std(ddof=1) / np.sqrt(n)
    return float(a.mean()), float(half)


def load():
    runs = defaultdict(dict)
    with open(RUNS) as f:
        for r in map(json.loads, f):
            runs[r["arm"]][r["seed"]] = r
    return runs


def summarise(runs):
    arms = {}
    for arm in ARM_ORDER:
        if arm not in runs:
            continue
        seeds = sorted(runs[arm])
        entry = {"label": ARM_LABELS[arm], "n_seeds": len(seeds), "seeds": seeds,
                 "trainable_params": runs[arm][seeds[0]]["trainable_params"],
                 "best_epochs": [runs[arm][s]["best_epoch"] for s in seeds]}
        for split, key, _ in METRICS:
            vals = [runs[arm][s][split][key] for s in seeds]
            m, h = ci(vals)
            entry[f"{split}.{key}"] = {"mean": m, "ci95": h, "values": vals}
        if arm == "qnn_trained":
            entry["circuit_weight_change_max"] = max(
                runs[arm][s].get("circuit_weight_change", 0.0) for s in seeds)
        arms[arm] = entry

    contrasts = []
    for a, b, question in CONTRASTS:
        if a not in runs or b not in runs:
            continue
        common = sorted(set(runs[a]) & set(runs[b]))
        row = {"a": a, "b": b, "question": question, "n_pairs": len(common)}
        for split, key, _ in METRICS:
            d = [runs[a][s][split][key] - runs[b][s][split][key] for s in common]
            m, h = ci(d)
            if len(d) >= 2 and np.std(d, ddof=1) > 0:
                p = float(stats.ttest_1samp(d, 0.0).pvalue)
            else:
                p = float("nan")
            row[f"{split}.{key}"] = {"mean_diff": m, "ci95": h, "p": p, "diffs": d}
        contrasts.append(row)

    return {"arms": arms, "contrasts": contrasts}


def plot(summary, save_path="outputs/fig_study.png"):
    panels = [("balanced.accuracy", "Balanced accuracy"),
              ("imbalanced.precision", "Precision at 1:10"),
              ("imbalanced.avg_precision", "Average precision at 1:10")]
    arms = [a for a in ARM_ORDER if a in summary["arms"]]
    colors = {"cnn_original": "#8a8a86", "cnn_matched": CNN_COLOR, "cnn_matched_frozen": CNN_COLOR,
              "qnn_frozen": QNN_COLOR, "qnn_trained": QNN_COLOR}
    hollow = {"cnn_matched_frozen", "qnn_frozen"}

    fig, axes = plt.subplots(1, 3, figsize=(13, 0.62 * len(arms) + 1.9), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (key, title) in zip(axes, panels):
        _style(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=1)
        for i, arm in enumerate(arms):
            y = len(arms) - 1 - i
            e = summary["arms"][arm][key]
            c = colors[arm]
            ax.scatter(e["values"], [y] * len(e["values"]), s=14, color=c, alpha=0.35, zorder=2,
                       edgecolor="none")
            ax.plot([e["mean"] - e["ci95"], e["mean"] + e["ci95"]], [y, y], color=c, lw=2,
                    solid_capstyle="round", zorder=3)
            face = SURFACE if arm in hollow else c
            ax.scatter([e["mean"]], [y], s=70, facecolor=face, edgecolor=c, linewidth=2, zorder=4)
            ax.annotate(f"{e['mean']:.3f}", (e["mean"], y), xytext=(0, 9),
                        textcoords="offset points", ha="center", va="bottom", fontsize=9, color=INK)
        ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold", color=INK)
        lo = min(min(summary["arms"][a][key]["values"]) for a in arms)
        ax.set_xlim(max(0, lo - 0.03), 1.0)
        ax.set_xticks([t for t in ax.get_xticks() if t <= 1.0 + 1e-9])
        ax.set_xlim(max(0, lo - 0.03), 1.0)
    axes[0].set_yticks(range(len(arms)), [ARM_LABELS[a] for a in reversed(arms)])
    for lbl in axes[0].get_yticklabels():
        lbl.set_color(INK)
    n = max(summary["arms"][a]["n_seeds"] for a in arms)
    fig.text(0.01, 0.01, f"Big dot = mean over {n} seeds, bar = 95% CI, small dots = individual seeds. "
             "Hollow = front end left at random init.", fontsize=9, color=MUTED)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(save_path, dpi=160, facecolor=SURFACE)
    print(f"saved {save_path}")


def main():
    summary = summarise(load())
    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved {OUT}")
    for arm, e in summary["arms"].items():
        a, p = e["balanced.accuracy"], e["imbalanced.precision"]
        print(f"  {arm:19s} n={e['n_seeds']} params={e['trainable_params']:>6,}  "
              f"acc {a['mean']:.4f} +/- {a['ci95']:.4f}  1:10 prec {p['mean']:.4f} +/- {p['ci95']:.4f}")
    for c in summary["contrasts"]:
        print(f"  {c['question']} (n={c['n_pairs']})")
        for split, key, name in METRICS:
            d = c[f"{split}.{key}"]
            print(f"      {name:24s} diff {d['mean_diff']:+.4f} +/- {d['ci95']:.4f}  p={d['p']:.3g}")
    plot(summary)


if __name__ == "__main__":
    main()
