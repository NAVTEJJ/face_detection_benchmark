"""Gate G0: can trivial image statistics separate the classes?

A face/non-face benchmark is only measuring face detection if a model actually
has to model faces. When the positive and negative images come from different
sources (LFW photographs vs CIFAR-10 thumbnails) they differ in ways that have
nothing to do with content: focus, compression, dynamic range, how much of the
frame the subject fills. A classifier can score very high by learning "which
dataset did this come from", and the headline accuracy then says nothing about
face detection at all.

G0 measures that directly. Three deliberately stupid baselines:

  sharpness  - variance of the Laplacian, one scalar per image, thresholded
  intensity  - mean pixel value, one scalar per image, thresholded
  pixels     - logistic regression straight on the 1024 raw pixels

If any of these score high, the split is separable without face modelling and
the benchmark is confounded. Low trivial-baseline scores do not prove the task
is good, but high ones prove it is bad, which is the useful direction.

Run standalone to compare negative-class sets:
    python -m src.baseline_g0
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE

_LAPLACIAN = torch.tensor(
    [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
).view(1, 1, 3, 3)


def laplacian_variance(imgs):
    """Classic focus measure: high for sharp images, low for blurred ones."""
    x = torch.as_tensor(imgs, dtype=torch.float32)
    if x.dim() == 3:
        x = x.unsqueeze(1)
    edges = F.conv2d(x, _LAPLACIAN, padding=1)
    return edges.var(dim=(1, 2, 3)).numpy()


def _scalar_gate(values, y):
    """Best achievable accuracy/AUC from thresholding one scalar."""
    auc = roc_auc_score(y, values)
    # a scalar cue can point either way; a threshold classifier may use both
    if auc < 0.5:
        auc, values = 1.0 - auc, -values
    order = np.argsort(values)
    v_sorted, y_sorted = values[order], y[order]
    # accuracy for every possible split point, vectorised
    pos_below = np.cumsum(y_sorted)
    neg_below = np.cumsum(1 - y_sorted)
    total_pos, total_neg = y_sorted.sum(), len(y_sorted) - y_sorted.sum()
    correct = (neg_below + (total_pos - pos_below))
    # also allow the degenerate "predict everything positive" threshold
    best = max(int(correct.max()), int(total_pos))
    return float(best / len(y_sorted)), float(auc)


def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float(abs(a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def run_gate(X, y, label=""):
    """X: [N,1,32,32] float in [0,1]. y: 1 = face, 0 = non-face."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y).astype(int)

    sharp = laplacian_variance(X)
    mean_intensity = X.reshape(len(X), -1).mean(axis=1)

    sharp_acc, sharp_auc = _scalar_gate(sharp, y)
    int_acc, int_auc = _scalar_gate(mean_intensity, y)
    d = cohens_d(sharp[y == 1], sharp[y == 0])

    flat = X.reshape(len(X), -1)
    Xtr, Xte, ytr, yte = train_test_split(
        flat, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    lr = LogisticRegression(max_iter=2000, n_jobs=-1)
    lr.fit(Xtr, ytr)
    p = lr.predict_proba(Xte)[:, 1]
    pix_acc = accuracy_score(yte, (p >= 0.5).astype(int))
    pix_auc = roc_auc_score(yte, p)

    return {
        "label":         label,
        "n":             len(y),
        "sharpness_acc": sharp_acc,
        "sharpness_auc": sharp_auc,
        "sharpness_d":   d,
        "intensity_acc": int_acc,
        "intensity_auc": int_auc,
        "pixel_acc":     float(pix_acc),
        "pixel_auc":     float(pix_auc),
    }


def verdict(r):
    """PASS means no trivial cue carries the split on its own.

    Only the scalar cues decide the verdict. Raw-pixel logistic accuracy is
    reported but deliberately excluded: a genuinely structured task can be
    linearly separable in pixel space and should be. tests/test_gate_g0.py
    constructs exactly that case - two classes with identical sharpness and
    identical mean intensity, differing only in spatial arrangement - where
    the pixel model reaches AUC 1.000 while both scalars sit at chance. Gating
    on the pixel number would reject that task, which would be wrong. A high
    pixel score next to high scalar scores is the pattern worth worrying
    about, and the scalars are what catch it.
    """
    worst_auc = max(r["sharpness_auc"], r["intensity_auc"])
    if worst_auc >= 0.90:
        return "FAIL", f"a single scalar reaches AUC {worst_auc:.3f}; the split is separable without face modelling"
    if worst_auc >= 0.80:
        return "WEAK", f"a single scalar reaches AUC {worst_auc:.3f}; trivial cues carry a large share of the signal"
    return "PASS", f"no single scalar exceeds AUC {worst_auc:.3f}"


def print_report(r):
    v, why = verdict(r)
    print(f"\n  {r['label']}  (n={r['n']})")
    print(f"    sharpness (Laplacian var) : acc {r['sharpness_acc']:.4f}  AUC {r['sharpness_auc']:.4f}  Cohen's d {r['sharpness_d']:.2f}")
    print(f"    mean intensity            : acc {r['intensity_acc']:.4f}  AUC {r['intensity_auc']:.4f}")
    print(f"    raw-pixel logistic        : acc {r['pixel_acc']:.4f}  AUC {r['pixel_auc']:.4f}")
    print(f"    GATE G0: {v} - {why}")
    return v


if __name__ == "__main__":
    import os
    from src import config
    from src.cifar import load_cifar10, to_grayscale
    from src.dataset import _load_lfw, _cache_dir

    faces = _load_lfw().numpy()
    n_pos = len(faces)
    all_data, all_labels = load_cifar10(_cache_dir())
    gray_all = to_grayscale(all_data)

    print("\n=== GATE G0: trivial-baseline separability ===")
    print("Same faces, two different negative sets.\n")

    results = []
    for name, classes in [
        ("OLD negatives: airplane/automobile/ship/truck", [0, 1, 8, 9]),
        ("NEW negatives: all 10 CIFAR-10 classes",        list(range(10))),
    ]:
        mask = np.isin(all_labels, classes)
        pool = gray_all[mask]
        rng = np.random.default_rng(config.RANDOM_STATE)
        idx = rng.choice(len(pool), size=n_pos, replace=False)
        negs = pool[idx][:, None, :, :]

        X = np.concatenate([faces, negs])
        y = np.concatenate([np.ones(n_pos, dtype=int), np.zeros(n_pos, dtype=int)])
        r = run_gate(X, y, label=name)
        print_report(r)
        results.append(r)

    old, new = results
    print("\n  Change from widening the negative set:")
    for k, nice in [("sharpness_auc", "sharpness AUC"),
                    ("pixel_auc", "raw-pixel AUC"),
                    ("sharpness_d", "sharpness Cohen's d")]:
        print(f"    {nice:22s} {old[k]:.3f} -> {new[k]:.3f}  ({new[k] - old[k]:+.3f})")

    os.makedirs("outputs", exist_ok=True)
    import json
    with open("outputs/gate_g0.json", "w") as f:
        json.dump({"old": old, "new": new}, f, indent=2)
    print("\n  Written to outputs/gate_g0.json")
