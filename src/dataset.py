import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.cifar import load_cifar10, to_grayscale, CLASS_NAMES
from src.config import (
    RANDOM_STATE, IMAGE_SIZE, BATCH_SIZE,
    LFW_MIN_FACES, CIFAR_NEGATIVE_CLASSES, TEST_SPLIT,
    IMBALANCE_RATIO, IMBALANCE_MAX_FACES,
)


def _cache_dir():
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cache")


def _load_lfw():
    print("Loading LFW faces...")
    try:
        lfw = fetch_lfw_people(min_faces_per_person=LFW_MIN_FACES, resize=1.0, color=False)
    except Exception as e:
        raise RuntimeError(f"LFW download failed: {e}")

    imgs = torch.tensor(lfw.images, dtype=torch.float32).unsqueeze(1)
    imgs = F.interpolate(imgs, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
    print(f"  LFW: {imgs.shape[0]} face images -> resized to {IMAGE_SIZE}x{IMAGE_SIZE}")
    return imgs


def _load_cifar_negatives(n_needed, n_extra=0):
    """Sample negatives from CIFAR-10.

    Returns (main, extra) where `extra` is disjoint from `main` and is reserved
    for the imbalanced evaluation, so those background patches appear in
    neither the training nor the balanced test split.
    """
    print("Loading CIFAR-10 negatives...")
    all_data, all_labels = load_cifar10(_cache_dir())

    mask = np.isin(all_labels, CIFAR_NEGATIVE_CLASSES)
    neg_imgs = all_data[mask]
    neg_labels = all_labels[mask]
    neg_gray = to_grayscale(neg_imgs)

    total = n_needed + n_extra
    if total > len(neg_gray):
        raise ValueError(
            f"Need {total} negatives but only {len(neg_gray)} available "
            f"from classes {CIFAR_NEGATIVE_CLASSES}."
        )

    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(neg_gray), size=total, replace=False)
    main_idx, extra_idx = idx[:n_needed], idx[n_needed:]

    used = sorted(set(CIFAR_NEGATIVE_CLASSES))
    names = ", ".join(CLASS_NAMES[c] for c in used)
    counts = np.bincount(neg_labels[main_idx], minlength=10)
    print(f"  CIFAR-10 negatives: {n_needed} sampled from {mask.sum()} available")
    print(f"    classes ({len(used)}): {names}")
    print(f"    per-class in train+test negatives: "
          + ", ".join(f"{CLASS_NAMES[c]}={counts[c]}" for c in used))
    if n_extra:
        print(f"    held-out pool for imbalanced eval: {n_extra} (disjoint)")

    def _as_tensor(a):
        return torch.tensor(a, dtype=torch.float32).unsqueeze(1)

    return _as_tensor(neg_gray[main_idx]), _as_tensor(neg_gray[extra_idx])


def _balance_chart(n_pos, n_neg):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(["Face (LFW)", "Non-Face (CIFAR)"], [n_pos, n_neg],
           color=["#4A90D9", "#E8543A"], edgecolor="white", linewidth=1.2)
    ax.set_title("Dataset Class Balance", fontsize=13, fontweight="bold")
    ax.set_ylabel("Samples")
    for i, v in enumerate([n_pos, n_neg]):
        ax.text(i, v + 20, str(v), ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(n_pos, n_neg) * 1.15)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def load_data():
    face_imgs  = _load_lfw()
    n_pos      = len(face_imgs)

    # Size the held-out pool from how many faces will survive into the test
    # split, capped so the quanvolutional feature cost stays bounded.
    n_test_faces = min(IMBALANCE_MAX_FACES, int(n_pos * TEST_SPLIT))
    n_extra      = n_test_faces * IMBALANCE_RATIO

    nonface_imgs, extra_negatives = _load_cifar_negatives(n_pos, n_extra=n_extra)
    n_neg = len(nonface_imgs)

    X = torch.cat([face_imgs, nonface_imgs], dim=0)
    y = torch.cat([torch.ones(n_pos, dtype=torch.long),
                   torch.zeros(n_neg, dtype=torch.long)], dim=0)

    X_np = X.numpy()
    y_np = y.numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X_np, y_np,
        test_size=TEST_SPLIT,
        random_state=RANDOM_STATE,
        stratify=y_np
    )

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_ds  = TensorDataset(torch.tensor(X_test),  torch.tensor(y_test))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    balance_fig = _balance_chart(n_pos, n_neg)

    print(f"\nDataset ready:")
    print(f"  Total:  {len(X_np)} samples ({n_pos} faces, {n_neg} non-faces)")
    print(f"  Train:  {len(X_train)} | Test: {len(X_test)}")

    metadata = {
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "extra_negatives": extra_negatives.numpy(),
        "n_imbalance_faces": n_test_faces,
        "class_balance_fig": balance_fig,
    }
    return train_loader, test_loader, metadata


def build_imbalanced_set(meta):
    """Assemble the 1:IMBALANCE_RATIO evaluation set.

    Faces come from the test split (never trained on); background comes from
    the held-out CIFAR pool (in neither split). Faces are subsampled to
    IMBALANCE_MAX_FACES so the negative pool, which is RATIO times larger,
    stays affordable to push through the quantum circuit.
    """
    X_test, y_test = meta["X_test"], meta["y_test"]
    rng = np.random.default_rng(RANDOM_STATE)

    face_idx = np.flatnonzero(y_test == 1)
    n_faces  = meta["n_imbalance_faces"]
    if len(face_idx) > n_faces:
        face_idx = rng.choice(face_idx, size=n_faces, replace=False)

    faces = X_test[face_idx]
    bg    = meta["extra_negatives"][: len(face_idx) * IMBALANCE_RATIO]

    X = np.concatenate([faces, bg])
    y = np.concatenate([np.ones(len(faces), dtype=np.int64),
                        np.zeros(len(bg), dtype=np.int64)])

    order = rng.permutation(len(X))
    return X[order], y[order]
