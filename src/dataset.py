import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from torchvision.datasets import CIFAR10
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (
    RANDOM_STATE, IMAGE_SIZE, BATCH_SIZE,
    LFW_MIN_FACES, CIFAR_NEGATIVE_CLASSES, TEST_SPLIT
)


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


def _load_cifar_negatives(n_needed):
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cache")
    print("Loading CIFAR-10 negatives...")
    try:
        cifar_train = CIFAR10(root=cache_dir, train=True, download=True)
        cifar_test  = CIFAR10(root=cache_dir, train=False, download=True)
    except Exception as e:
        raise RuntimeError(f"CIFAR-10 download failed: {e}")

    all_data   = np.concatenate([np.array(cifar_train.data), np.array(cifar_test.data)])
    all_labels = np.array(cifar_train.targets + cifar_test.targets)

    mask = np.isin(all_labels, CIFAR_NEGATIVE_CLASSES)
    neg_imgs = all_data[mask]

    # BT.601 luma grayscale conversion
    neg_gray = (
        0.299 * neg_imgs[:, :, :, 0].astype(np.float32) +
        0.587 * neg_imgs[:, :, :, 1].astype(np.float32) +
        0.114 * neg_imgs[:, :, :, 2].astype(np.float32)
    ) / 255.0

    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(neg_gray), size=n_needed, replace=False)
    neg_gray = neg_gray[idx]

    neg_tensor = torch.tensor(neg_gray, dtype=torch.float32).unsqueeze(1)
    print(f"  CIFAR-10 negatives: {n_needed} sampled from {mask.sum()} available")
    return neg_tensor


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
    nonface_imgs = _load_cifar_negatives(n_pos)
    n_neg      = len(nonface_imgs)

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
        "X_test": X_test,
        "y_test": y_test,
        "class_balance_fig": balance_fig,
    }
    return train_loader, test_loader, metadata
