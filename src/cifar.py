"""Minimal CIFAR-10 reader.

Reads the standard pickled batch files directly instead of going through
torchvision.datasets.CIFAR10. Two reasons: the cached batches are already
vendored under data/cache/, and torchvision's binary extension is tied to a
specific torch build (a mismatch raises "operator torchvision::nms does not
exist" at import time, before any dataset code runs). Nothing here needs
torchvision, so we drop the dependency.
"""

import os
import pickle
import tarfile
import urllib.request

import numpy as np

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
BATCH_DIR = "cifar-10-batches-py"
TRAIN_BATCHES = [f"data_batch_{i}" for i in range(1, 6)]
TEST_BATCH = "test_batch"

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def _ensure_batches(cache_dir):
    batch_dir = os.path.join(cache_dir, BATCH_DIR)
    if all(os.path.exists(os.path.join(batch_dir, b)) for b in TRAIN_BATCHES + [TEST_BATCH]):
        return batch_dir

    os.makedirs(cache_dir, exist_ok=True)
    archive = os.path.join(cache_dir, "cifar-10-python.tar.gz")
    if not os.path.exists(archive):
        print(f"  downloading CIFAR-10 from {CIFAR_URL} ...")
        urllib.request.urlretrieve(CIFAR_URL, archive)

    print("  extracting CIFAR-10 archive ...")
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(cache_dir)
    return batch_dir


def _read_batch(path):
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    # stored as (N, 3072) flat uint8, channel-major: 1024 R, then G, then B
    data = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.array(d[b"labels"], dtype=np.int64)
    return data, labels


def load_cifar10(cache_dir):
    """Return (images uint8 [N,32,32,3], labels int64 [N]) for train+test combined."""
    batch_dir = _ensure_batches(cache_dir)
    datas, labels = [], []
    for name in TRAIN_BATCHES + [TEST_BATCH]:
        d, l = _read_batch(os.path.join(batch_dir, name))
        datas.append(d)
        labels.append(l)
    return np.concatenate(datas), np.concatenate(labels)


def to_grayscale(imgs_uint8):
    """BT.601 luma, scaled to [0, 1]."""
    return (
        0.299 * imgs_uint8[:, :, :, 0].astype(np.float32)
        + 0.587 * imgs_uint8[:, :, :, 1].astype(np.float32)
        + 0.114 * imgs_uint8[:, :, :, 2].astype(np.float32)
    ) / 255.0
