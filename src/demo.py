"""Helpers behind notebooks/quantum_face_simulator.ipynb.

Kept out of the notebook so its cells stay short and readable. Everything here
reuses the study's own code paths (same split, same models, same circuit), so
what the notebook shows is what the study measured.
"""

import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from src.config import IMBALANCE_RATIO, N_LAYERS, N_QUBITS, TEST_SPLIT, VAL_SPLIT
from src.dataset import _load_cifar_negatives, _load_lfw
from src.models.classical_12net import Classical12Net
from src.models.torch_quanv import TorchQuanvNet, _cnot, _rx, _ry, _rz

ARMS = ["cnn_original", "cnn_matched", "cnn_matched_frozen", "qnn_frozen", "qnn_trained"]
LABELS = {
    "cnn_original": "CNN, original (big head)",
    "cnn_matched": "CNN, matched, trained",
    "cnn_matched_frozen": "CNN, matched, random",
    "qnn_frozen": "QNN, random circuit",
    "qnn_trained": "QNN, trained circuit",
}
COLORS = {
    "cnn_original": "#8a8a86", "cnn_matched": "#4A90D9", "cnn_matched_frozen": "#9cc3ec",
    "qnn_frozen": "#f2a08f", "qnn_trained": "#E8543A",
}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------- data

def load_demo_data(seed=0):
    """Identical to study.py's data handling for one seed."""
    faces = _load_lfw().numpy()
    n_pos = len(faces)
    n_test_faces = int(round(n_pos * TEST_SPLIT))
    negs, pool = _load_cifar_negatives(n_pos, n_extra=(n_test_faces + 1) * IMBALANCE_RATIO)
    X = np.concatenate([faces, negs.numpy()]).astype(np.float32)
    y = np.concatenate([np.ones(n_pos, int), np.zeros(n_pos, int)])

    idx = np.arange(len(X))
    tr, te = train_test_split(idx, test_size=TEST_SPLIT, random_state=seed, stratify=y)
    tr, va = train_test_split(tr, test_size=VAL_SPLIT, random_state=seed, stratify=y[tr])
    return {
        "X_train": X[tr], "y_train": y[tr],
        "X_val": X[va], "y_val": y[va],
        "X_test": X[te], "y_test": y[te],
        "background": pool.numpy().astype(np.float32),
    }


# ---------------------------------------------------------------- models

def build_model(arm):
    from study import MatchedCNN
    return {
        "cnn_original": lambda: Classical12Net(),
        "cnn_matched": lambda: MatchedCNN(),
        "cnn_matched_frozen": lambda: MatchedCNN(frozen=True),
        "qnn_frozen": lambda: TorchQuanvNet(trainable=False),
        "qnn_trained": lambda: TorchQuanvNet(trainable=True),
    }[arm]()


def load_models(ckpt_dir=None):
    ckpt_dir = ckpt_dir or os.path.join(ROOT, "checkpoints")
    models = {}
    for arm in ARMS:
        m = build_model(arm)
        m.load_state_dict(torch.load(os.path.join(ckpt_dir, f"{arm}.pt"), weights_only=True))
        m.eval()
        models[arm] = m
    return models


def predict_proba(model, X, batch=256):
    X = torch.as_tensor(np.asarray(X, dtype=np.float32))
    if X.dim() == 3:
        X = X.unsqueeze(1)
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(F.softmax(model(X[i:i + batch]), dim=1)[:, 1])
    return torch.cat(out).numpy()


def front_end_maps(model, arm, image):
    """The 4 (or 16) feature maps the first layer produces for one image."""
    x = torch.as_tensor(np.asarray(image, dtype=np.float32)).reshape(1, 1, 32, 32)
    with torch.no_grad():
        if arm.startswith("qnn"):
            return model.features(x)[0].numpy()
        if arm.startswith("cnn_matched"):
            return torch.tanh(model.conv(x))[0].numpy()
        return model.conv_block[0](x)[0].numpy()


# ---------------------------------------------------------------- circuit

def circuit_state(pixels, weights):
    """Statevector of the quanvolution circuit for one 2x2 patch.

    Same gates as quanv_circuit / torch_quanv.circuit_expvals. Returns the 16
    basis-state probabilities (wire 0 = leftmost bit) and the 4 Pauli-Z values.
    """
    p = torch.as_tensor(np.asarray(pixels, dtype=np.float32)).reshape(1, N_QUBITS)
    w = torch.as_tensor(np.asarray(weights, dtype=np.float32)).reshape(N_LAYERS, N_QUBITS, 3)
    state = torch.zeros((1,) + (2,) * N_QUBITS, dtype=torch.complex64)
    state[(0,) + (0,) * N_QUBITS] = 1.0
    enc = (p * math.pi).to(torch.complex64)
    shape = (1,) + (1,) * (N_QUBITS - 1)
    for i in range(N_QUBITS):
        state = _ry(state, i, enc[:, i].reshape(shape))
    for layer in range(N_LAYERS):
        for q in range(N_QUBITS):
            state = _rx(state, q, w[layer, q, 0])
            state = _ry(state, q, w[layer, q, 1].to(torch.complex64))
            state = _rz(state, q, w[layer, q, 2])
        for q in range(N_QUBITS):
            state = _cnot(state, q, (q + 1) % N_QUBITS)
    probs = (state.real ** 2 + state.imag ** 2).reshape(-1).numpy().astype(float)
    probs = probs / probs.sum()
    bits = np.array([[(k >> (N_QUBITS - 1 - q)) & 1 for q in range(N_QUBITS)] for k in range(2 ** N_QUBITS)])
    z = ((1 - 2 * bits) * probs[:, None]).sum(0)
    return probs, z, bits


def sample_z(probs, bits, shots, rng):
    """Pauli-Z estimates from a finite number of measurement shots."""
    counts = rng.multinomial(shots, probs)
    return ((1 - 2 * bits) * counts[:, None]).sum(0) / shots, counts


def draw_circuit(pixels, weights):
    """Matplotlib drawing of the circuit, via PennyLane."""
    import pennylane as qml
    from src.models.quanv_12net import quanv_circuit
    fig, _ = qml.draw_mpl(quanv_circuit, decimals=2, style="pennylane")(
        torch.tensor(np.asarray(pixels, dtype=np.float32)),
        torch.tensor(np.asarray(weights, dtype=np.float32)))
    return fig


# ---------------------------------------------------------------- deployment

def deployment_metrics(p_faces, p_background, ratio, threshold):
    """Expected detector behaviour when background outnumbers faces ratio:1.

    Recall comes from the faces and the false-positive rate from the background
    pool, both measured at `threshold`. Precision at a given ratio follows from
    those two: TP / (TP + FP) = recall / (recall + ratio * FPR).
    """
    recall = float((p_faces >= threshold).mean())
    fpr = float((p_background >= threshold).mean())
    precision = recall / (recall + ratio * fpr) if recall + ratio * fpr > 0 else 0.0
    return {"recall": recall, "fpr": fpr, "precision": precision,
            "false_alarms_per_100_faces": 100 * ratio * fpr}
