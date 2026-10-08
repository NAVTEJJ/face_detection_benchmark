"""Controlled multi-seed study.

Fixes the comparison problems in the single-run benchmark (main.py):

  * capacity: the matched CNN has exactly the QNN's shape - a 2x2, stride-2,
    4-channel front end feeding the same pool and the same 256 -> 16 -> 2 head
  * training: the quantum circuit is trained end-to-end in one arm, with
    gradients checked against PennyLane (src/models/torch_quanv.py)
  * fairness: same epochs, optimiser and batch size for every model; each
    keeps the epoch with the lowest validation loss
  * variance: 5 seeds, each a different train/val/test split and init;
    results are mean and 95% CI over seeds

Arms
  cnn_original        the original CNN (16 ch 3x3, 4,096-input head), for reference
  cnn_matched         2x2/s2 conv, 4 ch, tanh, trained          (20 front-end params)
  cnn_matched_frozen  same conv left at its random init          (classical random features)
  qnn_frozen          quantum circuit at its random init          (as in main.py)
  qnn_trained         quantum circuit trained end-to-end          (24 front-end params)

Results are appended to outputs/study_runs.jsonl after every run, so the
study can be stopped and resumed; finished (seed, arm) pairs are skipped.

    python study.py
"""

import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from src.config import (BATCH_SIZE, IMBALANCE_RATIO, LEARNING_RATE, STUDY_EPOCHS,
                        STUDY_SEEDS, TEST_SPLIT, VAL_SPLIT)
from src.dataset import _load_cifar_negatives, _load_lfw
from src.models.classical_12net import Classical12Net
from src.models.torch_quanv import (TorchQuanvNet, validate_against_pennylane,
                                    validate_gradients)

RUNS = "outputs/study_runs.jsonl"
ARMS = ["cnn_original", "cnn_matched", "cnn_matched_frozen", "qnn_frozen", "qnn_trained"]

torch.set_num_threads(os.cpu_count() or 4)


class MatchedCNN(nn.Module):
    """Classical counterpart with the QNN's exact geometry.

    2x2 kernel, stride 2, 4 output channels: one output per patch per channel,
    like the 4 qubits. tanh keeps outputs in [-1, 1], the range of a Pauli-Z
    expectation. Then the same pool and head as the QNN.
    """

    def __init__(self, frozen=False):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        if frozen:
            for p in self.conv.parameters():
                p.requires_grad_(False)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(4 * 8 * 8, 16), nn.ReLU(), nn.Linear(16, 2),
        )

    def forward(self, x):
        return self.classifier(self.pool(torch.tanh(self.conv(x))))


class FeatureHead(nn.Module):
    """Trains only the head on precomputed (frozen) quantum features."""

    def __init__(self, qnet):
        super().__init__()
        self.qnet = qnet

    def forward(self, feats):
        return self.qnet.forward_from_features(feats)


def n_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def predict(model, X):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            out.append(F.softmax(model(X[i:i + 256]), dim=1)[:, 1])
    return torch.cat(out).numpy()


def fit(model, Xtr, ytr, Xval, yval, seed):
    """Same loop for every arm: Adam, fixed epochs, keep the best-val-loss epoch."""
    g = torch.Generator().manual_seed(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=LEARNING_RATE)
    best, best_loss, best_epoch, history = None, float("inf"), -1, []

    for epoch in range(STUDY_EPOCHS):
        model.train()
        order = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(Xtr), BATCH_SIZE):
            idx = order[i:i + BATCH_SIZE]
            opt.zero_grad()
            F.cross_entropy(model(Xtr[idx]), ytr[idx]).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vloss = sum(F.cross_entropy(model(Xval[i:i + 256]), yval[i:i + 256], reduction="sum").item()
                        for i in range(0, len(Xval), 256)) / len(Xval)
        history.append(vloss)
        if vloss < best_loss:
            best_loss, best_epoch, best = vloss, epoch + 1, copy.deepcopy(model.state_dict())

    model.load_state_dict(best)
    return best_epoch, best_loss, history


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    fp = int(((pred == 1) & (y == 0)).sum())
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, p),
        "avg_precision": average_precision_score(y, p),
        "false_positives": fp,
        "fpr": fp / max(1, int((y == 0).sum())),
    }


def done_pairs():
    if not os.path.exists(RUNS):
        return set()
    with open(RUNS) as f:
        return {(r["seed"], r["arm"]) for r in map(json.loads, f)}


def main():
    os.makedirs("outputs", exist_ok=True)

    v, gd = validate_against_pennylane(), validate_gradients()
    print(f"torch circuit vs PennyLane: values {v:.1e}, gradients {gd:.1e}")
    assert v < 1e-5 and gd < 1e-5, "torch circuit disagrees with PennyLane"

    faces = _load_lfw().numpy()
    n_pos = len(faces)
    n_test_faces = int(round(n_pos * TEST_SPLIT))
    negs, pool = _load_cifar_negatives(n_pos, n_extra=(n_test_faces + 1) * IMBALANCE_RATIO)
    X = np.concatenate([faces, negs.numpy()]).astype(np.float32)
    y = np.concatenate([np.ones(n_pos, int), np.zeros(n_pos, int)])
    pool = pool.numpy()

    finished = done_pairs()
    t_all = time.time()

    # fast arms for every seed first, the slow trained circuit last
    order = [(s, a) for a in ARMS[:-1] for s in STUDY_SEEDS] + [(s, ARMS[-1]) for s in STUDY_SEEDS]
    for seed, arm in order:
        if (seed, arm) in finished:
            continue

        idx = np.arange(len(X))
        tr, te = train_test_split(idx, test_size=TEST_SPLIT, random_state=seed, stratify=y)
        tr, va = train_test_split(tr, test_size=VAL_SPLIT, random_state=seed, stratify=y[tr])

        rng = np.random.default_rng(seed)
        te_faces = te[y[te] == 1]
        bg = pool[rng.permutation(len(pool))[: len(te_faces) * IMBALANCE_RATIO]]
        X_imb = np.concatenate([X[te_faces], bg])
        y_imb = np.concatenate([np.ones(len(te_faces), int), np.zeros(len(bg), int)])

        T = lambda a: torch.tensor(a)
        Xtr, Xva, Xte, Xim = T(X[tr]), T(X[va]), T(X[te]), T(X_imb)
        ytr, yva = T(y[tr]), T(y[va])

        torch.manual_seed(seed)
        t0 = time.time()
        if arm == "cnn_original":
            model = Classical12Net()
        elif arm == "cnn_matched":
            model = MatchedCNN()
        elif arm == "cnn_matched_frozen":
            model = MatchedCNN(frozen=True)
        elif arm == "qnn_trained":
            model = TorchQuanvNet(trainable=True)
        else:  # qnn_frozen: features never change, so compute them once
            qnet = TorchQuanvNet(trainable=False)
            with torch.no_grad():
                Xtr, Xva, Xte, Xim = (qnet.features(a) for a in (Xtr, Xva, Xte, Xim))
            model = FeatureHead(qnet)

        w_before = model.weights.detach().clone() if arm == "qnn_trained" else None
        best_epoch, best_vloss, hist = fit(model, Xtr, ytr, Xva, yva, seed)

        rec = {
            "seed": seed, "arm": arm,
            "trainable_params": n_trainable(model),
            "best_epoch": best_epoch, "val_loss": best_vloss, "val_history": hist,
            "balanced": metrics(y[te], predict(model, Xte)),
            "imbalanced": metrics(y_imb, predict(model, Xim)),
            "n_test": int(len(te)), "n_imb_faces": int(len(te_faces)), "n_imb_bg": int(len(bg)),
            "seconds": round(time.time() - t0, 1),
        }
        if w_before is not None:
            rec["circuit_weight_change"] = float((model.weights.detach() - w_before).abs().max())

        with open(RUNS, "a") as f:
            f.write(json.dumps(rec) + "\n")
        b, im = rec["balanced"], rec["imbalanced"]
        print(f"[{time.time() - t_all:6.0f}s] seed {seed} {arm:19s} epoch {best_epoch:2d} | "
              f"acc {b['accuracy']:.4f} auc {b['roc_auc']:.4f} | "
              f"1:10 prec {im['precision']:.4f} AP {im['avg_precision']:.4f} FP {im['false_positives']:4d} | "
              f"{rec['seconds']}s", flush=True)

    print("study complete")


if __name__ == "__main__":
    main()
