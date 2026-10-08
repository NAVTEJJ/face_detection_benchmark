import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

from src.config import BATCH_SIZE, LATENCY_REPEATS, IMAGE_SIZE, N_QUBITS


def evaluate_model(model, test_loader, device="cpu"):
    model.eval()
    all_true, all_pred, all_prob = [], [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs  = F.softmax(logits, dim=1).cpu().numpy()
            preds  = probs.argmax(axis=1)
            all_true.append(y_batch.numpy())
            all_pred.append(preds)
            all_prob.append(probs)

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_prob = np.concatenate(all_prob)
    return y_true, y_pred, y_prob


def evaluate_qnn_from_features(model, feat_test, y_test_tensor, device="cpu"):
    ds = TensorDataset(feat_test, y_test_tensor)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()
    all_true, all_pred, all_prob = [], [], []

    with torch.no_grad():
        for feat_batch, y_batch in loader:
            feat_batch = feat_batch.to(device)
            logits = model.forward_from_features(feat_batch)
            probs  = F.softmax(logits, dim=1).cpu().numpy()
            preds  = probs.argmax(axis=1)
            all_true.append(y_batch.numpy())
            all_pred.append(preds)
            all_prob.append(probs)

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_prob = np.concatenate(all_prob)
    return y_true, y_pred, y_prob


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_true, y_prob[:, 1]),
        "conf_matrix": confusion_matrix(y_true, y_pred),
        "y_true":    y_true,
        "y_pred":    y_pred,
        "y_prob":    y_prob[:, 1],
    }


def measure_cnn_latency(model, device="cpu"):
    model.eval()
    dummy = torch.zeros(1, 1, IMAGE_SIZE, IMAGE_SIZE, device=device)
    times = []
    with torch.no_grad():
        for _ in range(LATENCY_REPEATS):
            t0 = time.perf_counter()
            model(dummy)
            times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.std(times))


def measure_qnn_patch_latency(q_layer):
    dummy_patch = torch.zeros(N_QUBITS)
    times = []
    for _ in range(LATENCY_REPEATS):
        t0 = time.perf_counter()
        _ = q_layer(dummy_patch)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.std(times))


def build_metrics_dict(y_true, y_pred, y_prob,
                       model, train_time,
                       latency_mean, latency_std, latency_full,
                       n_params,
                       q_params=None, circuit_depth=None):
    d = compute_metrics(y_true, y_pred, y_prob)
    d.update({
        "n_params":      n_params,
        "q_params":      q_params if q_params is not None else "--",
        "circuit_depth": circuit_depth,
        "train_time":    train_time,
        "latency_mean":  latency_mean,
        "latency_std":   latency_std,
        "latency_full":  latency_full,
    })
    return d


def print_metrics(name, metrics):
    print(f"\n{'='*50}")
    print(f" {name}")
    print(f"{'='*50}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1-Score  : {metrics['f1']:.4f}")
    print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(f"  Params    : {metrics['n_params']:,}")
    if metrics.get("q_params") and metrics["q_params"] != "--":
        print(f"  Q-Params  : {metrics['q_params']}  |  Depth: {metrics.get('circuit_depth','?')}")
    print(f"  Train time: {metrics['train_time']:.1f}s")
    print(f"  Latency   : {metrics['latency_mean']:.2f} +/- {metrics['latency_std']:.2f} ms/patch")
    print(f"  Full img  : {metrics['latency_full']:.2f} ms")
