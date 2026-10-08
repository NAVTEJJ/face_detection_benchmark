import os
import sys
import json
import random
import numpy as np
import torch

# reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

os.makedirs("outputs", exist_ok=True)
os.makedirs(os.path.join("data", "cache"), exist_ok=True)


from src.dataset import load_data, build_imbalanced_set
from src.models.classical_12net import Classical12Net
from src.models.quanv_12net import QuanvNet, precompute_features
from src.benchmark.trainer import train_cnn, train_qnn
from src.benchmark.evaluator import (
    evaluate_model, evaluate_qnn_from_features,
    build_metrics_dict, print_metrics,
    measure_cnn_latency, measure_qnn_patch_latency,
    evaluate_imbalanced, print_imbalanced,
)
from src.baseline_g0 import run_gate, print_report
from src.plots import (
    generate_arch_diagram,
    plot_training_curves, plot_roc_curves,
    plot_confusion_matrices, plot_misclassified,
    plot_comparison_table,
)
from src.config import IMAGE_SIZE, CIFAR_NEGATIVE_CLASSES


def main():
    print("\n--- Face Detection Benchmark (Stage 1) ---")
    print("Classical 12-Net CNN vs Quanvolutional QNN\n")

    print("Loading dataset...")
    train_loader, test_loader, meta = load_data()
    meta["class_balance_fig"].savefig("outputs/class_balance.png", dpi=120, bbox_inches="tight")

    X_test_np = meta["X_test"]
    y_test_np = meta["y_test"]

    # --- Gate G0: is the split separable without modelling faces? ---
    # Runs before any training. If trivial statistics already carry the split,
    # whatever the two models score afterwards is not a face-detection result.
    print("\n=== Gate G0: trivial-baseline separability ===")
    g0_train = run_gate(meta["X_train"], meta["y_train"], label="train split")
    g0_verdict = print_report(g0_train)
    if g0_verdict == "FAIL":
        print("\n  WARNING: G0 failed. Downstream accuracy measures dataset provenance,")
        print("           not face detection. Treat the comparison as uninterpretable.")

    # --- Classical CNN ---
    print("\nTraining Classical CNN...")
    cnn_model = Classical12Net()
    cnn_history, cnn_train_time = train_cnn(cnn_model, train_loader)

    cnn_y_true, cnn_y_pred, cnn_y_prob = evaluate_model(cnn_model, test_loader)
    cnn_lat_mean, cnn_lat_std = measure_cnn_latency(cnn_model)
    cnn_lat_full = cnn_lat_mean

    cnn_metrics = build_metrics_dict(
        cnn_y_true, cnn_y_pred, cnn_y_prob,
        cnn_model, cnn_train_time,
        cnn_lat_mean, cnn_lat_std, cnn_lat_full,
        n_params=cnn_model.count_parameters(),
    )
    print_metrics("Classical 12-Net CNN", cnn_metrics)

    # --- Quanvolutional QNN ---
    print("\nSetting up Quanvolutional QNN...")
    qnn_model = QuanvNet()
    depth = qnn_model.circuit_depth()
    print(f"  Circuit depth: {depth}  |  Variational params: {qnn_model.count_quantum_params()}")

    train_imgs, train_lbls = [], []
    for X_b, y_b in train_loader:
        train_imgs.append(X_b.numpy())
        train_lbls.append(y_b.numpy())
    train_imgs = np.concatenate(train_imgs)
    train_lbls = np.concatenate(train_lbls)

    cache_dir = os.path.join("data", "cache")
    feat_train = precompute_features(
        train_imgs, qnn_model.q_layer,
        os.path.join(cache_dir, "quanv_features_train.pt")
    )
    feat_test = precompute_features(
        X_test_np, qnn_model.q_layer,
        os.path.join(cache_dir, "quanv_features_test.pt")
    )

    print("Training QNN...")
    qnn_history, qnn_train_time = train_qnn(
        qnn_model,
        feat_train,
        torch.tensor(train_lbls, dtype=torch.long)
    )

    qnn_y_true, qnn_y_pred, qnn_y_prob = evaluate_qnn_from_features(
        qnn_model, feat_test, torch.tensor(y_test_np, dtype=torch.long)
    )
    qnn_lat_mean, qnn_lat_std = measure_qnn_patch_latency(qnn_model.q_layer)
    qnn_lat_full = qnn_lat_mean * (IMAGE_SIZE // 2) ** 2

    qnn_metrics = build_metrics_dict(
        qnn_y_true, qnn_y_pred, qnn_y_prob,
        qnn_model, qnn_train_time,
        qnn_lat_mean, qnn_lat_std, qnn_lat_full,
        n_params=qnn_model.count_parameters(),
        q_params=qnn_model.count_quantum_params(),
        circuit_depth=depth,
    )
    print_metrics("Quanvolutional QNN", qnn_metrics)

    # --- Imbalanced (deployment ratio) evaluation ---
    print(f"\n{'='*50}")
    print(" Imbalanced evaluation")
    print(f"{'='*50}")
    X_imb, y_imb = build_imbalanced_set(meta)
    print(f"  Set: {int((y_imb==1).sum())} faces / {int((y_imb==0).sum())} background")
    print("  Background drawn from a CIFAR pool held out of both train and test.")

    imb_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(X_imb), torch.tensor(y_imb)),
        batch_size=64, shuffle=False,
    )
    _, _, cnn_imb_prob = evaluate_model(cnn_model, imb_loader)
    cnn_imb = evaluate_imbalanced(y_imb, cnn_imb_prob[:, 1])
    print_imbalanced("Classical CNN", cnn_imb, balanced=cnn_metrics)

    feat_imb = precompute_features(
        X_imb, qnn_model.q_layer, os.path.join(cache_dir, "quanv_features_imbalanced.pt")
    )
    _, _, qnn_imb_prob = evaluate_qnn_from_features(
        qnn_model, feat_imb, torch.tensor(y_imb, dtype=torch.long)
    )
    qnn_imb = evaluate_imbalanced(y_imb, qnn_imb_prob[:, 1])
    print_imbalanced("Quanvolutional QNN", qnn_imb, balanced=qnn_metrics)

    # --- Visualizations ---
    print("\nGenerating plots...")
    plot_training_curves(cnn_history, qnn_history)
    plot_roc_curves(cnn_metrics, qnn_metrics)
    plot_confusion_matrices(cnn_metrics, qnn_metrics)

    better_metrics = cnn_metrics if cnn_metrics["f1"] >= qnn_metrics["f1"] else qnn_metrics
    plot_misclassified(better_metrics, X_test_np, y_test_np)

    plot_comparison_table(cnn_metrics, qnn_metrics)

    print("\nGenerating architecture diagram...")
    generate_arch_diagram(
        cnn_params=cnn_metrics["n_params"],
        qnn_params=qnn_metrics["n_params"],
        circuit_depth=depth
    )

    def serialise(d):
        out = {}
        for k, v in d.items():
            if isinstance(v, (np.ndarray, )):
                continue
            elif isinstance(v, (np.integer,)):
                out[k] = int(v)
            elif isinstance(v, (np.floating,)):
                out[k] = float(v)
            else:
                out[k] = v
        return out

    metrics_json = {
        "gate_g0": serialise(g0_train),
        "cnn": serialise(cnn_metrics),
        "qnn": serialise(qnn_metrics),
        "cnn_imbalanced": serialise(cnn_imb),
        "qnn_imbalanced": serialise(qnn_imb),
        "negative_classes": CIFAR_NEGATIVE_CLASSES,
        "quantum_frozen": qnn_model.quantum_frozen,
    }
    with open("outputs/metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=2)
    print("Metrics saved to outputs/metrics.json")

    # --- Summary ---
    print("\n--- Summary ---")
    print(f"Gate G0: sharpness AUC {g0_train['sharpness_auc']:.3f}, "
          f"raw-pixel AUC {g0_train['pixel_auc']:.3f}  ->  {g0_verdict}")
    print("Quantum layer: " + ("FROZEN (random projection; 24 params never trained)"
                               if qnn_model.quantum_frozen else "trained"))
    print(f"Imbalanced 1:10 precision  CNN {cnn_imb['precision']:.4f}  |  QNN {qnn_imb['precision']:.4f}")
    winner = "CNN" if cnn_metrics["f1"] >= qnn_metrics["f1"] else "QNN"
    print(f"Better model (F1): {winner}")
    print(f"CNN  F1={cnn_metrics['f1']:.4f}  AUC={cnn_metrics['roc_auc']:.4f}  Latency={cnn_metrics['latency_mean']:.2f}ms")
    print(f"QNN  F1={qnn_metrics['f1']:.4f}  AUC={qnn_metrics['roc_auc']:.4f}  Latency={qnn_metrics['latency_mean']:.2f}ms/patch  Full={qnn_metrics['latency_full']:.1f}ms")
    print(f"\nAll outputs in outputs/")
    print("Done.\n")


if __name__ == "__main__":
    main()
