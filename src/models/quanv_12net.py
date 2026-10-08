import os
import multiprocessing
import time
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml

from src.config import N_QUBITS, N_LAYERS, KERNEL_SIZE, STRIDE, QUANTUM_OUT_CHANNELS, IMAGE_SIZE

dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quanv_circuit(inputs, weights):
    for i in range(N_QUBITS):
        qml.RY(inputs[i] * torch.pi, wires=i)
    for layer in range(N_LAYERS):
        for q in range(N_QUBITS):
            qml.RX(weights[layer, q, 0], wires=q)
            qml.RY(weights[layer, q, 1], wires=q)
            qml.RZ(weights[layer, q, 2], wires=q)
        for q in range(N_QUBITS):
            qml.CNOT(wires=[q, (q + 1) % N_QUBITS])
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

weight_shapes = {"weights": (N_LAYERS, N_QUBITS, 3)}


def quanvolve(image_tensor, q_layer):
    out_size = IMAGE_SIZE // STRIDE
    feature_map = torch.zeros(QUANTUM_OUT_CHANNELS, out_size, out_size)
    for i in range(out_size):
        for j in range(out_size):
            patch = image_tensor[0,
                                  i * STRIDE : i * STRIDE + KERNEL_SIZE,
                                  j * STRIDE : j * STRIDE + KERNEL_SIZE].flatten()
            result = q_layer(patch)
            feature_map[:, i, j] = result if result.dim() == 1 else torch.stack(list(result))
    return feature_map


def _quanvolve_worker(image_np, weights_np):
    import pennylane as qml
    import numpy as np
    dev = qml.device("default.qubit", wires=4)

    @qml.qnode(dev)
    def local_quanv_circuit(inputs, weights):
        for i in range(4):
            qml.RY(inputs[i] * np.pi, wires=i)
        for layer in range(2):
            for q in range(4):
                qml.RX(weights[layer, q, 0], wires=q)
                qml.RY(weights[layer, q, 1], wires=q)
                qml.RZ(weights[layer, q, 2], wires=q)
            for q in range(4):
                qml.CNOT(wires=[q, (q + 1) % 4])
        return [qml.expval(qml.PauliZ(i)) for i in range(4)]

    out_size = image_np.shape[1] // 2
    feature_map = np.zeros((4, out_size, out_size), dtype=np.float32)

    for i in range(out_size):
        for j in range(out_size):
            patch = image_np[0, i*2:i*2+2, j*2:j*2+2].flatten()
            res = local_quanv_circuit(patch, weights_np)
            feature_map[:, i, j] = res
    return feature_map


def _worker_helper(args):
    return _quanvolve_worker(args[0], args[1])


def precompute_features(images, q_layer, cache_path):
    if os.path.exists(cache_path):
        print(f"  Loading cached quanv features from {cache_path}")
        return torch.load(cache_path, weights_only=True)

    import multiprocessing
    n = len(images)
    weights_np = q_layer.weights.detach().cpu().numpy()
    num_workers = min(12, max(1, os.cpu_count() - 2))
    print(f"  Computing quanv features for {n} images using {num_workers} parallel workers...")

    features_list = [None] * n
    t0 = time.time()

    with multiprocessing.Pool(processes=num_workers) as pool:
        args = [(images[idx], weights_np) for idx in range(n)]
        for idx, feat_map in enumerate(pool.imap(_worker_helper, args)):
            features_list[idx] = feat_map
            if (idx + 1) % 100 == 0 or idx + 1 == n:
                elapsed = time.time() - t0
                rate = (idx + 1) / elapsed
                rem_time = (n - (idx + 1)) / rate if rate > 0 else 0
                print(f"    [{idx+1}/{n}] processed | Rate: {rate:.2f} img/s | Est. remaining: {rem_time/60:.1f} min")

    features = torch.tensor(np.array(features_list), dtype=torch.float32)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    try:
        torch.save(features, cache_path)
        print(f"  Saved to {cache_path}")
    except OSError as e:
        print(f"  Warning: could not save cache ({e}).")

    return features


class QuanvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_layer = qml.qnn.TorchLayer(quanv_circuit, weight_shapes)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(QUANTUM_OUT_CHANNELS * 8 * 8, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward_from_features(self, quanv_features):
        return self.classifier(self.pool(quanv_features))

    def forward(self, x):
        batch_features = torch.stack([quanvolve(x[i], self.q_layer) for i in range(len(x))])
        return self.forward_from_features(batch_features)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_quantum_params(self):
        return N_LAYERS * N_QUBITS * 3

    def circuit_depth(self):
        dummy_in = torch.zeros(N_QUBITS)
        dummy_w  = torch.zeros(N_LAYERS, N_QUBITS, 3)
        specs = qml.specs(quanv_circuit)(dummy_in, dummy_w)
        return specs["resources"].depth
