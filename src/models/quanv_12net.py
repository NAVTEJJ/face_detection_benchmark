import hashlib
import os
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


_batch_dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(_batch_dev)
def _batched_circuit(inputs, weights):
    """Same circuit as quanv_circuit, evaluated over a batch of patches.

    PennyLane broadcasts a leading batch dimension on gate parameters, so a
    whole image's worth of patches (and then some) goes through one tape
    construction and one vectorised statevector simulation. The per-patch loop
    in quanvolve() rebuilds the tape 256 times per image, which dominates the
    cost: batching is ~175x faster here and produces bit-identical values.
    quanvolve() is kept as the readable reference that _validate_batched()
    checks against.
    """
    for i in range(N_QUBITS):
        qml.RY(inputs[..., i] * np.pi, wires=i)
    for layer in range(N_LAYERS):
        for q in range(N_QUBITS):
            qml.RX(weights[layer, q, 0], wires=q)
            qml.RY(weights[layer, q, 1], wires=q)
            qml.RZ(weights[layer, q, 2], wires=q)
        for q in range(N_QUBITS):
            qml.CNOT(wires=[q, (q + 1) % N_QUBITS])
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


def _extract_patches(images):
    """[N,1,H,W] -> [N, n_patches, KERNEL_SIZE**2], patch-row-major.

    Matches quanvolve()'s ordering exactly: patch (i, j) flattens as
    [img[2i,2j], img[2i,2j+1], img[2i+1,2j], img[2i+1,2j+1]], and patches run
    i-major so index i*out_size + j lands back at feature_map[:, i, j].
    """
    x = np.asarray(images, dtype=np.float32)
    n, _, h, w = x.shape
    out = h // STRIDE
    k = KERNEL_SIZE
    # (N, out, k, out, k) -> (N, out, out, k, k) -> (N, out*out, k*k)
    return (
        x[:, 0]
        .reshape(n, out, k, out, k)
        .transpose(0, 1, 3, 2, 4)
        .reshape(n, out * out, k * k)
    )


def _validate_batched(weights_np, n_images=2, seed=0):
    """Assert the fast path reproduces the per-patch reference."""
    rng = np.random.default_rng(seed)
    imgs = rng.random((n_images, 1, IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)

    layer = qml.qnn.TorchLayer(quanv_circuit, weight_shapes)
    with torch.no_grad():
        layer.weights.copy_(torch.tensor(weights_np, dtype=layer.weights.dtype))
        reference = torch.stack(
            [quanvolve(torch.tensor(im), layer) for im in imgs]
        ).detach().numpy()

    fast = _run_batched(imgs, weights_np)
    return float(np.abs(reference - fast).max())


def _run_batched(images, weights_np, chunk=65536):
    out = IMAGE_SIZE // STRIDE
    patches = _extract_patches(images)               # (N, P, k*k)
    n, p, _ = patches.shape
    flat = patches.reshape(n * p, -1)

    pieces = []
    for start in range(0, len(flat), chunk):
        res = _batched_circuit(flat[start:start + chunk], weights_np)
        pieces.append(np.stack([np.asarray(r) for r in res], axis=0))   # (4, chunk)
    feats = np.concatenate(pieces, axis=1)           # (4, N*P)

    return (
        feats.reshape(QUANTUM_OUT_CHANNELS, n, out, out)
        .transpose(1, 0, 2, 3)
        .astype(np.float32)
    )


def _cache_key(images, weights_np):
    """Fingerprint everything the cached features depend on.

    Keying the cache on filename alone is unsafe: change the negative class
    set, the split seed, the circuit shape or the weight init and the stale
    .pt file still loads, silently pairing old features with new labels. The
    hash covers the exact pixels, the circuit weights, and the circuit
    geometry, so any of those changing produces a different cache file.
    """
    h = hashlib.sha256()
    arr = np.ascontiguousarray(np.asarray(images, dtype=np.float32))
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())
    h.update(np.ascontiguousarray(weights_np, dtype=np.float64).tobytes())
    h.update(str((N_QUBITS, N_LAYERS, KERNEL_SIZE, STRIDE, QUANTUM_OUT_CHANNELS)).encode())
    return h.hexdigest()[:16]


def precompute_features(images, q_layer, cache_path):
    weights_np = q_layer.weights.detach().cpu().numpy()
    key = _cache_key(images, weights_np)
    root, ext = os.path.splitext(cache_path)
    cache_path = f"{root}_{key}{ext}"

    if os.path.exists(cache_path):
        print(f"  Loading cached quanv features from {cache_path}")
        return torch.load(cache_path, weights_only=True)

    n = len(images)
    print(f"  Computing quanv features for {n} images (batched circuit evaluation)...")

    drift = _validate_batched(weights_np)
    print(f"    fast path vs per-patch reference: max abs diff {drift:.2e}")
    if drift > 1e-6:
        raise RuntimeError(
            f"Batched quanvolution diverged from the reference implementation "
            f"(max abs diff {drift:.3e}). Refusing to cache suspect features."
        )

    t0 = time.time()
    features = torch.tensor(_run_batched(images, weights_np), dtype=torch.float32)
    print(f"    done in {time.time() - t0:.1f}s ({n / max(1e-9, time.time() - t0):.0f} img/s)")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    try:
        torch.save(features, cache_path)
        print(f"  Saved to {cache_path}")
    except OSError as e:
        print(f"  Warning: could not save cache ({e}).")

    return features


class QuanvNet(nn.Module):
    """Quanvolutional front end feeding a small classical classifier.

    The circuit weights are FROZEN by default, and that is not a flag chosen
    for speed: it is what the pipeline already did. Features are computed once
    in precompute_features() and cached to disk, then training runs through
    forward_from_features(), which takes that cached tensor as a leaf input.
    The loss never reaches q_layer.weights, so its .grad stays None and Adam
    skips it. Leaving requires_grad=True made count_parameters() report 24
    trained parameters that in fact never moved.

    So the quantum layer is a fixed random projection. That is a legitimate
    thing to benchmark (cf. random-feature baselines), but it has to be
    described as one. Set freeze_quantum=False and train through forward()
    to actually optimise the circuit, at which point the disk cache must be
    bypassed or it will serve stale features.
    """

    def __init__(self, freeze_quantum=True):
        super().__init__()
        self.q_layer = qml.qnn.TorchLayer(quanv_circuit, weight_shapes)
        self.quantum_frozen = freeze_quantum
        if freeze_quantum:
            for p in self.q_layer.parameters():
                p.requires_grad_(False)
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

    def count_frozen_parameters(self):
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)

    def count_quantum_params(self):
        return N_LAYERS * N_QUBITS * 3

    def circuit_depth(self):
        dummy_in = torch.zeros(N_QUBITS)
        dummy_w  = torch.zeros(N_LAYERS, N_QUBITS, 3)
        specs = qml.specs(quanv_circuit)(dummy_in, dummy_w)
        return specs["resources"].depth
