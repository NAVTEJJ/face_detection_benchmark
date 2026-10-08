"""The quanvolution circuit as plain PyTorch tensor ops.

Same circuit as quanv_circuit in quanv_12net.py (RY encoding, N_LAYERS of
RX/RY/RZ on every qubit followed by a CNOT ring, Pauli-Z on every qubit), but
written directly as a 4-qubit statevector so that:

  * every patch of a batch is simulated at once, and
  * autograd flows into the circuit weights, so the circuit can be trained.

The PennyLane version is the reference. validate_against_pennylane() checks
values agree before any result from this module is trusted.

State layout: complex tensor (N, 2, 2, 2, 2), axis q+1 is wire q. Wire 0 is
the most significant bit, as in PennyLane.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import N_QUBITS, N_LAYERS, KERNEL_SIZE, STRIDE, QUANTUM_OUT_CHANNELS

assert N_QUBITS == KERNEL_SIZE * KERNEL_SIZE == QUANTUM_OUT_CHANNELS


def _apply_1q(state, q, m00, m01, m10, m11):
    s = state.movedim(q + 1, -1)
    a, b = s[..., 0], s[..., 1]
    out = torch.stack([m00 * a + m01 * b, m10 * a + m11 * b], dim=-1)
    return out.movedim(-1, q + 1)


def _ry(state, q, theta):
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    return _apply_1q(state, q, c, -s, s, c)


def _rx(state, q, theta):
    c = torch.cos(theta / 2).to(state.dtype)
    s = (-1j * torch.sin(theta / 2)).to(state.dtype)
    return _apply_1q(state, q, c, s, s, c)


def _rz(state, q, theta):
    e = torch.exp(-0.5j * theta.to(state.dtype))
    return _apply_1q(state, q, e, 0, 0, e.conj())


def _cnot(state, control, target):
    s = state.movedim((control + 1, target + 1), (-2, -1))
    out = torch.stack([s[..., 0, :], s[..., 1, :].flip(-1)], dim=-2)
    return out.movedim((-2, -1), (control + 1, target + 1))


def circuit_expvals(patches, weights):
    """patches: (N, 4) pixel values in [0, 1]; weights: (N_LAYERS, 4, 3).

    Returns (N, 4) Pauli-Z expectations, one per wire.
    """
    n = patches.shape[0]
    state = torch.zeros((n,) + (2,) * N_QUBITS, dtype=torch.complex64, device=patches.device)
    state[(slice(None),) + (0,) * N_QUBITS] = 1.0

    enc = (patches * math.pi).to(torch.complex64)
    shape = (n,) + (1,) * (N_QUBITS - 1)          # per-sample angle, broadcast over the rest
    for i in range(N_QUBITS):
        state = _ry(state, i, enc[:, i].reshape(shape))

    w = weights.to(torch.float32)
    for layer in range(N_LAYERS):
        for q in range(N_QUBITS):
            state = _rx(state, q, w[layer, q, 0])
            state = _ry(state, q, w[layer, q, 1].to(torch.complex64))
            state = _rz(state, q, w[layer, q, 2])
        for q in range(N_QUBITS):
            state = _cnot(state, q, (q + 1) % N_QUBITS)

    probs = state.real ** 2 + state.imag ** 2
    out = []
    for i in range(N_QUBITS):
        p = probs.movedim(i + 1, -1).reshape(n, -1, 2).sum(dim=1)
        out.append(p[:, 0] - p[:, 1])
    return torch.stack(out, dim=1)


def quanvolve_batch(images, weights):
    """images: (B, 1, H, W) -> (B, 4, H/2, W/2), same patch order as quanvolve()."""
    b, _, h, w = images.shape
    patches = F.unfold(images, kernel_size=KERNEL_SIZE, stride=STRIDE)   # (B, 4, P)
    patches = patches.transpose(1, 2).reshape(-1, KERNEL_SIZE * KERNEL_SIZE)
    z = circuit_expvals(patches, weights)                               # (B*P, 4)
    out = h // STRIDE
    return z.reshape(b, out, out, QUANTUM_OUT_CHANNELS).permute(0, 3, 1, 2)


class TorchQuanvNet(nn.Module):
    """Quanv front end + the same classifier head as QuanvNet.

    trainable=False reproduces the frozen random-projection QNN; trainable=True
    lets gradients reach the 24 circuit parameters.
    """

    def __init__(self, trainable=False):
        super().__init__()
        # TorchLayer's default init: uniform on [0, 2*pi)
        w = torch.rand(N_LAYERS, N_QUBITS, 3) * 2 * math.pi
        self.weights = nn.Parameter(w, requires_grad=trainable)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(QUANTUM_OUT_CHANNELS * 8 * 8, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def features(self, x):
        return quanvolve_batch(x, self.weights)

    def forward_from_features(self, feats):
        return self.classifier(self.pool(feats))

    def forward(self, x):
        return self.forward_from_features(self.features(x))


def validate_against_pennylane(n_images=3, seed=0):
    """Max abs difference vs the PennyLane per-patch reference."""
    import numpy as np
    from src.models.quanv_12net import _run_batched, _validate_batched

    rng = np.random.default_rng(seed)
    w = rng.uniform(0, 2 * math.pi, size=(N_LAYERS, N_QUBITS, 3))
    imgs = rng.random((n_images, 1, 32, 32)).astype("float32")

    # PennyLane batched path is itself checked against the per-patch loop
    assert _validate_batched(w) < 1e-6
    ref = _run_batched(imgs, w)
    ours = quanvolve_batch(torch.tensor(imgs), torch.tensor(w, dtype=torch.float32)).numpy()
    return float(np.abs(ref - ours).max())


def validate_gradients(seed=0):
    """Autograd vs parameter-shift on PennyLane, for the circuit weights."""
    import numpy as np
    import pennylane as qml

    rng = np.random.default_rng(seed)
    w = rng.uniform(0, 2 * math.pi, size=(N_LAYERS, N_QUBITS, 3))
    patch = rng.random(N_QUBITS)

    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev, diff_method="parameter-shift")
    def ref(weights):
        for i in range(N_QUBITS):
            qml.RY(patch[i] * np.pi, wires=i)
        for layer in range(N_LAYERS):
            for q in range(N_QUBITS):
                qml.RX(weights[layer, q, 0], wires=q)
                qml.RY(weights[layer, q, 1], wires=q)
                qml.RZ(weights[layer, q, 2], wires=q)
            for q in range(N_QUBITS):
                qml.CNOT(wires=[q, (q + 1) % N_QUBITS])
        return qml.expval(qml.PauliZ(0) + qml.PauliZ(1) + qml.PauliZ(2) + qml.PauliZ(3))

    g_ref = qml.grad(ref)(qml.numpy.array(w, requires_grad=True))

    wt = torch.tensor(w, dtype=torch.float32, requires_grad=True)
    circuit_expvals(torch.tensor(patch, dtype=torch.float32)[None], wt).sum().backward()
    return float(np.abs(np.asarray(g_ref) - wt.grad.numpy()).max())


if __name__ == "__main__":
    print("values   max abs diff vs PennyLane:", validate_against_pennylane())
    print("gradient max abs diff vs param-shift:", validate_gradients())
