"""The PyTorch circuit used to train the quantum layer must match PennyLane.

Checks values on random images and gradients against PennyLane's
parameter-shift rule, and that a training step actually moves the circuit
weights (the original pipeline's weights never moved).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from src.models.torch_quanv import TorchQuanvNet, validate_against_pennylane, validate_gradients

ok = True

v = validate_against_pennylane(n_images=3, seed=1)
print(f"values vs PennyLane       max abs diff {v:.2e}  {'OK' if v < 1e-5 else 'FAIL'}")
ok &= v < 1e-5

g = validate_gradients(seed=1)
print(f"gradients vs param-shift  max abs diff {g:.2e}  {'OK' if g < 1e-5 else 'FAIL'}")
ok &= g < 1e-5

torch.manual_seed(0)
for trainable in (False, True):
    m = TorchQuanvNet(trainable=trainable)
    w0 = m.weights.detach().clone()
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    x, y = torch.rand(8, 1, 32, 32), torch.randint(0, 2, (8,))
    opt.zero_grad(); F.cross_entropy(m(x), y).backward(); opt.step()
    moved = float((m.weights.detach() - w0).abs().max())
    expect = moved > 0 if trainable else moved == 0
    print(f"trainable={trainable!s:5}  circuit weights moved by {moved:.4f}  {'OK' if expect else 'FAIL'}")
    ok &= expect

print("\nTORCH CIRCUIT TESTS:", "PASSED" if ok else "FAILED")
sys.exit(0 if ok else 1)
