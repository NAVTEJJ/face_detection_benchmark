# Face Detection Benchmark: Classical CNN vs. Quanvolutional QNN
### Stage 1 of a One-Shot Face Recognition System

---

## Quick Start

```bash
cd face_detection_benchmark
pip install -r requirements.txt
python main.py
```

First run downloads LFW (~200 MB, cached to `~/scikit_learn_data`) and CIFAR-10 (~170 MB, cached to `data/cache/`). The quanvolutional feature maps are computed once and cached to `data/cache/quanv_features_*.pt` — subsequent runs skip this and load from disk.

All outputs land in `outputs/`.

---

## Project Structure

```
face_detection_benchmark/
├── src/
│   ├── config.py              ← every hyperparameter lives here
│   ├── dataset.py             ← LFW + CIFAR-10 loading, balancing, splitting
│   ├── plots.py               ← ROC, confusion, error grid, and block diagram
│   ├── models/                ← Model architectures
│   │   ├── classical_12net.py ← Classical 12-Net CNN
│   │   └── quanv_12net.py     ← Quanvolutional QNN
│   └── benchmark/             ← Training and evaluation engines
│       ├── trainer.py         ← training loops
│       └── evaluator.py       ← metrics and latency tests
├── notebooks/
│   └── face_detection_benchmark.ipynb ← interactive notebook
├── main.py                    ← end-to-end CLI runner
├── requirements.txt           ← pinned project dependencies
├── data/cache/                ← where LFW and CIFAR datasets are cached
└── outputs/                   ← comparison tables and evaluation curves
```

---

## Dataset Details

| Source | Role | Classes Used | Preprocessing |
|--------|------|-------------|---------------|
| LFW (`fetch_lfw_people`) | Positive (Face) | All identities with ≥20 photos | Resize to 32×32 grayscale, normalize to [0,1] |
| CIFAR-10 | Negative (Non-Face) | airplane, automobile, ship, truck | BT.601 grayscale, already 32×32 |

Class balance enforced 1:1. 80/20 stratified split, `random_state=42`.

LFW was chosen over CBCL specifically because its multi-photo-per-identity structure supports Phase 2's triplet loss training. Same dataset, zero rework between phases.

---

## Architecture

Both models share the same structure — only the first feature-extraction layer differs.

```
Classical 12-Net:
  Conv2d(1→16, 3×3) → ReLU → MaxPool(3×3, s=2) → Flatten → Linear(4096→16) → ReLU → Linear(16→2)

Quanv 12-Net:
  Quanv(4q, 2×2) → (MaxPool) → Flatten → Linear(256→16) → ReLU → Linear(16→2)
  ↑ this one layer is the entire experiment
```

**Quantum circuit:** RY angle encoding → 2-layer ring ansatz (RX+RY+RZ per qubit + CNOT ring) → Pauli-Z measurement per qubit. 24 variational parameters total.

---

## Results

| Metric                         | Classical 12-Net CNN | Quanvolutional QNN |
|--------------------------------|----------------------|--------------------|
| Test Accuracy                  | 0.9893               | 0.9512             |
| Precision                      | 0.9933               | 0.9565             |
| Recall                         | 0.9851               | 0.9455             |
| F1-Score                       | 0.9892               | 0.9510             |
| ROC-AUC                        | 0.9989               | 0.9896             |
| Total Trainable Params         | 65,746               | 4,170              |
| Quantum Variational Params     | --                   | 24                 |
| Qubit Count                    | --                   | 4                  |
| Circuit Depth                  | --                   | 15                 |
| Training Time (s)              | 8.0                  | 1.0                |
| Inference Latency / patch (ms) | 0.15 +/- 0.03        | 9.12 +/- 1.97      |
| Inference / full 32x32 (ms)    | 0.15                 | 2334.69            |

All evaluation plots (ROC curves, confusion matrices, training histories, misclassification examples, and comparison tables) are generated and saved under the `outputs/` directory.

---

## Engineering Rationale

**Why 12-net, not ResNet/MobileNet:**
The point of Stage 1 is measuring what one conv→quanv swap does, not maximising accuracy. A deep network's many layers would drown out the quantum layer's signal. The 12-net is small enough that the first layer's contribution is measurable — and it's a real published architecture, not a toy.

**Why quanvolution, not a generic VQC:**
A bolt-on quantum layer at the end of a classical feature extractor is addition, not conversion. Quanvolution replaces the conv layer structurally — same architecture, one layer swapped — so the comparison table is measuring one variable.

**Why LFW + CIFAR-10, not CBCL:**
CBCL has no per-identity structure. It works for binary detection but dead-ends at Phase 2. LFW's multi-photo-per-identity hierarchy is the specific property that makes triplet-loss embedding training possible without collecting new data.

---

## Phase 1–4 Roadmap

| Phase | Goal | Key Change from Previous |
|-------|------|--------------------------|
| **1 (this)** | Conv↔quanv detection benchmark | — |
| **2** | Triplet-loss embedding, one-shot verification | Swap softmax head for 128-d embedding; triplet loss; cosine threshold |
| **3** | Qubit scaling analysis | Formal qubit allocation model as image/dataset size grows |
| **4** | Hybrid deployment at scale | QNN moves to end-stage on compressed embeddings, not raw patches |

Phases 1→2 is a head swap, not a rebuild. The same backbone carries forward.

---

## Known Limitations

- `default.qubit` simulation is 100–1000× slower than actual NISQ hardware. Latency numbers in the comparison table reflect simulation overhead, not physical circuit execution time.
- LFW at `min_faces=20` yields ~3000 face images — small by production standards, appropriate for a simulator-constrained benchmark.
- 4-qubit, 2-layer ansatz may hit barren plateaus with more qubits/depth; gradient norms should be monitored in Phase 3's scaling analysis.
