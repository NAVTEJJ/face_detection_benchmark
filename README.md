# Face Detection Benchmark: Classical CNN vs. Quanvolutional QNN
### Stage 1 of a One-Shot Face Recognition System

---

**Results at a glance:** open `PRESENTATION.html` (single file, works offline) or `PRESENTATION.pdf`. Rebuild with `python make_presentation.py`.

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
│   ├── cifar.py               ← CIFAR-10 reader (no torchvision dependency)
│   ├── baseline_g0.py         ← trivial-baseline separability gate
│   ├── config.py              ← every hyperparameter lives here
│   ├── dataset.py             ← LFW + CIFAR-10 loading, balancing, splitting
│   ├── plots.py               ← ROC, confusion, error grid, and block diagram
│   ├── plots_findings.py      ← G0 and 1:10 figures, read from outputs/
│   ├── models/                ← Model architectures
│   │   ├── classical_12net.py ← Classical 12-Net CNN
│   │   └── quanv_12net.py     ← Quanvolutional QNN
│   └── benchmark/             ← Training and evaluation engines
│       ├── trainer.py         ← training loops
│       └── evaluator.py       ← metrics and latency tests
├── tests/
│   ├── test_pipeline.py       ← split disjointness + imbalance ratio
│   └── test_gate_g0.py        ← G0 known-answer self-test
├── notebooks/
│   └── face_detection_benchmark.ipynb ← interactive notebook
├── main.py                    ← end-to-end CLI runner
├── make_presentation.py       ← builds PRESENTATION.html from outputs/
├── PRESENTATION.html          ← one-file results page
├── requirements.txt           ← pinned project dependencies
├── data/cache/                ← where LFW and CIFAR datasets are cached
└── outputs/                   ← tables, curves, run_log.txt, gate_g0.json
```

---

## Dataset Details

| Source | Role | Classes Used | Preprocessing |
|--------|------|-------------|---------------|
| LFW (`fetch_lfw_people`) | Positive (Face) | All identities with ≥20 photos — 62 people, 3,023 images | Resize to 32×32 grayscale, normalize to [0,1] |
| CIFAR-10 | Negative (Non-Face) | All 10 classes | BT.601 grayscale, already 32×32 |

Class balance enforced 1:1 (3,023 / 3,023). 80/20 stratified split, `random_state=42`. A further 6,040 CIFAR images are held out of both splits as the background pool for the imbalanced evaluation.

CIFAR-10 is read directly from its pickled batches by `src/cifar.py`. `torchvision.datasets` would do the same job, but torchvision's compiled extension is pinned to a specific torch build and raises `operator torchvision::nms does not exist` at import time on a mismatch, which has nothing to do with this project.

**Why all ten negative classes, not four.** The original set was airplane/automobile/ship/truck: four rigid man-made categories with no animals, no fur or skin texture, and no centred subjects. Adding the six animal classes puts eyes, texture and centred blob structure on the negative side, so the model has to do face modelling rather than category discrimination. Measured effect on the trivial-baseline gate below: sharpness AUC 0.658 → 0.543, Cohen's d 0.63 → 0.29.

LFW was chosen over CBCL specifically because its multi-photo-per-identity structure supports Phase 2's triplet loss training. Same dataset, zero rework between phases.

---

## Gate G0: is the split separable without modelling faces?

When positives and negatives come from different sources they differ in focus, compression and framing, so a classifier can score well by learning *which dataset an image came from*. `src/baseline_g0.py` runs three deliberately stupid baselines before any training: variance of the Laplacian (one scalar), mean intensity (one scalar), and logistic regression on the raw 1,024 pixels.

| Baseline | Old negatives (4 classes) | All 10 classes |
|---|---|---|
| Sharpness (Laplacian variance) | AUC 0.658, d = 0.63 | **AUC 0.543, d = 0.29** |
| Mean intensity | AUC 0.502 | AUC 0.566 |
| Raw-pixel logistic | AUC 0.991 | AUC 0.993 |
| **Verdict** | PASS | **PASS** |

Both configurations pass. The four-class split was *not* badly confounded, contrary to what a blur-confound in a related project suggested it might be — widening the negatives improved the sharpness margin but did not rescue a broken benchmark, because it was not broken.

Only the scalar cues decide the verdict. A high raw-pixel score is expected and is not evidence of a confound: `tests/test_gate_g0.py` constructs two classes with identical sharpness and identical mean intensity that differ only in spatial arrangement, where the pixel model reaches AUC 1.000 while both scalars sit at chance. Gating on the pixel number would reject a perfectly good task.

---

## Architecture

The intent is that only the first feature-extraction layer differs. In the code as written it does not — see the capacity note below the diagram.

```
Classical 12-Net:
  Conv2d(1→16, 3×3) → ReLU → MaxPool(3×3, s=2) → Flatten → Linear(4096→16) → ReLU → Linear(16→2)

Quanv 12-Net:
  Quanv(4q, 2×2) → (MaxPool) → Flatten → Linear(256→16) → ReLU → Linear(16→2)
  ↑ this one layer is the entire experiment
```

**Quantum circuit:** RY angle encoding → 2-layer ring ansatz (RX+RY+RZ per qubit + CNOT ring) → Pauli-Z measurement per qubit. 24 circuit parameters total.

**Those 24 parameters are frozen, not trained.** Features are computed once in `precompute_features()` and cached to disk; training then runs through `forward_from_features()`, which takes the cached tensor as a leaf input. The loss never reaches `q_layer.weights`, so its `.grad` stays `None` and Adam skips it. This was already true before the parameters were marked `requires_grad=False` — the only thing that changed is that `count_parameters()` stopped reporting 24 trained parameters that never moved (4,170 → 4,146).

So the quantum layer is a **fixed random projection**, and the QNN arm is a random-feature baseline. That is a legitimate thing to measure, but it has to be described as one: no claim about *learned* or *variational* quantum features is supported by this code as it stands. Pass `QuanvNet(freeze_quantum=False)` and train through `forward()` to actually optimise the circuit, bypassing the cache.

**The two arms are not capacity-matched.** Only the first layer is described as differing, but the CNN emits 16 channels at 16×16 (flatten 4,096) while the QNN emits 4 channels that are pooled to 8×8 (flatten 256), and the CNN trains for 15 epochs against the QNN's 10. The classical classifier head therefore has 16× the input width and 50% more training. The accuracy gap cannot be attributed to the quantum layer until those are equalised.

---

## Results

### Balanced test set (1:1, n = 1,210)

| Metric                         | Classical 12-Net CNN | Quanvolutional QNN |
|--------------------------------|----------------------|--------------------|
| Test Accuracy                  | 0.9917               | 0.9545             |
| Precision                      | 0.9983               | 0.9630             |
| Recall                         | 0.9851               | 0.9455             |
| F1-Score                       | 0.9917               | 0.9541             |
| ROC-AUC                        | 0.9999               | 0.9891             |
| Total Trainable Params         | 65,746               | 4,146              |
| Quantum Params (frozen)        | --                   | 24                 |
| Qubit Count                    | --                   | 4                  |
| Circuit Depth                  | --                   | 15                 |
| Inference Latency / patch (ms) | 0.76                 | 15.39 +/- 3.23     |
| Inference / full 32x32 (ms)    | 0.76                 | 3939.84            |

### Imbalanced test set (1:10, 604 faces / 6,040 background)

Background drawn from a CIFAR pool held out of both train and test, so none of it was seen during training.

| Metric | Classical CNN | Quanvolutional QNN |
|---|---|---|
| Precision | 0.9566 (−0.042) | **0.5856 (−0.377)** |
| Recall | 0.9851 | 0.9454 |
| F1 | 0.9706 (−0.021) | 0.7232 (−0.231) |
| ROC-AUC | 0.9996 | 0.9840 |
| Average Precision | 0.9962 | 0.9202 |
| False positives | 27 (FPR 0.0045) | 404 (FPR 0.0669) |

**This is the result that matters, and the balanced table hides it.** Moving to the deployment ratio costs the CNN 4 points of precision and the QNN 38. ROC-AUC barely moves for either model (0.9891 → 0.9840 for the QNN) because ROC-AUC is invariant to class balance — which is exactly why it should not be a headline number for a detector. Average precision, which is not invariant, drops 0.989 → 0.920.

The QNN's 6.7% false-positive rate is tolerable at 1:1 and produces 404 false alarms at 1:10. In a real sliding window with thousands of background patches per image, it would be unusable at this threshold.

All evaluation plots (ROC curves, confusion matrices, training histories, misclassification examples, and comparison tables) are generated and saved under the `outputs/` directory. Full console output is in `outputs/run_log.txt`; gate numbers in `outputs/gate_g0.json`.

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

- **Single seed, single split.** Every number here comes from one run with `random_state=42`. There are no error bars, so the CNN-QNN gap has no variance estimate attached and small differences should not be read as real.
- **No validation set.** The 80/20 split is train/test only; epoch counts are fixed rather than selected, and the test set is the only holdout.
- **The quantum layer is frozen** (see Architecture). The QNN arm measures a random quantum projection, not a trained variational circuit.
- **The arms are not capacity-matched** (see Architecture). Head width and epoch budget both differ.
- `default.qubit` simulation is 100–1000× slower than actual NISQ hardware. Latency numbers reflect simulation overhead, not physical circuit execution time.
- LFW at `min_faces=20` yields 3,023 face images from 62 identities — small by production standards, appropriate for a simulator-constrained benchmark.
- 4-qubit, 2-layer ansatz may hit barren plateaus with more qubits/depth; gradient norms should be monitored in Phase 3's scaling analysis.

---

## Performance note

`quanvolve()` evaluates the circuit once per 2×2 patch, rebuilding the PennyLane tape 256 times per image, which dominates runtime. `_run_batched()` instead uses PennyLane's parameter broadcasting to push every patch through one vectorised statevector simulation: **~175× faster**, and validated against the per-patch reference on every cache miss (max abs diff 2.4e-7, below the 1e-6 threshold at which `precompute_features()` refuses to write the cache). The full 6,644-image imbalanced set takes 44s instead of a projected ~2 hours.

The feature cache is keyed by a SHA-256 of the exact pixels, the circuit weights and the circuit geometry. Keying on filename alone meant that changing the negative class set, the split seed or the ansatz silently reloaded stale features against new labels.

---

## Tests

```bash
python tests/test_pipeline.py   # split disjointness, imbalance ratio, held-out background
python tests/test_gate_g0.py    # G0 fires on known confounds, stays quiet on a clean task
```
