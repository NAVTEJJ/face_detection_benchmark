# Face Detection Benchmark: Classical CNN vs Quanvolutional QNN

Stage 1 of a one-shot face recognition system. The question is narrow: **if the first layer of a small face/non-face classifier is replaced by a 4-qubit quantum circuit (a "quanvolutional" layer), what changes?**

Faces come from LFW, non-faces from CIFAR-10, and every image is a 32×32 grayscale patch. The quantum circuit runs on a classical simulator (PennyLane, plus an equivalent PyTorch implementation used for training).

**Results page:** https://navtejj.github.io/face_detection_benchmark/PRESENTATION.html (also `PRESENTATION.html` and `PRESENTATION.pdf` in this repo, both work offline)

---

## Contents

1. [Key findings](#1-key-findings)
2. [Setup](#2-setup)
3. [How to run](#3-how-to-run)
4. [What each output file is](#4-what-each-output-file-is)
5. [Repository layout](#5-repository-layout)
6. [Method](#6-method)
7. [Results in detail](#7-results-in-detail)
8. [Implementation notes](#8-implementation-notes)
9. [Limitations](#9-limitations)
10. [Next steps](#10-next-steps)
11. [Troubleshooting](#11-troubleshooting)
12. [References](#12-references)

---

## 1. Key findings

1. **The benchmark is valid.** No single image statistic separates faces from non-faces: sharpness reaches AUC 0.54 and mean brightness 0.57, where 0.5 is chance. So the models are not just learning which dataset an image came from (section 6.4).
2. **At a realistic class ratio the QNN falls apart.** With 10 background patches per face, the original QNN's precision drops from 0.963 to 0.586 (404 false alarms). The CNN goes from 0.998 to 0.957. ROC-AUC hides this, because it doesn't depend on class balance.
3. **Once capacity is matched, most of the CNN–QNN gap disappears.** In the original comparison the CNN's classifier had 16× more inputs and trained for more epochs. A classical layer with exactly the quantum layer's shape, also left at random weights, performs the same as the random quantum layer over 5 seeds: the balanced-accuracy difference is −0.0003 ± 0.022.
4. **A trained quantum layer is no better than a trained classical layer of the same shape** (balanced accuracy +0.001 ± 0.009, 1:10 precision +0.032 ± 0.064, both inside zero).
5. **Training the circuit does help it.** Compared with the frozen circuit on the same splits, accuracy rises by 0.013 ± 0.009 (p = 0.015) and 1:10 precision by 0.056 ± 0.043 (p = 0.022). That is about the same gain training gives the classical layer (0.962 → 0.974).

**Bottom line:** on this task, a 4-qubit quanvolutional layer behaves like a classical layer with the same number of outputs, trained or not. There is no quantum advantage here, and the large gap in the original comparison came from the classical model's much bigger classifier head.

---

## 2. Setup

**Requirements:** Python 3.10 or newer. A CPU is enough; no GPU and no quantum hardware are needed.

```bash
git clone https://github.com/NAVTEJJ/face_detection_benchmark.git
cd face_detection_benchmark

python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt
```

Tested with Python 3.11.8, torch 2.13.0 (CPU), PennyLane 0.45.1, scikit-learn 1.5.2, numpy 2.4.4, scipy 1.15.3 and matplotlib 3.10.8.

**Data downloads automatically on the first run. There is nothing to fetch by hand.**

| Dataset | Size | Saved to |
|---|---|---|
| LFW (funneled) | ~233 MB | `~/scikit_learn_data/lfw_home/` (by scikit-learn) |
| CIFAR-10 (python version) | ~170 MB | `data/cache/` |

Check that everything works before running anything long:

```bash
python tests/test_pipeline.py      # splits never overlap, 1:10 set built correctly
python tests/test_gate_g0.py       # validity gate catches known confounds
python tests/test_torch_quanv.py   # trainable circuit matches PennyLane (values + gradients)
```

Each should finish within a few seconds and end with `PASSED`.

---

## 3. How to run

Run every command from the repository root. Timings are for a 4-core laptop CPU, after the data has downloaded.

### Step 1: single benchmark run (about 2–3 minutes)

```bash
python main.py
```

In order, this:
1. loads LFW faces and CIFAR-10 non-faces (1:1), and splits them 80/20 with seed 42
2. runs **Gate G0**, the dataset validity check
3. trains the original CNN and the QNN, with the quantum circuit frozen at random weights
4. evaluates both on the balanced test set **and** on a 1:10 set (faces vs held-out background)
5. writes every figure, `outputs/metrics.json` and the console log to `outputs/`

### Step 2: validity gate, old vs new negatives (under a minute)

```bash
python -m src.baseline_g0
```

Compares the original 4-class non-face set (airplane, car, ship, truck) with all 10 CIFAR classes. Writes `outputs/gate_g0.json`.

### Step 3: controlled 5-seed study (about 40 minutes)

```bash
python study.py                 # trains 5 models x 5 seeds
python -m src.study_report      # mean ± 95% CI, paired comparisons, figure
```

This is the fair comparison (section 6.3). The runs take roughly 30 s each for the classical models and the frozen QNN, and about 6 min each for the trained quantum circuit. Progress is saved after every run, so if it stops, run it again and it skips what's already finished.

### Step 4: rebuild figures and the results page (seconds)

```bash
python -m src.plots_findings    # validity-gate and 1:10 figures
python make_presentation.py     # PRESENTATION.html, built from outputs/
```

For the PDF, open `PRESENTATION.html` in a browser and print to PDF.

### Optional: notebook

```bash
jupyter notebook notebooks/face_detection_benchmark.ipynb
```

This walks through the single run (step 1) cell by cell.

---

## 4. What each output file is

| File | Made by | Contents |
|---|---|---|
| `outputs/run_log.txt` | `main.py` | Full console output of the reported single run |
| `outputs/metrics.json` | `main.py` | Single-run metrics: G0, balanced, 1:10 |
| `outputs/gate_g0.json` | `src.baseline_g0` | Validity gate, old vs new negatives |
| `outputs/study_runs.jsonl` | `study.py` | One line per (seed, model) with all metrics |
| `outputs/study_log.txt` | `study.py` | Console output of the study |
| `outputs/study_summary.json` | `src.study_report` | Means, 95% CIs, paired differences, p-values |
| `outputs/fig_gate_g0.png` | `src.plots_findings` | Validity gate figure |
| `outputs/fig_imbalance.png` | `src.plots_findings` | Precision and ROC-AUC, balanced vs 1:10 |
| `outputs/fig_study.png` | `src.study_report` | All study models with confidence intervals |
| `outputs/roc_curves.png`, `confusion_matrices.png`, `training_curves.png`, `comparison_table.png`, `misclassified.png`, `class_balance.png` | `main.py` | Single-run figures |
| `PRESENTATION.html` / `.pdf` | `make_presentation.py` | Everything above on one page |

---

## 5. Repository layout

```
face_detection_benchmark/
├── main.py                     single benchmark run
├── study.py                    controlled 5-seed study
├── make_presentation.py        builds PRESENTATION.html from outputs/
├── PRESENTATION.html / .pdf    results page
├── requirements.txt
├── src/
│   ├── config.py               every setting (seeds, epochs, ratios, circuit shape)
│   ├── dataset.py              LFW + CIFAR loading, splits, held-out 1:10 pool
│   ├── cifar.py                CIFAR-10 reader (no torchvision needed)
│   ├── baseline_g0.py          validity gate
│   ├── plots.py                single-run figures
│   ├── plots_findings.py       gate and 1:10 figures
│   ├── study_report.py         study statistics and figure
│   ├── models/
│   │   ├── classical_12net.py  original CNN
│   │   ├── quanv_12net.py      QNN on PennyLane (frozen circuit, cached features)
│   │   └── torch_quanv.py      same circuit in PyTorch, trainable, checked vs PennyLane
│   └── benchmark/
│       ├── trainer.py          training loops for main.py
│       └── evaluator.py        metrics, 1:10 evaluation, latency
├── tests/
│   ├── test_pipeline.py
│   ├── test_gate_g0.py
│   └── test_torch_quanv.py
├── notebooks/face_detection_benchmark.ipynb
├── outputs/                    all results and figures (committed)
└── data/cache/                 downloaded CIFAR-10 and feature cache (not committed)
```

---

## 6. Method

### 6.1 Data

| | Source | Count | Preprocessing |
|---|---|---|---|
| Faces | LFW, people with ≥ 20 photos | 62 people, 3,023 images | grayscale, resized to 32×32, scaled to [0, 1] |
| Non-faces | CIFAR-10, all 10 classes | 3,023 (matched 1:1) | BT.601 grayscale, already 32×32 |
| 1:10 background pool | CIFAR-10 | 6,050 | in neither the training nor the test split |

Split: 80% train / 20% test, stratified. In the study, 10% of the training part is also held out as a validation set.

### 6.2 Models

**The quantum circuit** works on each 2×2 patch, stride 2:
- encoding: pixel value *x* → `RY(πx)` on one of 4 qubits
- 2 layers, each with `RX`, `RY`, `RZ` on every qubit, then a ring of CNOTs (0→1→2→3→0)
- readout: Pauli-Z expectation on each qubit, giving 4 feature maps of 16×16
- 24 circuit parameters, circuit depth 15

Every model below then uses **max-pool → dense 16 → dense 2**.

| Model | Front end | Trainable params | Used in |
|---|---|---|---|
| CNN, original | 3×3 conv, 16 channels (head gets 4,096 inputs) | 65,746 | `main.py`, study |
| CNN, matched | 2×2 conv, stride 2, 4 channels, tanh | 4,166 | study |
| CNN, matched, frozen | same conv, left at random init | 4,146 | study |
| QNN, frozen | quantum circuit, left at random init | 4,146 | `main.py`, study |
| QNN, trained | quantum circuit, trained end-to-end | 4,170 | study |

The **matched CNN** has the same shape as the quantum layer: one output per 2×2 patch per channel, 4 channels, with outputs kept in [−1, 1] like a Pauli-Z expectation. The **frozen** versions answer the question "is a random quantum projection better than a random classical one?"

### 6.3 Training

| | `main.py` (single run) | `study.py` (controlled) |
|---|---|---|
| Optimiser | Adam, lr 1e-3, batch 64 | Adam, lr 1e-3, batch 64 |
| Epochs | CNN 15, QNN 10 | 12 for every model |
| Epoch chosen by | last epoch | lowest validation loss |
| Seeds | 1 (42) | 5 (0–4), each a different split and init |
| Quantum circuit | frozen | frozen and trained versions |

### 6.4 Evaluation

- **Gate G0 (validity):** before training, three deliberately trivial baselines are run: Laplacian variance (sharpness), mean brightness, and logistic regression on raw pixels. If one number per image can separate the classes (AUC ≥ 0.90), the benchmark is measuring the dataset source, not faces. Only the single-number cues decide pass/fail, because a real task can be linearly separable in pixel space; `tests/test_gate_g0.py` shows this.
- **Balanced test (1:1):** accuracy, precision, recall, F1, ROC-AUC.
- **Deployment ratio (1:10):** every test face plus 10× as many background images that were never used in training. Reported: precision, recall, F1, ROC-AUC, **average precision** and false positives. Precision and average precision change with the class ratio; ROC-AUC doesn't, which is why ROC-AUC alone can't be the headline for a detector.
- **Study statistics:** mean ± 95% t-interval over 5 seeds. Models are compared **paired** on the same splits; "no clear difference" means the 95% interval of the paired difference includes zero.

---

## 7. Results in detail

### 7.1 Validity gate

| Single-number cue | Old negatives (4 vehicle classes) | All 10 classes |
|---|---|---|
| Sharpness (Laplacian variance) | AUC 0.658, Cohen's d 0.63 | **AUC 0.543, d 0.29** |
| Mean brightness | AUC 0.502 | AUC 0.566 |
| Raw-pixel logistic (not gated) | AUC 0.991 | AUC 0.993 |
| **Verdict** | PASS | **PASS** |

### 7.2 Single run (`main.py`, seed 42, original models)

| | CNN, original | QNN, frozen |
|---|---|---|
| Balanced accuracy | 0.992 | 0.955 |
| Balanced ROC-AUC | 1.000 | 0.989 |
| Precision, 1:1 → 1:10 | 0.998 → 0.957 | 0.963 → **0.586** |
| Average precision at 1:10 | 0.996 | 0.920 |
| False positives at 1:10 | 27 | **404** |
| Trainable parameters | 65,746 | 4,146 (+24 frozen) |

This run is **not** a fair comparison: the CNN has a 16× larger head and more epochs, and there is only one seed. Section 7.3 is the fair version.

### 7.3 Controlled study (5 seeds, mean ± 95% CI)

| Model | Params | Balanced acc. | ROC-AUC | 1:10 precision | 1:10 avg. precision | 1:10 false pos. |
|---|---|---|---|---|---|---|
| CNN, original | 65,746 | 0.991 ± 0.003 | 1.000 ± 0.000 | 0.896 ± 0.036 | 0.995 ± 0.001 | 70 ± 27 |
| CNN, matched | 4,166 | 0.974 ± 0.011 | 0.996 ± 0.003 | 0.766 ± 0.092 | 0.965 ± 0.020 | 186 ± 97 |
| CNN, matched, frozen | 4,146 | 0.962 ± 0.014 | 0.993 ± 0.004 | 0.716 ± 0.097 | 0.943 ± 0.033 | 237 ± 100 |
| QNN, frozen | 4,146 | 0.962 ± 0.010 | 0.993 ± 0.004 | 0.742 ± 0.026 | 0.958 ± 0.018 | 201 ± 27 |
| QNN, trained | 4,170 | 0.975 ± 0.006 | 0.997 ± 0.000 | 0.798 ± 0.032 | 0.976 ± 0.005 | 149 ± 30 |

**Paired comparisons** (difference = first minus second):

| Question | Balanced acc. | 1:10 precision | 1:10 avg. precision |
|---|---|---|---|
| Random quantum vs random classical features | −0.000 ± 0.022 | +0.026 ± 0.106 | +0.016 ± 0.044 |
| Trained quantum vs trained classical layer | +0.001 ± 0.009 | +0.032 ± 0.064 | +0.011 ± 0.021 |
| Trained circuit vs frozen circuit | **+0.013 ± 0.009** | **+0.056 ± 0.043** | **+0.018 ± 0.016** |

Only the last comparison excludes zero: training the circuit helps it, but a trained circuit is still no better than a trained classical layer of the same shape. In the trained runs the circuit parameters moved by up to 0.47 rad, so the gradients really do reach the circuit. All comparisons, with p-values, are in `outputs/study_summary.json`.

---

## 8. Implementation notes

- **The QNN in `main.py` has a frozen circuit.** Its features are computed once and cached, and the classifier trains on that cached tensor, so no gradient ever reaches the 24 circuit parameters. They are marked `requires_grad=False` so the parameter count is honest (4,146 trainable, not 4,170). The trained version is in `study.py`.
- **Trainable circuit (`src/models/torch_quanv.py`).** The same circuit is written as a 4-qubit statevector in plain PyTorch, so autograd reaches the circuit weights. It matches PennyLane to about 5×10⁻⁷ in value and 3×10⁻⁷ in gradient, compared against PennyLane's parameter-shift rule (`tests/test_torch_quanv.py`).
- **Fast quantum features.** `quanv_12net.py` evaluates every patch in one vectorised PennyLane call (parameter broadcasting) instead of rebuilding the circuit 256 times per image. It's about 175× faster, and on every cache miss it is checked against the per-patch version; it refuses to cache if they differ by more than 10⁻⁶.
- **Safe feature cache.** Cached features are keyed by a hash of the exact pixels, circuit weights and circuit shape, so changing the data or circuit can never reload stale features.
- **No torchvision.** CIFAR-10 is read straight from its pickled batches (`src/cifar.py`), which avoids torchvision/torch version mismatches.
- **The name "12-Net"** in the file names refers to the CNN cascade of Li et al. (2015), which uses 12×12 inputs. The networks here take 32×32 inputs and are 12-Net-*style*.

---

## 9. Limitations

- **Simulated, noiseless circuit.** All quantum results are exact simulation on a CPU. Real hardware noise could only make the QNN numbers worse.
- **Small circuit:** 4 qubits, 2 layers, 24 parameters. Nothing here says how larger circuits behave.
- **Balanced training only.** Every model trains at 1:1 and is only tested at 1:10.
- **Patch classification, not full detection.** Each 32×32 patch is classified on its own; there is no sliding-window detection over whole images yet.
- **Faces vs CIFAR objects.** Negatives are natural-object thumbnails, not background crops from the same photos as the faces.
- **12 epochs.** Several models pick their last epoch, which means they are still improving. With a fixed budget, the comparison partly reflects learning speed.

---

## 10. Next steps

1. Train at the skewed ratio and compare plain cross-entropy with focal loss.
2. Add depolarising noise to the circuit to see how much of the QNN result survives on realistic hardware.
3. Scale the circuit (more layers, larger patches), with the matched classical layer scaled alongside it.
4. Run as a sliding-window detector on full images (e.g. WIDER FACE) and report average precision.
5. Longer training with early stopping, so no model is cut off while still improving.

---

## 11. Troubleshooting

**`RuntimeError: operator torchvision::nms does not exist`**
torchvision was built for a different torch version. This project doesn't need torchvision, so either `pip uninstall torchvision` or make sure nothing imports it.

**LFW loads fewer than 3,023 faces (e.g. 2,537)**
An earlier download was interrupted, and scikit-learn reuses the half-extracted folder. Delete it and run again:
```bash
# macOS/Linux
rm -rf ~/scikit_learn_data/lfw_home/lfw_funneled ~/scikit_learn_data/lfw_home/joblib
# Windows (PowerShell)
Remove-Item -Recurse -Force "$env:USERPROFILE\scikit_learn_data\lfw_home\lfw_funneled", "$env:USERPROFILE\scikit_learn_data\lfw_home\joblib"
```
A complete LFW has 5,749 people and 13,233 images.

**LFW download is very slow or fails**
The archive is about 233 MB from figshare. If the download stops partway, delete `~/scikit_learn_data/lfw_home/lfw-funneled.tgz` and rerun.

**`study.py` was interrupted**
Run it again. Finished (seed, model) pairs are read from `outputs/study_runs.jsonl` and skipped. To start over, delete that file.

**Results differ slightly from the tables**
Training on CPU is seeded but can vary in the last decimal place across machines and library versions. The study's confidence intervals are the numbers to compare.

---

## 12. References

- G. B. Huang, M. Ramesh, T. Berg, E. Learned-Miller. *Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments.* UMass Amherst Tech. Report 07-49, 2007.
- A. Krizhevsky. *Learning Multiple Layers of Features from Tiny Images.* Tech. report, University of Toronto, 2009.
- M. Henderson, S. Shakya, S. Pradhan, T. Cook. *Quanvolutional neural networks: powering image recognition with quantum circuits.* Quantum Machine Intelligence 2, 2020.
- V. Bergholm et al. *PennyLane: Automatic differentiation of hybrid quantum-classical computations.* arXiv:1811.04968, 2018.
- H. Li, Z. Lin, X. Shen, J. Brandt, G. Hua. *A Convolutional Neural Network Cascade for Face Detection.* CVPR 2015.
- H. A. Rowley, S. Baluja, T. Kanade. *Neural Network-Based Face Detection.* IEEE TPAMI 20(1), 1998.
- T.-Y. Lin, P. Goyal, R. Girshick, K. He, P. Dollár. *Focal Loss for Dense Object Detection.* ICCV 2017.
