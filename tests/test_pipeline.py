"""Integration check with synthetic 'faces' so we exercise the wiring
without waiting on LFW. Validates disjointness, ratios, and shapes."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import src.dataset as ds

# fake LFW: 500 synthetic images
def fake_lfw():
    rng = np.random.default_rng(7)
    return torch.tensor(rng.random((500,1,32,32)), dtype=torch.float32)
ds._load_lfw = fake_lfw

train_loader, test_loader, meta = ds.load_data()
X_imb, y_imb = ds.build_imbalanced_set(meta)

print("\n--- assertions ---")
n_pos, n_neg = int((y_imb==1).sum()), int((y_imb==0).sum())
print(f"imbalanced set: {n_pos} faces / {n_neg} bg  ratio 1:{n_neg/n_pos:.1f}")
assert n_neg == n_pos * 10, f"ratio wrong: {n_neg} vs {n_pos*10}"

# every imbalanced face must come from the test split
test_faces = meta["X_test"][meta["y_test"]==1]
imb_faces  = X_imb[y_imb==1]
tf = {f.tobytes() for f in test_faces}
assert all(f.tobytes() in tf for f in imb_faces), "a face leaked in from outside the test split"
print("all imbalanced faces come from the held-out test split: OK")

# background must appear in NEITHER train nor test
seen = {r.tobytes() for r in meta["X_train"]} | {r.tobytes() for r in meta["X_test"]}
imb_bg = X_imb[y_imb==0]
overlap = sum(1 for b in imb_bg if b.tobytes() in seen)
print(f"background patches also present in train/test: {overlap}")
assert overlap == 0, f"{overlap} background patches were seen during training"
print("background pool is disjoint from both splits: OK")

# train/test disjoint
tr = {r.tobytes() for r in meta["X_train"]}
te = {r.tobytes() for r in meta["X_test"]}
assert not (tr & te), "train/test overlap"
print("train and test splits are disjoint: OK")
print("\nALL PIPELINE ASSERTIONS PASSED")
