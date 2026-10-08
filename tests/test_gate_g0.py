"""Does G0 actually detect a confound when one is present?

Three constructed splits with known answers. If the gate cannot flag a
deliberately blurred positive class, it cannot be trusted on real data.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F
from src.baseline_g0 import run_gate, verdict

rng = np.random.default_rng(11)

def blur(x, k=4):
    t = torch.tensor(x)
    t = F.avg_pool2d(t, k)
    return F.interpolate(t, size=(32, 32), mode="bilinear", align_corners=False).numpy()

n = 400
cases = []

# 1. CONFOUNDED: positives blurred, negatives sharp. Sharpness alone separates.
pos = blur(rng.random((n,1,32,32)).astype(np.float32))
neg = rng.random((n,1,32,32)).astype(np.float32)
cases.append(("blur confound (expect FAIL)", pos, neg, "FAIL"))

# 2. CONFOUNDED: positives systematically brighter. Mean intensity separates.
pos = np.clip(rng.random((n,1,32,32)).astype(np.float32)*0.4 + 0.55, 0, 1)
neg = np.clip(rng.random((n,1,32,32)).astype(np.float32)*0.4 + 0.05, 0, 1)
cases.append(("intensity confound (expect FAIL)", pos, neg, "FAIL"))

# 3. CLEAN: same marginal statistics, differ only in spatial structure.
base = rng.random((2*n,1,32,32)).astype(np.float32)
pos, neg = base[:n].copy(), base[n:].copy()
for i in range(n):                      # structure, without shifting mean or sharpness
    pos[i,0] = np.sort(pos[i,0], axis=0)
    neg[i,0] = np.sort(neg[i,0], axis=1)
cases.append(("structure only (expect not FAIL)", pos, neg, "NOT_FAIL"))

print()
ok = True
for label, p, q, expected in cases:
    X = np.concatenate([p, q]); y = np.concatenate([np.ones(len(p),int), np.zeros(len(q),int)])
    r = run_gate(X, y, label=label)
    v, why = verdict(r)
    passed = (v == expected) if expected != "NOT_FAIL" else (v != "FAIL")
    ok &= passed
    print(f"  {label}")
    print(f"    sharpness AUC {r['sharpness_auc']:.3f} | intensity AUC {r['intensity_auc']:.3f} "
          f"| pixel AUC {r['pixel_auc']:.3f}")
    print(f"    verdict {v}  -> {'OK' if passed else 'WRONG, expected ' + expected}\n")

print("GATE G0 SELF-TEST:", "PASSED" if ok else "FAILED")
sys.exit(0 if ok else 1)
