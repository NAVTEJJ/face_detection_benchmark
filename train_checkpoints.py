"""Train the five study models once (seed 0) and save them for the notebook.

Uses exactly the study's data split, initialisation and training loop, so the
saved models reproduce study.py's seed-0 results. The notebook then loads them
instead of training live.

Writes
  checkpoints/<arm>.pt          model weights
  checkpoints/scores.npz        each model's face probability on the seed-0
                                test set and on the full held-out background pool

    python train_checkpoints.py
"""

import json
import os
import time

import numpy as np
import torch

from src.demo import ARMS, build_model, load_demo_data, predict_proba
from study import FeatureHead, fit, metrics

OUT = "checkpoints"
SEED = 0


def main():
    os.makedirs(OUT, exist_ok=True)
    d = load_demo_data(SEED)
    Xtr, ytr = torch.tensor(d["X_train"]), torch.tensor(d["y_train"])
    Xva, yva = torch.tensor(d["X_val"]), torch.tensor(d["y_val"])

    scores = {"y_test": d["y_test"]}
    study = {}
    if os.path.exists("outputs/study_runs.jsonl"):
        with open("outputs/study_runs.jsonl") as f:
            study = {r["arm"]: r for r in map(json.loads, f) if r["seed"] == SEED}

    for arm in ARMS:
        t0 = time.time()
        torch.manual_seed(SEED)
        model = build_model(arm)
        path = f"{OUT}/{arm}.pt"
        if os.path.exists(path):            # resumable: reuse models already saved
            model.load_state_dict(torch.load(path, weights_only=True))
        elif arm == "qnn_frozen":
            # circuit never changes, so train the head on precomputed features (as study.py does)
            with torch.no_grad():
                Ftr, Fva = model.features(Xtr), model.features(Xva)
            fit(FeatureHead(model), Ftr, ytr, Fva, yva, SEED)
            torch.save(model.state_dict(), path)
        else:
            fit(model, Xtr, ytr, Xva, yva, SEED)
            torch.save(model.state_dict(), path)

        p_test = predict_proba(model, d["X_test"])
        p_bg = predict_proba(model, d["background"])
        scores[f"{arm}_test"] = p_test.astype(np.float32)
        scores[f"{arm}_bg"] = p_bg.astype(np.float32)

        acc = metrics(d["y_test"], p_test)["accuracy"]
        ref = study.get(arm, {}).get("balanced", {}).get("accuracy")
        note = f" (study seed 0: {ref:.4f})" if ref is not None else ""
        print(f"{arm:19s} test acc {acc:.4f}{note}  {time.time() - t0:.0f}s", flush=True)

    np.savez_compressed(f"{OUT}/scores.npz", **scores)
    print("saved checkpoints/")


if __name__ == "__main__":
    main()
