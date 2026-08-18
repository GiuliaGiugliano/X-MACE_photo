#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from mace.calculators import MACECalculator

import argparse

ap = argparse.ArgumentParser(
    description="Evaluate a trained X-MACE model on the training/validation split.")
ap.add_argument("--xyz", required=True,
                help="Training set, ASE .xyz (e.g. ../X-MACE_photo_data/training_set_dataset_file/"
                     "full_system_delta_s0_t1_train.xyz)")
ap.add_argument("--model", required=True, help="Trained model (.model)")
ap.add_argument("--valid-indices", required=True,
                help="Text file of validation indices, one per line, as written by the training run")
ap.add_argument("--metrics", required=True,
                help="Training metrics file, e.g. results/<name>_train.txt (mae_e is read from the last line)")
ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="default: cpu")
ap.add_argument("--model-type", default="EmbeddingEMACE")
ap.add_argument("--n-energies", type=int, default=5,
                help="Must match --n_energies used at training time (default: 5)")
ap.add_argument("--outdir", default=".", help="Where to write train_scatter.png")
args = ap.parse_args()

XYZ_FILE     = args.xyz
MODEL_FILE   = args.model
DEVICE       = args.device
MODEL_TYPE   = args.model_type
N_ENG        = args.n_energies
VALID_IDX_F  = args.valid_indices
METRICS_FILE = args.metrics
OUTDIR       = Path(args.outdir); OUTDIR.mkdir(parents=True, exist_ok=True)

def load_mae(path: str | Path) -> float:
    """Return the float associated with 'mae_e' on the last line of the file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        last = fh.readlines()[-1]

    try:                                # case JSON valid
        return float(json.loads(last.strip())["mae_e"])
    except Exception:
        m = re.search(r'"mae_e"\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', last)
        if not m:
            raise ValueError(f'"mae_e" not found in {path}')
        return float(m.group(1))

# ---------- calculator ----------------------------------------------------
calculator = MACECalculator(
    model_paths=[MODEL_FILE],
    device=DEVICE,
    n_energies=N_ENG,
    model_type=MODEL_TYPE,
    compute_forces=False,
)

# ---------- index validation ---------------------------------------------
valid_idx = np.loadtxt(VALID_IDX_F, dtype=int).tolist()

# ---------- load structure ----------------------------------------------
db = read(XYZ_FILE, ":")

avg_pred_train, avg_ref_train = [], []
avg_pred_val,  avg_ref_val   = [], []

for idx, mol in enumerate(db):
    calculator.calculate(mol, properties=["energy"])
    e_pred = np.array(calculator.results["energy"]).ravel()

    raw = mol.info.get("REF_energy")
    if raw is None:
        raise KeyError(f"[{idx}] REF_energy missing")
    ref_arr = np.array(json.loads(raw) if isinstance(raw, str) else raw).ravel()

    if e_pred.shape != ref_arr.shape:
        raise ValueError(f"[{idx}] Shape mismatch: pred {e_pred.shape} vs ref {ref_arr.shape}")

    avg_pred, avg_ref = e_pred.mean(), ref_arr.mean()

    if idx in valid_idx:
        avg_pred_val.append(avg_pred)
        avg_ref_val.append(avg_ref)
    else:
        avg_pred_train.append(avg_pred)
        avg_ref_train.append(avg_ref)

avg_pred_train = np.array(avg_pred_train)
avg_ref_train  = np.array(avg_ref_train)
avg_pred_val   = np.array(avg_pred_val)
avg_ref_val    = np.array(avg_ref_val)

print(f"Train: {len(avg_pred_train)} molecules, Validation: {len(avg_pred_val)} molecules")

# ---------- MAE from metrics file ------------------------------------------
mae_val = load_mae(METRICS_FILE)

# ---------- scatter plot ---------------------------------------------------
plt.figure(figsize=(10, 10))

plt.scatter(avg_ref_train, avg_pred_train,
            s=20, alpha=0.7, color="skyblue", label="Train")
plt.scatter(avg_ref_val,   avg_pred_val,
            s=20, alpha=0.7, color="orange",  label="Validation")

vmin = min(avg_ref_train.min(), avg_ref_val.min(),
           avg_pred_train.min(), avg_pred_val.min())
vmax = max(avg_ref_train.max(), avg_ref_val.max(),
           avg_pred_train.max(), avg_pred_val.max())
plt.plot([vmin, vmax], [vmin, vmax], '--', color='gray', lw=1)

plt.xlabel(r"Reference $\Delta \mathrm{E}_{(T_1 - S_0)}$ / eV", fontsize=30) #put the x-reference label
plt.ylabel(r"Predicted $\Delta \mathrm{E}_{(T_1 - S_0)}$ / eV", fontsize=30) #put the y-predicted label
plt.title(f"MAE = {mae_val:.3f} eV", fontsize=32)      # put the training MAE as title 
plt.grid(True, ls=":", lw=0.5)
plt.gca().set_aspect("equal", adjustable="box")
plt.legend(fontsize=18, markerscale=1.5)

plt.tight_layout()
plt.savefig(OUTDIR / "train_scatter.png", dpi=1200)
plt.close()
print("✓ Save: train_scatter.png")
