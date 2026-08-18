#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from mace.calculators import MACECalculator

# ——— CONFIG —————————————————————————————————————————————————————
XYZ_FILE     = "full_system_delta_s0_t1_train.xyz" #put the training set dataset 
MODEL_FILE   = "file.model"  #put the file.model from the training phase 
DEVICE       = "cuda"            # or "cpu"
MODEL_TYPE   = "EmbeddingEMACE"
N_ENG        = 2
VALID_IDX_F  = "valid_indices_300.txt"
METRICS_FILE = "results/file_train.txt"   # in the results directory there is a file_train.txt where read the MAE
# ————————————————————————————————————————————————————————————————

def load_mae(path: str | Path) -> float:
    """Givefloat associated to  'mae_e' in the last line of the file."""
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
        raise KeyError(f"[{idx}] REF_energy mancante")
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

print(f"✅ Train: {len(avg_pred_train)} molecule, Validation: {len(avg_pred_val)} molecule")

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
plt.savefig("train_scatter.png", dpi=1200)
plt.show()
print("✓ Save: train_scatter.png")
