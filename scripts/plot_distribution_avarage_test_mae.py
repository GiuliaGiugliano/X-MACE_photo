#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from mace.calculators import MACECalculator

# ——— CONFIG —————————————————————————————————————————————————————
XYZ_FILE   = "full_system_delta_s0_t1.xyz"   #put the test set dataset
MODEL_FILE = "file.model" #put the file.model from the training phase
DEVICE     = "cuda"            # or "cpu"
MODEL_TYPE = "EmbeddingEMACE"
N_ENG      = 2
# ————————————————————————————————————————————————————————————————

# 1) Calcolatore (solo energie)
calculator = MACECalculator(
    model_paths=[MODEL_FILE],
    device=DEVICE,
    n_energies=N_ENG,
    model_type=MODEL_TYPE,
    compute_forces=False,
)

# 2) Load xyz structure
db = read(XYZ_FILE, ":")

avg_preds, avg_refs = [], []

# 3) Loop molecule
for idx, mol in enumerate(db):
    # predizione
    calculator.calculate(mol, properties=["energy"])
    e_pred = np.array(calculator.results["energy"]).ravel()

    # references
    raw = mol.info.get("REF_energy")
    if raw is None:
        raise KeyError(f"[{idx}] REF_energy mancante")
    ref_arr = np.array(json.loads(raw) if isinstance(raw, str) else raw).ravel()

    if e_pred.shape != ref_arr.shape:
        raise ValueError(f"[{idx}] Shape mismatch: pred {e_pred.shape} vs ref {ref_arr.shape}")

    avg_preds.append(e_pred.mean())
    avg_refs.append(ref_arr.mean())

avg_preds = np.array(avg_preds)
avg_refs  = np.array(avg_refs)

print(f"✅ Avarage values computed for  {len(avg_preds)} molecules")

# 4) MAE
mae = np.mean(np.abs(avg_preds - avg_refs))
print(f"📊 MAE del test set: {mae:.6f} eV")

# 5) Scatter plot
plt.figure(figsize=(10, 10))

plt.scatter(avg_refs, avg_preds,
            s=20, alpha=0.7, color="limegreen")

# Identity line
vmin = min(avg_refs.min(), avg_preds.min())
vmax = max(avg_refs.max(), avg_preds.max())
plt.plot([vmin, vmax], [vmin, vmax], '--', color='gray', lw=1)

plt.xlabel(r"Reference $\Delta \mathrm{E}_{(T_1 - S_0)}$ / eV", fontsize=30) #put the x-reference label
plt.ylabel(r"Predicted $\Delta \mathrm{E}_{(T_1 - S_0)}$ / eV", fontsize=30) #put the y-predicted label
plt.title(f"MAE = {mae:.3f} eV", fontsize=32)      # print the test MAE as title

plt.grid(True, ls=":", lw=0.5)
plt.gca().set_aspect("equal", adjustable="box")

plt.tight_layout()
plt.savefig("scatter_test.png", dpi=1200)
plt.show()

# 6) Save png of the scatter plot for the test set predictions vs references
Path("testset_MAE.txt").write_text(f"Mean Absolute Error (MAE): {mae:.6f} eV\n")
print("✓ Save: scatter_test.png  and  testset_MAE.txt")
