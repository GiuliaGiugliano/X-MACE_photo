#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from mace.calculators import MACECalculator

import argparse

ap = argparse.ArgumentParser(
    description="Evaluate a trained X-MACE model on a test set: scatter plot + MAE.")
ap.add_argument("--xyz", required=True,
                help="Test set, ASE .xyz (e.g. ../X-MACE_photo_data/test_set_dataset_file/"
                     "full_system_delta_s0_t1_test.xyz)")
ap.add_argument("--model", required=True,
                help="Trained model (.model). Use convert_model_to_cpu.py first if --device cpu")
ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="default: cpu")
ap.add_argument("--model-type", default="EmbeddingEMACE")
ap.add_argument("--n-energies", type=int, default=5,
                help="Must match --n_energies used at training time (default: 5)")
ap.add_argument("--label", default=r"\Delta \mathrm{E}_{(T_1 - S_0)}",
                help="LaTeX label for the plotted property")
ap.add_argument("--outdir", default=".", help="Where to write scatter_test.png and testset_MAE.txt")
args = ap.parse_args()

XYZ_FILE   = args.xyz
MODEL_FILE = args.model
DEVICE     = args.device
MODEL_TYPE = args.model_type
N_ENG      = args.n_energies
OUTDIR     = Path(args.outdir); OUTDIR.mkdir(parents=True, exist_ok=True)

# 1) Calculator (energies only)
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
    # prediction
    calculator.calculate(mol, properties=["energy"])
    e_pred = np.array(calculator.results["energy"]).ravel()

    # references
    raw = mol.info.get("REF_energy")
    if raw is None:
        raise KeyError(f"[{idx}] REF_energy missing")
    ref_arr = np.array(json.loads(raw) if isinstance(raw, str) else raw).ravel()

    if e_pred.shape != ref_arr.shape:
        raise ValueError(f"[{idx}] Shape mismatch: pred {e_pred.shape} vs ref {ref_arr.shape}")

    avg_preds.append(e_pred.mean())
    avg_refs.append(ref_arr.mean())

avg_preds = np.array(avg_preds)
avg_refs  = np.array(avg_refs)

print(f"Average values computed for {len(avg_preds)} molecules")

# 4) MAE
mae = np.mean(np.abs(avg_preds - avg_refs))
print(f"Test set MAE: {mae:.6f} eV")

# 5) Scatter plot
plt.figure(figsize=(10, 10))

plt.scatter(avg_refs, avg_preds,
            s=20, alpha=0.7, color="limegreen")

# Identity line
vmin = min(avg_refs.min(), avg_preds.min())
vmax = max(avg_refs.max(), avg_preds.max())
plt.plot([vmin, vmax], [vmin, vmax], '--', color='gray', lw=1)

plt.xlabel(rf"Reference ${args.label}$ / eV", fontsize=30)
plt.ylabel(rf"Predicted ${args.label}$ / eV", fontsize=30)
plt.title(f"MAE = {mae:.3f} eV", fontsize=32)      # print the test MAE as title

plt.grid(True, ls=":", lw=0.5)
plt.gca().set_aspect("equal", adjustable="box")

plt.tight_layout()
plt.savefig(OUTDIR / "scatter_test.png", dpi=1200)
plt.close()

# 6) Save png of the scatter plot for the test set predictions vs references
(OUTDIR / "testset_MAE.txt").write_text(f"Mean Absolute Error (MAE): {mae:.6f} eV\n")
print(f"Saved: {OUTDIR}/scatter_test.png and {OUTDIR}/testset_MAE.txt")
