#!/usr/bin/env python3
import numpy as np
import pandas as pd
from ase.io import read
from mace.calculators import MACECalculator

import argparse

ap = argparse.ArgumentParser(
    description="Predict photophysical properties for a screening dataset with a trained X-MACE model.")
ap.add_argument("--xyz", required=True,
                help="Dataset, ASE .xyz with ground-state geometry and total charge")
ap.add_argument("--model", required=True, nargs="+",
                help="One or more trained models (.model). Run convert_model_to_cpu.py first for --device cpu")
ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="default: cpu")
ap.add_argument("--model-type", default="EmbeddingEMACE")
ap.add_argument("--n-energies", type=int, default=5,
                help="Must match --n_energies used at training time (default: 5)")
ap.add_argument("--out", default="predictions.csv", help="Output CSV (default: predictions.csv)")
args = ap.parse_args()

XYZ_FILE    = args.xyz
MODEL_FILES = args.model
DEVICE      = args.device
MODEL_TYPE  = args.model_type
N_ENG       = args.n_energies
OUT_CSV     = args.out

# 1) Calculator
calculator = MACECalculator(
    model_paths=MODEL_FILES,
    device=DEVICE,
    n_energies=N_ENG,
    model_type=MODEL_TYPE,
    compute_forces=False
)

# 2) Load structure
structures = list(read(XYZ_FILE, ":"))  # read all frames

results = []
for idx, mol in enumerate(structures):
    frame_number = idx + 1
    # energy for each of the N_ENG runs
    calculator.calculate(mol, properties=["energy"])
    energies = np.array(calculator.results["energy"])  # shape (n_samples, N_ENG)

    # average over runs
    mean_energies = energies.mean(axis=0)  # shape (N_ENG,)

    # take "source" if present
    src = mol.info.get("source", "unknown")

    # prepare output line
    entry = {'source': src, 'frame': frame_number}
    for i in range(N_ENG):
        entry[f'mean_energy_{i}'] = float(mean_energies[i])
    results.append(entry)


# 4) Export in csv 
cols = ['source', 'frame'] + [f'mean_energy_{i}' for i in range(N_ENG)]
df = pd.DataFrame(results, columns=cols)
df.to_csv(OUT_CSV, index=False)

print(f"Predictions saved in '{OUT_CSV}' ({len(df)} lines).")
print(f"Columns: {cols}")
