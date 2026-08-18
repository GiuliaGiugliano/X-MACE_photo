#!/usr/bin/env python3
import numpy as np
import pandas as pd
from ase.io import read
from mace.calculators import MACECalculator

# ——— CONFIG —————————————————————————————————————————————————————
XYZ_FILE    = "virtual_screening_dataset.xyz" #put your dataset 
MODEL_FILES = ["file.model"] # put the cpu model file
DEVICE      = "cuda"            # oppure "cpu"
MODEL_TYPE  = "EmbeddingEMACE"
N_ENG       = 5                   # numero di energie da predire (≥2)
OUT_CSV     = "predictions.csv"
# ————————————————————————————————————————————————————————————————

# 1) Calculator
calculator = MACECalculator(
    model_paths=MODEL_FILES,
    device=DEVICE,
    n_energies=N_ENG,
    model_type=MODEL_TYPE,
    compute_forces=False
)

# 2) Load structure
structures = list(read(XYZ_FILE, ":"))  # legge tutti i frame

results = []
for idx, mol in enumerate(structures):
    frame_number = idx + 1
    # calcola l'energia per ognuno dei N_ENG run
    calculator.calculate(mol, properties=["energy"])
    energies = np.array(calculator.results["energy"])  # shape (n_samples, N_ENG)

    # avarage on each run
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
print(f"Col: {cols}")
