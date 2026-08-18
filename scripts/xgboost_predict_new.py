#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
The input file must be in .xyz format with geometry and charge.
The model loads the assets written by classification.py:
• model.json    (XGBoost model)
• scaler.pkl    (StandardScaler for SOAP)
• species.json  (list of chemical elements)
It returns the predicted class and the associated probabilities for the two classes:
class 0 → f ≤ 0.03
class 1 → f > 0.03
"""

import argparse, json, joblib
import numpy as np
from pathlib import Path
from ase.io import read
from dscribe.descriptors import SOAP
import xgboost as xgb

SOAP_PAR = dict(r_cut=5.0, n_max=8, l_max=6, sigma=0.3, periodic=False)

def load_model_and_scaler(assets: Path):
    """Load the assets written by classification.py (model.json, scaler.pkl,
    species.json). Filenames must match what classification.py saves."""
    bst = xgb.Booster()
    bst.load_model(str(assets / "model.json"))

    scaler = joblib.load(assets / "scaler.pkl")

    with open(assets / "species.json") as f:
        species = json.load(f)

    return bst, scaler, species

def predict_class(xyz_path: Path, assets: Path):
    mol = read(xyz_path)
    bst, scaler, species = load_model_and_scaler(assets)

    soap = SOAP(species=species, **SOAP_PAR)
    X = soap.create(mol, n_jobs=-1).mean(axis=0).reshape(1, -1)
    if X.shape[1] != scaler.n_features_in_:
        raise SystemExit(
            f"Descriptor length {X.shape[1]} does not match the trained scaler "
            f"({scaler.n_features_in_}). SOAP_PAR here must match the values used "
            f"in classification.py when the model was trained.")
    X_scaled = scaler.transform(X)

    dnew = xgb.DMatrix(X_scaled)
    proba = bst.predict(dnew)[0]
    predicted_class = np.argmax(proba)

    print(f"Predicted class: {predicted_class}")
    print(f"Probability → class 0 (f ≤ 0.03): {proba[0]:.4f}, class 1 (f > 0.03): {proba[1]:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="XYZ file of the new molecule")
    ap.add_argument("--assets", default=".",
                    help="Directory holding model.json, scaler.pkl and species.json "
                         "(the --outdir used by classification.py)")
    args = ap.parse_args()
    predict_class(Path(args.input), Path(args.assets))

