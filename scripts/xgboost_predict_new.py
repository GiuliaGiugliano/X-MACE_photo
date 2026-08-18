#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
The input file must be in .xyz format with geometry and charge.
The model will automatically load:
• model_fstrength.json (XGBoost model)
• scaler_fstrength.pkl (StandardScaler for SOAP)
• species.json (list of chemical elements)
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

def load_model_and_scaler():
    bst = xgb.Booster()
    bst.load_model("model_fstrength.json")

    scaler = joblib.load("scaler_fstrength.pkl")

    with open("species.json") as f:
        species = json.load(f)

    return bst, scaler, species

def predict_class(xyz_path: Path):
    mol = read(xyz_path)
    bst, scaler, species = load_model_and_scaler()

    soap = SOAP(species=species, **SOAP_PAR)
    X = soap.create(mol, n_jobs=-1).mean(axis=0).reshape(1, -1)
    X_scaled = scaler.transform(X)

    dnew = xgb.DMatrix(X_scaled)
    proba = bst.predict(dnew)[0]
    predicted_class = np.argmax(proba)

    print(f"Predicted class: {predicted_class}")
    print(f"Probability → class 0 (f ≤ 0.03): {proba[0]:.4f}, class 1 (f > 0.03): {proba[1]:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="File XYZ della nuova molecola")
    args = ap.parse_args()
    predict_class(Path(args.input))

