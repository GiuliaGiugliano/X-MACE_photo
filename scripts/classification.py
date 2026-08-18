#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, random
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from ase.io import read, iread
from dscribe.descriptors import SOAP
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, f1_score, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.model_selection import KFold
import xgboost as xgb
import joblib

# -------------------------- PARAMETERS -------------------------- #
SOAP_PAR   = dict(r_cut=5.0, n_max=8, l_max=6, sigma=0.3, periodic=False)
NUM_CLASS  = 2
CV_FOLDS   = 7
RSEED      = 42
BINS       = [0.0, 0.03, 10.0]
N_SEARCH   = 50          # random configurations to try
EARLY_STOP = 30           # round without improving
# -------------------------------------------------------------------- #

def xgb_base_params(device: str) -> dict:
    """Base XGBoost parameters.

    `gpu_id` / `predictor` were removed in XGBoost 3.1; `device` replaces both.
    """
    return dict(objective="multi:softprob", num_class=NUM_CLASS,
                tree_method="hist", device=device, seed=RSEED)


def osc_class(v: float) -> int:
    return 0 if v <= BINS[1] else 1

def detect_species(xyz_paths):
    elems = set()
    for p in xyz_paths:
        for mol in iread(p, ":"):
            elems.update(mol.get_chemical_symbols())
    return sorted(elems)

def load_xyz(path: Path):
    mols, y = [], []
    for mol in read(path, ":"):
        raw = mol.info["REF_energy"]
        arr = json.loads(raw.lstrip("_JSON").strip()) if isinstance(raw, str) else raw
        mols.append(mol)
        y.append(osc_class(float(arr[0][0])))
    return mols, np.array(y, int)

def soap_pool(mols, soap):
    return np.vstack([soap.create(m, n_jobs=-1).mean(axis=0).astype(np.float32) for m in mols])

def sample_weights(labels):
    cnt, tot = Counter(labels), len(labels)
    return np.array([tot / (NUM_CLASS * cnt[l]) for l in labels], float)

def weighted_obj(preds, dtrain):
    y   = dtrain.get_label().astype(int)
    w   = sample_weights(y)[:, None]
    p   = np.exp(preds.reshape(-1, NUM_CLASS) - preds.max(1)[:, None])
    p  /= p.sum(1, keepdims=True)
    oh  = np.zeros_like(p); oh[np.arange(y.size), y] = 1
    grad = (p - oh) * w
    hess = p * (1 - p) * w
    return grad.ravel(), hess.ravel()

# ------------- RANDOM SEARCH ------------- #
def random_cfg():
    return {
        "max_depth"       : random.choice([8, 9, 10, 11, 12]),
        "learning_rate"   : random.choice([0.05, 0.1, 0.15]),
        "min_child_weight": random.choice([1, 2, 3, 5]),
        "subsample"       : random.choice([0.7, 0.8, 0.9, 1.0]),
        "colsample_bytree": random.choice([0.6, 0.7, 0.8, 1.0]),
        "reg_alpha"       : random.choice([0, 0.1, 0.5]),
        "reg_lambda"      : random.choice([1.0, 2.0, 5.0]),
        "n_estimators"    : random.choice([300, 400, 500]),
    }

def evaluate_cfg(cfg, X, y, device):
    kf = KFold(CV_FOLDS, shuffle=True, random_state=RSEED)
    f1s = []
    for tr, val in kf.split(X, y):
        dtr, dval = xgb.DMatrix(X[tr], label=y[tr]), xgb.DMatrix(X[val], label=y[val])
        p = dict(**xgb_base_params(device),
                 **{k:v for k,v in cfg.items() if k!="n_estimators"})
        bst = xgb.train(p, dtr, cfg["n_estimators"], obj=weighted_obj,
                        evals=[(dval,"val")], verbose_eval=False, early_stopping_rounds=EARLY_STOP)
        pred = np.argmax(bst.predict(dval),1)
        f1s.append(f1_score(y[val], pred, average="macro"))
    return np.mean(f1s)

# ----------------------------- MAIN ----------------------------- #
def main(train_xyz: Path, test_xyz: Path, outdir: Path, device: str):
    outdir.mkdir(parents=True, exist_ok=True)
    random.seed(RSEED); np.random.seed(RSEED)

    species = detect_species([train_xyz, test_xyz])
    soap    = SOAP(species=species, **SOAP_PAR)

    mol_tr, y_tr = load_xyz(train_xyz)
    mol_te, y_te = load_xyz(test_xyz)

    X_tr_raw, X_te_raw = soap_pool(mol_tr, soap), soap_pool(mol_te, soap)
    scaler = StandardScaler().fit(X_tr_raw)
    X_tr, X_te = scaler.transform(X_tr_raw), scaler.transform(X_te_raw)

    # --------- random search ---------
    best_f1, best_cfg = -1, None
    print(f"Random search over {N_SEARCH} configurations...")
    for i in range(1, N_SEARCH+1):
        cfg = random_cfg()
        f1  = evaluate_cfg(cfg, X_tr, y_tr, device)
        if f1 > best_f1:
            best_f1, best_cfg = f1, cfg
        print(f"[{i:03}/{N_SEARCH}] F1 = {f1:.4f} | cfg = {cfg}")
    print(f"\nBest configuration: {best_cfg}  (macro-F1 = {best_f1:.4f})")

    # --------- final training ---------
    dtrain, dtest = xgb.DMatrix(X_tr, label=y_tr), xgb.DMatrix(X_te, label=y_te)
    params = dict(**xgb_base_params(device),
                  **{k:v for k,v in best_cfg.items() if k!="n_estimators"})

    print("\nFinal training...")
    bst = xgb.train(params, dtrain, best_cfg["n_estimators"], obj=weighted_obj,
                    evals=[(dtrain,"train")], verbose_eval=False)
    print("Done.\n")

    # ------------------- REPORT & PLOT ------------------- #
    y_prob = bst.predict(dtest)[:,1]
    y_pred = np.argmax(bst.predict(dtest),1)

    # report
    rep = classification_report(y_te, y_pred, digits=4)
    print("=== Classification Report (test) ===\n", rep)
    (outdir / "classification_report.txt").write_text(rep)

    # Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(
        y_te, y_pred, display_labels=["f ≤ 0.03","f > 0.03"],
        cmap="Greens", colorbar=False)
    plt.title("Confusion Matrix"); plt.tight_layout()
    plt.savefig(outdir / "confusion_matrix.png", dpi=1200); plt.close()

    # ROC
    fpr, tpr, _ = roc_curve(y_te, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(); plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1],[0,1],"--"); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC Curve"); plt.legend(); plt.tight_layout()
    plt.savefig(outdir / "roc_curve.png", dpi=1200); plt.close()

    # Precision-Recall
    prec, rec, _ = precision_recall_curve(y_te, y_prob)
    ap = average_precision_score(y_te, y_prob)
    plt.figure(); plt.plot(rec, prec, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall")
    plt.legend(); plt.tight_layout(); plt.savefig(outdir / "precision_recall_curve.png", dpi=1200); plt.close()

    # Learning curve (logloss)
    evals = {}
    _ = xgb.train(params, dtrain, best_cfg["n_estimators"], obj=weighted_obj,
                  evals=[(dtrain,"train"),(dtest,"test")], evals_result=evals,
                  verbose_eval=False)
    tr_loss = evals["train"]["mlogloss"]; te_loss = evals["test"]["mlogloss"]
    rounds  = range(1, len(tr_loss)+1)
    plt.figure(); plt.plot(rounds, tr_loss, label="Train"); plt.plot(rounds, te_loss, label="Test")
    plt.xlabel("Round"); plt.ylabel("Log Loss"); plt.title("Learning Curve")
    plt.legend(); plt.tight_layout(); plt.savefig(outdir / "learning_curve.png", dpi=1200); plt.close()

    # ------------------- SAVE ASSETS ------------------- #
    bst.save_model(str(outdir / "model.json"))
    joblib.dump(scaler, outdir / "scaler.pkl")
    with open(outdir / "species.json","w") as f:
        json.dump(species, f)
    print(f"Saved model, scaler, species and plots to {outdir}/")

# ------------------ CLI ------------------ #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="Training set, ASE .xyz")
    ap.add_argument("--test",  required=True, help="Test set, ASE .xyz")
    ap.add_argument("--outdir", default=".",
                    help="Directory for model.json, scaler.pkl, species.json and plots")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="XGBoost device (default: cpu)")
    args = ap.parse_args()
    main(Path(args.train), Path(args.test), Path(args.outdir), args.device)
