# X‑MACE for photosensitiser discovery

X‑MACE is a deep learning framework for modelling excited-state potential energy
surfaces, with particular emphasis on regions near conical intersections. It builds on
the Message Passing Atomic Cluster Expansion (MACE) architecture by incorporating Deep
Sets, allowing smooth representations of inherently non-smooth energy landscapes.

This repository contains an adapted version of X‑MACE aimed at identifying novel
photosensitisers for photodynamic therapy. From a ground-state XYZ structure plus the
total molecular charge, the model predicts:

- UV-VIS spectra and first-excitation oscillator strength
- Intersystem crossing rates (log k<sub>ISC</sub>)
- HOMO–LUMO gaps
- S<sub>0</sub>–T<sub>1</sub> energy differences (relevant to the type II mechanism)
- S<sub>0</sub>–radical anion and T<sub>1</sub>–cation energy differences (type I mechanism)
- Transition dipole moments

---

## Repository organisation

The project is split across **two** repositories. All *code* lives here; only the
*datasets* live in the data repository, because they are too large (~294 MB) to be
practical in the code repository.

| Repository | Contents |
|---|---|
| [`X-MACE_photo`](https://github.com/GiuliaGiugliano/X-MACE_photo) (this repo) | X‑MACE source package, training entry point, and all analysis/prediction scripts |
| [`X-MACE_photo_data`](https://github.com/GiuliaGiugliano/X-MACE_photo_data) | Training, test and virtual-screening datasets in ASE `.xyz` format |

Layout of this repository:

```text
mace/            X-MACE source package (installed by `pip install .`)
scripts/         Training entry point and workflow scripts
  run_train.py
  classification.py
  convert_model_to_cpu.py
  plot_distribution_avarage_train_mae.py
  plot_distribution_avarage_test_mae.py
  xgboost_predict_new.py
  predict.py
```

Training runs write `logs/`, `checkpoints/` and `results/` into the current working
directory. These are generated at run time and are not tracked in git.

> **Note on paths.** Every command below is run from the **root of this repository**
> unless stated otherwise. Dataset paths are written as
> `../X-MACE_photo_data/...`, which assumes the two repositories are cloned
> side by side. Adjust the paths if you clone them elsewhere.

---

## Installation

Requires Python 3.7+ (3.8 recommended). Installation takes a few minutes on a normal
computer. A GPU is strongly recommended for training.

```bash
# Clone both repositories side by side
git clone https://github.com/GiuliaGiugliano/X-MACE_photo.git
git clone https://github.com/GiuliaGiugliano/X-MACE_photo_data.git

cd X-MACE_photo

# Create and activate an environment
conda create --name x-mace_photo-env python=3.8 -y
conda activate x-mace_photo-env

# Install X-MACE and its dependencies
pip install .
```

The classification and virtual-screening steps need a few extra packages. Install them
with:

```bash
pip install .
```

---

## Workflow

The full pipeline has four stages. Stages 1 and 2 are training; stages 3 and 4 are
prediction on new molecules.

### 1. Oscillator-strength classification (XGBoost)

Before X‑MACE regression on oscillator strengths, an XGBoost classifier discriminates
molecules with oscillator strength > 0.03. X‑MACE regression is then applied only to
the molecules that pass.

```bash
python3 scripts/classification.py \
  --train ../X-MACE_photo_data/training_set_dataset_file/full_system_oscillator_first_excitation_train.xyz \
  --test  ../X-MACE_photo_data/test_set_dataset_file/full_system_oscillator_first_excitation_test.xyz \
  --outdir classifier_assets \
  --device cpu
```

This writes the trained classifier assets (`model.json`, `scaler.pkl`,
`species.json`) plus diagnostic plots into `--outdir`. Use `--device cuda` on a GPU.

> **Note.** The classifier needs both classes (oscillator strength above and below
> 0.03). The published oscillator dataset contains only molecules above the
> threshold, so it cannot be used to retrain the classifier as-is.

### 2. X‑MACE regression training

```bash
python3 scripts/run_train.py \
  --train_file="../X-MACE_photo_data/training_set_dataset_file/full_system_excitation_energy_train.xyz" \
  --name="model" --seed=100 --valid_fraction=0.1 --E0s='average' \
  --model="EmbeddingEMACE" --r_max=5.0 --batch_size=5 --correlation=3 \
  --max_num_epochs=350 --ema --lr=0.001 --ema_decay=0.99 \
  --default_dtype="float32" --device=cuda \
  --hidden_irreps="256x0e + 256x1o" --MLP_irreps='256x0e' \
  --num_radial_basis=8 --num_interactions=2 \
  --energy_weight=100 --kisc_weight=0 --oscillator_weight=0 \
  --wavelen_weight=0 --hlgap_weight=0 \
  --error_table="EnergyNacsDipoleMAE" --scalar_key="REF_scalar" \
  --n_nacs=0 --n_dipoles=0 --n_socs=0 --n_oscillator=0 \
  --n_energies=5 --n_wavelen=0 --n_kisc=0 --n_hlgap=0
```

Pass the training file for the property you want to model. Available property files are
in `../X-MACE_photo_data/training_set_dataset_file/`:

| Property | Training file |
|---|---|
| Excitation energy | `full_system_excitation_energy_train.xyz` |
| HOMO–LUMO gap | `full_system_hlgap_train.xyz` |
| log k<sub>ISC</sub> | `full_system_logkisc_train.xyz` |
| Transition dipole | `full_system_transition_dipole_train.xyz` |
| ΔE(S<sub>0</sub>–T<sub>1</sub>) | `full_system_delta_s0_t1_train.xyz` |
| ΔE(S<sub>0</sub>–anion) | `full_system_delta_s0_an_train.xyz` |
| ΔE(cation–T<sub>1</sub>) | `full_system_delta_cat_t1_train.xyz` |
| Oscillator strength | `full_system_oscillator_first_excitation_train.xyz` |

The hyperparameters used for each property are given in the paper, *"Machine
learning-driven discovery of novel photosensitizer for cancer therapy"*.

### 3. Evaluation on the test set

Models are saved with CUDA tensors, so convert to CPU first:

```bash
python3 scripts/convert_model_to_cpu.py --input model.model --output model_cpu.model
```

Then evaluate and plot:

```bash
python3 scripts/plot_distribution_avarage_test_mae.py \
  --xyz ../X-MACE_photo_data/test_set_dataset_file/full_system_delta_s0_t1_test.xyz \
  --model model_cpu.model \
  --n-energies 5 \
  --outdir results_test
```

This predicts on the test set, plots the reference-vs-prediction scatter, and
reports the MAE. `--n-energies` must match the `--n_energies` used at training time.

The equivalent script for the training set additionally needs the validation index
file and the metrics file written by the training run:

```bash
python3 scripts/plot_distribution_avarage_train_mae.py \
  --xyz ../X-MACE_photo_data/training_set_dataset_file/full_system_delta_s0_t1_train.xyz \
  --model model_cpu.model \
  --valid-indices valid_indices.txt \
  --metrics results/model_train.txt \
  --n-energies 5
```

### 4. Virtual screening

First classify by oscillator strength, then predict the remaining properties.

```bash
# Keep molecules with oscillator strength > 0.03
python3 scripts/xgboost_predict_new.py \
  --input molecule.xyz \
  --assets classifier_assets

# Predict photophysical properties for the retained molecules
python3 scripts/predict.py \
  --xyz ../X-MACE_photo_data/virtual_screening_dataset/virtual_screening_dataset.xyz \
  --model model_cpu.model \
  --n-energies 5 \
  --out predictions.csv
```

`xgboost_predict_new.py` loads the classifier assets produced in stage 1 from
`--assets`. It checks that the SOAP descriptor length matches the trained scaler and
fails with a clear message if the two disagree.

`predict.py` writes one row per structure to `--out`, with the mean predicted energy
for each of the `--n-energies` states.

The selection thresholds reported in the paper (λmax > 400 nm, oscillator strength
> 0.03, HOMO-LUMO gap 1.5-3.5 eV, ΔE(T1-S0) > 0.88 eV, redox gaps > 3.5 eV) are
applied to these outputs; they are not currently implemented as a script.

---

## Datasets

Datasets are in the companion repository,
[`X-MACE_photo_data`](https://github.com/GiuliaGiugliano/X-MACE_photo_data):

- `training_set_dataset_file/` — training sets, one `.xyz` per property. Each molecule
  carries its charge, the reference property in the `REF_energy` array, and the
  ground-state XYZ coordinates.
- `test_set_dataset_file/` — test sets, same format.
- `virtual_screening_dataset/` — molecules from the ATC dataset used for screening,
  with charge and ground-state geometry.

---

## License

This project is licensed under the MIT License.
