X-MACE represents an advanced deep learning framework specifically designed to accurately model excited-state potential energy surfaces, with a particular emphasis on regions near conical intersections. This framework builds upon the Message Passing Atomic Cluster Expansion (MACE) architecture by incorporating Deep Sets, allowing for the generation of smooth representations of inherently non-smooth energy landscapes. In this study, we introduce an adapted version of X-MACE aimed at the identification of novel potential photosensitizers for applications in photodynamic therapy. The model demonstrates the capability to predict UV-VIS spectra, intersystem crossing rates, HOMO-LUMO gaps, the energy difference between the ground and triplet states, which is critical for the type II mechanism, the energy difference between the ground state and the radical anion form of the molecule, as well as the energy differential between the triplet state and the cation form of the photosensitizer pertinent to the type I mechanism. This is achieved by inputting the ground state xyz structure along with the total molecular charge.

# Installation

Ensure that Python 3.7+ is installed in your environment. To install X‑MACE and its dependencies clone the github repo and install locally. The installation should only take a few minutes on a normal computer. The following commands illustate this:

git clone https://github.com/GiuliaGiugliano/X-MACE_photo.git

cd x-mace

pip install .

The installation can be made also on a python environment

# Clone the repository

git clone https://github.com/GiuliaGiugliano/X-MACE_photo.git

cd x-mace

# Create and activate a new Python virtual environment using conda

conda create --name x-mace_photo-env python=3.8 -y

conda activate x-mace-env

# Install dependencies and X‑MACE

pip install .

# Usage for classification

python3 classification.py

For oscillator strength values training before X-MACE regression, an XGBoost classification model is used to discriminate molecules with an oscillator strength > 0.03. On the resulted molecules X-MACE regression is applied

# Usage for training 

python3 X-MACE_photo/scripts/run_train.py --train_file="full_system_excitation_energy_train.xyz" --name="model" --seed=100 --valid_fraction=0.1 --E0s='average' --model="EmbeddingEMACE" --r_max=5.0 --batch_size=5 --correlation=3 --max_num_epochs=350 --ema --lr=0.001 --ema_decay=0.99 --default_dtype="float32" --device=cuda --hidden_irreps="256x0e + 256x1o" --MLP_irreps='256x0e' --num_radial_basis=8 --num_interactions=2 --energy_weight=100 --kisc_weight=0 --oscillator_weight=0 --wavelen_weight=0 --hlgap_weight=0 --error_table="EnergyNacsDipoleMAE"  --scalar_key="REF_scalar" --n_nacs=0 --n_dipoles=0 --n_socs=0 --n_oscillator=0 --n_energies=5 --n_wavelen=0 --n_kisc=0 --n_hlgap=0


--train_file is followed by the the corrisponding property file in the ase xyz format. For each property the hyperparamers used are in the paper: "Machine learning-driven discovery of novel photosensitizer for cancer therapy"

# Usage for test

Before having the predictions convert the model in a cpu.model file

python3 convert_model_to_cpu.py

python3 plot_distribution_avarage_test_mae.py

The script plot_distribution_avarage_test_mae.py makes the prediction on the test set dataset and plots the scatter plot "Reference vs Prediction" and the MAE computed on the test set

# Usage for virtual screening classification (oscillator strength)

python3 xgboost_predict_new.py

The scripts takes the trained classification model and classifies molecules with oscillator strengths lower and higher than 0.03

# Usage for virtual screening X-MACE predictions

python3 predict.py 

The script predict.py makes the predictions of a certain photophysical property by putting the virtual screening dataset in the ase xyz format. It has to contains the xyz geometry of the ground state and the total charge of the molecule. I t prints the results in a csv file

# Dataset

All the training, test, virtual screening datasets and scripts are available via the following link:
https://github.com/GiuliaGiugliano/X-MACE_photo_data.git


