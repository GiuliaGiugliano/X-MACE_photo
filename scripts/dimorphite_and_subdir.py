import os
import subprocess
from rdkit import Chem
from rdkit.Chem import AllChem

# Funzione per calcolare carica formale
def get_formal_charge(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.GetFormalCharge(mol)

# Funzione per generare file XYZ da SMILES con stereochimica e idrogeni
def smiles_to_xyz(smiles, charge, out_file):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    
    mol = Chem.AddHs(mol)  
    AllChem.EmbedMolecule(mol, randomSeed=42)  # preserva stereochimica
    AllChem.UFFOptimizeMolecule(mol)

    conf = mol.GetConformer()
    with open(out_file, 'w') as f:
        f.write(f"{mol.GetNumAtoms()}\n")
        f.write(f"Generated from SMILES, charge={charge}\n")
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            f.write(f"{atom.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}\n")
    return True


# MAIN SCRIPT
for i in range(312): 
    folder = f"mol_{i}"
    print(f"🔍 Processing {folder}...")

    if not os.path.exists(folder):
        print(f"Folder {folder} not found")
        continue

    smi_file = os.path.join(folder, f"mol_{i}.smi")
    if not os.path.exists(smi_file):
        print(f"File {smi_file} not found")
        continue

    # Read SMILES
    with open(smi_file, "r") as f:
        smiles = f.readline().strip()

    #Dimorphite
    dimo_file = "smile_dimo.out"  # solo nome file perché cwd=folder
    dimorphite_cmd = (
        f'dimorphite_dl --ph_min 7.0 --ph_max 8.0 --precision 0 '
        f'--label_states --max_variants 4 '
        f'"{smiles}" --output "{dimo_file}"'
    )

    try:
        subprocess.run(dimorphite_cmd, cwd=folder, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error in {folder}: {e}")
        continue

    # collect all the smiles
    dimo_output_path = os.path.join(folder, dimo_file)
    total_smiles_file = os.path.join(folder, "smiles_tot.out")

    with open(dimo_output_path) as df, open(total_smiles_file, "w") as tf:
        variants = [line.strip() for line in df if line.strip()]
        tf.write(smiles + "\n")  # SMILES originale
        for v in variants:
            tf.write(v + "\n")

    # Remove doubles
    dedup_file = os.path.join(folder, "smiles_tot_dedup.out")
    unique_smiles = list(dict.fromkeys([smiles] + variants))

    with open(dedup_file, "w") as f:
        for s in unique_smiles:
            f.write(s + "\n")

    # Create the subdir for esach protomer
    for idx, s in enumerate(unique_smiles, start=1):
        subfolder = os.path.join(folder, f"mol_{i}_protomer_{idx}")
        os.makedirs(subfolder, exist_ok=True)

        # Salva SMILES
        sub_smi_file = os.path.join(subfolder, f"mol_{i}_protomer_{idx}.smi")
        with open(sub_smi_file, "w") as sf:
            sf.write(s)

        # 6️⃣ Calcola carica formale
        charge = get_formal_charge(s)
        charge_file = os.path.join(subfolder, "tot_charge.txt")
        with open(charge_file, "w") as cf:
            cf.write(str(charge))

        # 7️⃣ Genera file XYZ
        xyz_file = os.path.join(subfolder, f"mol_{i}_protomer_{idx}.xyz")

        try:
            if smiles_to_xyz(s, charge, xyz_file):
                print(f"   ✔ Protomer {idx} completato")
            else:
                print(f"Error for  protomer {idx}")
        except Exception as e:
            print(f"RDKIT error for  protomer {idx}: {e}")

    print(f"Completed {folder}\n")

print("\nFinished")

