#!/bin/bash
#SBATCH -t 48:00:00          # tempo massimo
#SBATCH -p barnard           # partizione
#SBATCH -N 1                 # 1 nodo
#SBATCH --ntasks=1           # un solo task (crest non usa MPI)
#SBATCH --cpus-per-task=16   # 16 core reali per il task
#SBATCH --mem=32G            # RAM totale disponibile al job

# ── ambiente ───────────────────────────────────────────────────────────────
module purge
module load release/24.10
module load GCC/13.2.0 OpenMPI/4.1.6
module load Anaconda3
source "$EBROOTANACONDA3/etc/profile.d/conda.sh"
conda activate /home/gigi895h/envs/crest-env

# ── variabili OpenMP/BLAS coerenti con i 16 core ───────────────────────────
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK      # 16
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ── run ─────────────────────────────────────────────────────────────────────
crest mol_0_protomer_1.xyz \
      --gfn2 --gbsa h2o \
      -T "$OMP_NUM_THREADS" --noreftopo

crest mol_0_protomer_1.xyz \
      --cregen crest_conformers.xyz --notopo \
      -T "$OMP_NUM_THREADS" --chrg 0
