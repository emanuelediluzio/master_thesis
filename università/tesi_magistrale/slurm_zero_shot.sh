#!/bin/bash
#SBATCH --job-name=zero_shot_gemma
#SBATCH --account=tesi_ediluzio
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/work/tesi_ediluzio/logs/slurm/zero_shot_%j.out
#SBATCH --error=/work/tesi_ediluzio/logs/slurm/zero_shot_%j.err

# Carica moduli necessari
module load cuda/11.8
module load python/3.9

# Attiva ambiente conda
source /work/tesi_ediluzio/miniconda3/etc/profile.d/conda.sh
conda activate svg_env

# Vai nella directory di lavoro
cd /work/tesi_ediluzio

# Esegui l'inferenza Zero-shot
python zero_shot_inference.py

echo "Inferenza Zero-shot completata"
