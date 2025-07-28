#!/bin/bash

#SBATCH --job-name=tokenisation
#SBATCH -C a100          # Partition pour A100
#SBATCH --account=wka@a100       # Votre compte valide
#SBATCH --qos=qos_gpu_a100-t3
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16           # 16 cœurs CPU
#SBATCH --gres=gpu:1                 # 1 GPU A100
#SBATCH --output=/lustre/fswork/projects/rech/wka/ufl73qn/TransformerProject/logs/tokenisation_%j.out  # Chemin absolu
#SBATCH --error=/lustre/fswork/projects/rech/wka/ufl73qn/TransformerProject/logs/tokenisation_%j.err    # Chemin absolu

eval $(idrenv -d wka) && cd $WORK

# Charge Python
module load tensorflow-gpu/py3/2.8.0  # TensorFlow avec support GPU

# Active l'environnement virtuel
source /lustre/fswork/projects/rech/wka/ufl73qn/TransformerProject/mon_env/bin/activate

# Déplacement dans le dossier source
cd /lustre/fswork/projects/rech/wka/ufl73qn/TransformerProject/src
pwd

export CUDA_VISIBLE_DEVICES=0

# Exécution du script Python

python FiltrageDonneesEntree.py
python Tokenisation.py

# #submission command is : sbatch tokenisation.sh
