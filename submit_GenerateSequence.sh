#!/bin/bash

#SBATCH --job-name=GenerateSequence
#SBATCH -C a100          # Partition pour A100
#SBATCH --account=wka@a100       # Votre compte valide
#SBATCH --qos=qos_gpu_a100-t3
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16           # 16 cœurs CPU
#SBATCH --gres=gpu:1                 # 1 GPU A100
#SBATCH --output=/lustre/fswork/projects/rech/wka/ufl73qn/TransformerProject/logs/GenerateSequence_%j.out  # Chemin absolu
#SBATCH --error=/lustre/fswork/projects/rech/wka/ufl73qn/TransformerProject/logs/GenerateSequence_%j.err    # Chemin absolu

module purge

eval $(idrenv -d wka) && cd $WORK

# Chargement des modules nécessaires (version corrigée pour Jean Zay)

module purge
#module load cuda/12.8.0
#module load python/3.12.7
module load tensorflow-gpu/py3/2.8.0  # TensorFlow avec support GPU


# Déplacement dans le dossier source
cd /lustre/fswork/projects/rech/wka/ufl73qn/TransformerProject/src
pwd

export CUDA_VISIBLE_DEVICES=0

# Exécution du script Python

python GeneratorSequence.py

# #submission command is : sbatch submit_GenerateSequence.sh
