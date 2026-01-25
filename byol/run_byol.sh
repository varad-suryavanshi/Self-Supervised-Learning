#!/bin/bash
#SBATCH --account=csci_ga_2572-2025fa
#SBATCH --partition=c24m170-a100-2
#SBATCH --job-name=byol_r50x2_1.5M
#SBATCH --gres=gpu:2                 
#SBATCH --time=48:00:00              
#SBATCH --output=byol_r50x2_%j.out   



# Activate your venv
cd /scratch/vs3273
source byol_env/bin/activate

# Go to BYOL project folder
cd BYOL_1.5M

# Run BYOL training
python3 train_byol.py \
  --output-dir ./checkpoints_resnet50x2_1.5M \
  --batch-size-per-gpu 512 \
  --epochs 800 \
  --base-lr 0.8 \
  --tau-base 0.996 \
  --grad-accum-steps 1
