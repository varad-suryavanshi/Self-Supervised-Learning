#!/bin/bash
#SBATCH --job-name=swav_preflight
#SBATCH --account=csci_ga_2572-2025fa
#SBATCH --partition=c24m170-a100-2   # or another CPU-capable partition
#SBATCH --gres=gpu:2
#SBATCH --time=0:30:00
#SBATCH --out=./out_check_swav_preflight.out
#SBATCH --err=./err_check_swav_preflight.err

conda activate swav_env

python - << 'EOF'
from src.multicropdataset import MultiCropDataset
from main_swav import args, parser   # or manually construct args in a small script

data_path = "/scratch/vs3273/swav_data"
size_crops = [96, 40]
nmb_crops = [2, 6]
min_scale_crops = [0.14, 0.05]
max_scale_crops = [1.0, 0.14]

ds = MultiCropDataset(
    data_path,
    size_crops,
    nmb_crops,
    min_scale_crops,
    max_scale_crops,
)

print("Dataset size:", len(ds))
# Try a couple of samples to force transforms
x = ds[0]
print("Got first sample with", len(x), "crops.")
EOF
