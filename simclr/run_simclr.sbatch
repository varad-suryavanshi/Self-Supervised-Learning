#!/bin/bash
#SBATCH --job-name=simclr_dl_96_a100_2
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/simclr_dl_96_a100_2_%j.out
#SBATCH --error=logs/simclr_dl_96_a100_2_%j.err


module purge
module load anaconda3
source ~/.bashrc
conda activate simclr_env

cd /scratch/vs3273/simclr_custom
mkdir -p logs

# Build filelist once if missing
if [ ! -f /vast/vs3273/DL_pretrain_1.5M/filelist_simclr_96px.txt ]; then
  echo "Filelist not found, building it..."
  python build_filelist.py
fi

echo "Starting SimCLR training on 2x A100..."

python run_dl_pretrain.py \
  --epochs 400 \
  -b 1024 \
  --effective-batch-size 4096 \
  --fp16-precision \
  --log-every-n-steps 200
  --ckpt-freq 5
