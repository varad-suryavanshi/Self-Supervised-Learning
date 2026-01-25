#!/bin/bash
#SBATCH --job-name=swav_96_1M_rn50w2
#SBATCH --account=csci_ga_2572-2025fa
#SBATCH --partition=c24m170-a100-2
#SBATCH --time=24:00:00              # max walltime
#SBATCH --gres=gpu:2
#SBATCH --output=/scratch/vs3273/swav_runs/swav_96_1M_rn50w2/slurm_%j.out
#SBATCH --error=/scratch/vs3273/swav_runs/swav_96_1M_rn50w2/slurm_%j.err

# Safety: create log dir if it doesn't exist
mkdir -p /scratch/vs3273/swav_runs/swav_96_1M_rn50w2


source /scratch/vs3273/miniconda3/etc/profile.d/conda.sh
# Activate conda
conda activate swav_env


which python
which torchrun || echo "torchrun not found in PATH"

# Go to your SwAV code directory
cd /scratch/vs3273/swav

# Single-node, 2-GPU launch
python -m torch.distributed.run --standalone --nproc_per_node=2 main_swav.py \
  --file_list /scratch/vs3273/DL_pretrain_1.5M/filelist_1p05M.txt \
  --dump_path /scratch/vs3273/swav_runs/swav_96_1M_rn50w2 \
  --arch resnet50w2 \
  --epochs 400 \
  --batch_size 512 \
  --base_lr 0.8 \
  --final_lr 0.0008 \
  --warmup_epochs 10 \
  --start_warmup 0.0 \
  --size_crops 96 40 \
  --nmb_crops 2 6 \
  --min_scale_crops 0.30 0.14 \
  --max_scale_crops 1.0 0.30 \
  --crops_for_assign 0 1 \
  --nmb_prototypes 3000 \
  --queue_length 3072 \
  --epoch_queue_starts 15 \
  --use_fp16 true \
  --sync_bn pytorch \
  --wd 1e-6 \
  --workers 8 \
  --freeze_prototypes_niters 5000 \
  --epsilon 0.03 \
  --temperature 0.1 \
  --checkpoint_freq 10 \
  --seed 31
