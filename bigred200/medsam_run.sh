#!/bin/bash
#SBATCH --job-name=medsam_run
#SBATCH --output=medsam_%j.out
#SBATCH --error=medsam_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -A <project_name>  # <-- replace with your project name

mkdir -p logs

module load python/3.11           # <-- replace with what you actually have
source myenv/bin/activate

python train_medsam.py \
  --train_csv ./medsam_meniscus_2d/split/train.csv \
  --val_csv ./medsam_meniscus_2d/split/val_clean.csv \
  --checkpoint ./checkpoints/medsam_vit_b.pth \
  --epochs 30 \
  --batch_size 1 \
  --num_workers 4 \
  --outdir ./outputs_run2