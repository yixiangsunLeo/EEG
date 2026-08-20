#!/bin/bash
#SBATCH --job-name=s4erp_adv
#SBATCH --partition=gpu3node
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/s4erp_adv_%j.out
#SBATCH --error=logs/s4erp_adv_%j.err

cd /share/home/yuan/SYX/eeg-mi-online
source activate ly_torch

python code/run.py \
    --config code/configs/experiments/foundation_cross_subject.yaml \
    --models s4erp_adv \
    --device cuda
