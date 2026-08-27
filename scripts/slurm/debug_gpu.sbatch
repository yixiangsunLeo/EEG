#!/bin/bash
#SBATCH -o logs/slurm/debug_gpu_%j.out
#SBATCH -e logs/slurm/debug_gpu_%j.err
#SBATCH -p gpu3node
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -t 0:10:00

source /share/software/anaconda3/2024.10/etc/profile.d/conda.sh
conda activate mi_torch_cu118
cd /share/home/yuan/SYX/eeg-mi-online

export CUDA_LAUNCH_BLOCKING=1
python -u -c "
import torch, sys
sys.path.insert(0, '.')
from code.models.eeg_foundation.adapter import build_eeg_foundation

model = build_eeg_foundation('s4erp_adv', n_channels=58, n_times=1000, n_classes=2, sfreq=250,
                              params={'n_subjects': 11, 'lambda_adv': 0.1})
model = model.cuda()

x = torch.randn(8, 58, 1000).cuda()
y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]).cuda()
sids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]).cuda()

print('uses_custom_loss:', model.uses_custom_loss)
print('Forward...')
out = model(x)
print('logits:', out['logits'].shape)
print('subject_logits:', out['subject_logits'].shape)
print('Training step...')
loss, parts = model.training_step(x, y, epoch=0, subject_ids=sids)
print('loss=%.4f parts=%s' % (loss.item(), parts))
print('Backward...')
loss.backward()
print('ALL OK on GPU')
"
