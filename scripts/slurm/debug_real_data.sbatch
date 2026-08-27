#!/bin/bash
#SBATCH -o logs/slurm/debug_real_%j.out
#SBATCH -e logs/slurm/debug_real_%j.err
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
import torch, sys, numpy as np
sys.path.insert(0, '.')
from code.models.eeg_foundation.adapter import build_eeg_foundation, normalize_trials
from code.datasets.session_splits import load_ok_sessions
from code.utils.paths import load_paths, resolve_manifest_path
from torch.utils.data import TensorDataset, DataLoader
import yaml

cfg = yaml.safe_load(open('code/configs/experiments/foundation_cross_subject.yaml'))
P = load_paths('.', require_raw=False)
manifest = resolve_manifest_path(cfg, P)
records = load_ok_sessions(manifest, status_filter=('ok',))

# Load sub-001 real data
from code.utils.io import load_session_npz
rec = [r for r in records if r.subject == 'sub-001'][0]
d = load_session_npz(rec.npz_path)
X = torch.from_numpy(np.ascontiguousarray(d['X'], dtype=np.float32))
X = normalize_trials(X, 'per_sample_zscore')
y = torch.from_numpy(d['y'].astype(np.int64))

# Check for NaN/Inf
print(f'X: {X.shape} NaN={torch.isnan(X).any()} Inf={torch.isinf(X).any()}')
print(f'y: {y.shape} unique={y.unique().tolist()}')

# Build model
model = build_eeg_foundation('s4erp_adv', n_channels=58, n_times=1000, n_classes=2, sfreq=250,
                              params={'n_subjects': 8, 'lambda_adv': 0.1})
model = model.cuda()

# Test with real data
x_gpu = X[:8].cuda()
y_gpu = y[:8].cuda()
sids = torch.zeros(8, dtype=torch.long).cuda()

print('Forward with real data...')
out = model(x_gpu)
print('logits:', out['logits'].shape, 'NaN=', torch.isnan(out['logits']).any())
print('subject_logits:', out['subject_logits'].shape, 'NaN=', torch.isnan(out['subject_logits']).any())

loss, parts = model.training_step(x_gpu, y_gpu, epoch=0, subject_ids=sids)
print('loss=%.4f parts=%s' % (loss.item(), parts))
loss.backward()
print('Backward OK')

# Test multiple batches
print('\n10 batches...')
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
for i in range(10):
    opt.zero_grad()
    xb = X[i*8:(i+1)*8].cuda()
    yb = y[i*8:(i+1)*8].cuda()
    sid = torch.randint(0, 8, (8,)).cuda()
    loss, parts = model.training_step(xb, yb, epoch=0, subject_ids=sid)
    loss.backward()
    opt.step()
    print(f'  batch {i}: loss={loss.item():.4f} ce={parts[\"ce\"]:.4f} adv={parts[\"adv\"]:.4f}')

print('ALL OK with real data on GPU')
"
