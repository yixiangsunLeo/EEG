#!/bin/bash
# One GPU job of the cross-subject benchmark. Resources follow the measured cluster limits
# (docs/PLAN_cross_subject_benchmark_and_fusion.md §7.1):
#
#   QOS 32cpu4gpu64g : MaxWall 48h, cpu=32, gres/gpu=4 ENFORCED, mem=64G NOT enforced
#   -> per job: 1 GPU, 8 CPUs, 40G. Four such jobs saturate the per-user GPU limit.
#   -> --mem=40G is measured (MaxRSS 15.6G on the 821 run). Cutting to 64/4=16G OOMs.
#
# `-t 47:30:00` leaves 30 min of headroom under MaxWall so the job is killed by our own
# limit rather than by Slurm, which keeps the exit status readable.
#
# Usage (normally via scripts/slurm/submit_bench.sh, not by hand):
#   sbatch -J bench_g1 scripts/slurm/bench_gpu.sbatch <config.yaml> [extra args...]
#
#SBATCH -o logs/slurm/%x-%j.out
#SBATCH -e logs/slurm/%x-%j.err
#SBATCH -p gpu2node
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH -t 47:30:00
#SBATCH --mail-type=FAIL

set -uo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/scripts/slurm/_common.sh" ]; then
  # shellcheck disable=SC1091
  source "${SLURM_SUBMIT_DIR}/scripts/slurm/_common.sh"
else
  # shellcheck disable=SC1091
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
fi

CONFIG="${1:?config path required (code/configs/experiments/*.yaml)}"
shift || true

cd "$PROJECT_ROOT"
mkdir -p logs/slurm

echo "=== job ${SLURM_JOB_NAME:-?} / ${SLURM_JOB_ID:-?} ==="
echo "Project : $PROJECT_ROOT"
echo "Env     : $MI_ENV"
echo "Node    : $(hostname)"
echo "Config  : $CONFIG"
echo "Extra   : $*"
echo "Start   : $(date -Is)"
echo "Git     : $(git rev-parse --short HEAD 2>/dev/null) $(git status --porcelain 2>/dev/null | wc -l) dirty file(s)"

mi_activate_conda
nvidia-smi || true
mi_cuda_failfast

# `--device cuda` fails fast when CUDA is missing (never silently fall back to CPU).
python -u code/run.py --config "$CONFIG" --device cuda "$@"
RC=$?

echo "End     : $(date -Is)"
echo "Exit    : $RC"

# Resource snapshot for the next round's budget. sacct data for a still-running job is
# incomplete, so this is best effort; submit_bench.sh runs a complete pass at the end.
if command -v sacct >/dev/null 2>&1 && [ -n "${SLURM_JOB_ID:-}" ]; then
  sacct -j "$SLURM_JOB_ID" --units=G \
    --format=JobID%20,JobName%18,State%13,Elapsed%11,Timelimit%11,ReqMem%9,MaxRSS%9,AllocTRES%40 \
    | tee "logs/slurm/sacct_${SLURM_JOB_NAME:-job}-${SLURM_JOB_ID}.txt"
fi

exit $RC
