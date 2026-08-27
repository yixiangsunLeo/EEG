#!/bin/bash
# SHU 2022 experiment 2 DualCD add-on: Flatten + Transformer only.
#
# Same protocol as the finished 7-model run (16:4:5, split_seeds 41-45, Adam 1e-4,
# effective batch 128, patience 25, monitor accuracy, curves, resume). Cells land in
# the existing v1 output/checkpoint dirs; the 175 supervised cells are skipped.
#
# Smoke first (2 epochs, 1 fold, isolated dirs) so a DualCD OOM cannot poison v1.
# Full jobs wait on the smoke via afterok.
#
# Usage:
#   bash scripts/slurm/submit_shu_cross_subject_dualcd.sh          # smoke + full
#   bash scripts/slurm/submit_shu_cross_subject_dualcd.sh smoke    # smoke only
#   bash scripts/slurm/submit_shu_cross_subject_dualcd.sh full     # full only (no smoke dep)
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1

CFG="code/configs/experiments/shu_cross_subject_5fold.yaml"
RUN="outputs/experiments/shu/cross_subject_5fold_v1"
SMOKE_OUT="outputs/experiments/shu/cross_subject_5fold_v1_dualcd_smoke"
SMOKE_CKPT="checkpoints/shu/cross_subject_5fold_v1_dualcd_smoke"
SBATCH="scripts/slurm/shu_gpu.sbatch"

TIME="48:00:00"
SMOKE_TIME="2:00:00"
MEM="40G"

WHAT="${1:-both}"
mkdir -p "$RUN" "$SMOKE_OUT" logs/slurm

SMOKE_ID=""
if [ "$WHAT" = "both" ] || [ "$WHAT" = "smoke" ]; then
  SMOKE_ID=$(sbatch --parsable -J "shuxs_dino_smoke" -t "$SMOKE_TIME" --mem="$MEM" \
    "$SBATCH" "$CFG" \
    --models dualcd_s4_flatten,dualcd_transformer \
    --split-seeds 41 --folds-subset 0 --max-epochs 2 \
    --out "$SMOKE_OUT" --ckpt-dir "$SMOKE_CKPT" \
    --tag-suffix dino_smoke)
  echo "smoke  job $SMOKE_ID  2 epochs, split41 fold0, isolated dirs"
fi

FLAT_ID=""
TF_ID=""
if [ "$WHAT" = "both" ] || [ "$WHAT" = "full" ]; then
  dep=()
  if [ -n "$SMOKE_ID" ]; then
    dep=(--dependency="afterok:${SMOKE_ID}")
  fi
  FLAT_ID=$(sbatch --parsable -J "shuxs_dino_flat" -t "$TIME" --mem="$MEM" "${dep[@]}" \
    "$SBATCH" "$CFG" --models dualcd_s4_flatten --tag-suffix dino_flat)
  echo "full   job $FLAT_ID  dualcd_s4_flatten  ${dep[*]:-(no dependency)}"

  TF_ID=$(sbatch --parsable -J "shuxs_dino_tf" -t "$TIME" --mem="$MEM" "${dep[@]}" \
    "$SBATCH" "$CFG" --models dualcd_transformer --tag-suffix dino_tf)
  echo "full   job $TF_ID  dualcd_transformer  ${dep[*]:-(no dependency)}"
fi

{
  echo "submitted $(date -Iseconds)"
  echo "SHU 2022 exp2 DualCD add-on: Flatten + Transformer"
  echo "config  $CFG"
  echo "recipe  identical to the 7-model run (Adam 1e-4, effective batch 128,"
  echo "        patience 25, monitor accuracy, curves, resume)"
  echo "micro   flatten 32 / transformer 16  (same as 821)"
  echo "out     $RUN   (supervised 175 cells skipped)"
  echo "smoke   $SMOKE_ID  out=$SMOKE_OUT"
  echo "full    flatten=$FLAT_ID  transformer=$TF_ID"
} | tee -a "$RUN/parallel_job_ids.txt"

squeue -u "$USER" -o '%.10i %.9P %.24j %.2t %.10M %.12l %R'
