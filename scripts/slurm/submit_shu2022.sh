#!/bin/bash
# Submit both SHU 2022 runs across THREE GPUs (advisor, 2026-08-16).
#
#   experiment 1  cross-session   ses-01+02+03 -> ses-04+05, one model per subject
#                 code/configs/experiments/shu_cross_session_3to2.yaml
#   experiment 2  cross-subject   5-fold over subjects, 16:4:5, split seeds 41-45
#                 code/configs/experiments/shu_cross_subject_5fold.yaml
#
# Concurrency. The 7 models are split into 3 groups, balanced with the per-fold times
# measured on the 821 run (atcnet 2.06 h > s4erp 0.74 > eeg_deformer 0.55 > eegnex 0.46 >
# eegnet_official 0.22; deepconvnet / fbcnet not measured yet). Each group gets one GPU.
# The experiment-2 job of a group waits for that group's experiment-1 job
# (`--dependency=afterany`), so exactly 3 GPUs are busy at any time.
#
# Resume. Re-running this script is safe and is the documented way to continue after a
# 48 h timeout: finished cells are skipped via their result.json, an interrupted cell
# continues from last.pt, and the cell signature refuses to resume onto a different
# configuration.
#
# Usage:
#   bash scripts/slurm/submit_shu2022.sh            # submit both experiments
#   bash scripts/slurm/submit_shu2022.sh cs         # experiment 1 only
#   bash scripts/slurm/submit_shu2022.sh xs         # experiment 2 only (no dependency)
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1

CFG_CS="code/configs/experiments/shu_cross_session_3to2.yaml"
CFG_XS="code/configs/experiments/shu_cross_subject_5fold.yaml"
RUN_CS="outputs/experiments/shu/cross_session_3to2_v1"
RUN_XS="outputs/experiments/shu/cross_subject_5fold_v1"
SBATCH="scripts/slurm/shu_gpu.sbatch"

# QOS caps wall time at 48 h on this cluster (probed 2026-08-12: 60/72/96 h rejected).
TIME="48:00:00"
# `mem` is not charged against the QOS, and the 821 run OOM'd when it was cut. Keep 40G.
MEM="40G"

# One GPU per group; balanced by the 821 per-fold timings.
MODEL_GROUPS=("atcnet,eegnet_official" "eeg_deformer,s4erp" "eegnex,deepconvnet,fbcnet")

WHAT="${1:-both}"
mkdir -p "$RUN_CS" "$RUN_XS" logs/slurm

CS_IDS=()
XS_IDS=()

if [ "$WHAT" = "both" ] || [ "$WHAT" = "cs" ]; then
  for i in "${!MODEL_GROUPS[@]}"; do
    models="${MODEL_GROUPS[$i]}"
    jid=$(sbatch --parsable -J "shucs_g$((i + 1))" -t "$TIME" --mem="$MEM" \
          "$SBATCH" "$CFG_CS" --models "$models" --tag-suffix "g$((i + 1))")
    CS_IDS+=("$jid")
    echo "cross-session  job $jid  gpu$((i + 1))  models=$models"
  done
fi

if [ "$WHAT" = "both" ] || [ "$WHAT" = "xs" ]; then
  for i in "${!MODEL_GROUPS[@]}"; do
    models="${MODEL_GROUPS[$i]}"
    dep=()
    # Chain onto this group's cross-session job so only 3 GPUs are ever in use.
    if [ "${#CS_IDS[@]}" -gt "$i" ]; then
      dep=(--dependency="afterany:${CS_IDS[$i]}")
    fi
    jid=$(sbatch --parsable -J "shuxs_g$((i + 1))" -t "$TIME" --mem="$MEM" "${dep[@]}" \
          "$SBATCH" "$CFG_XS" --models "$models" --tag-suffix "g$((i + 1))")
    XS_IDS+=("$jid")
    echo "cross-subject  job $jid  gpu$((i + 1))  models=$models  ${dep[*]:-（no dependency）}"
  done
fi

{
  echo "submitted $(date -Iseconds)"
  echo "gpus 3 (one per model group; experiment 2 chained after experiment 1)"
  echo "group1 ${MODEL_GROUPS[0]}"
  echo "group2 ${MODEL_GROUPS[1]}"
  echo "group3 ${MODEL_GROUPS[2]}"
  echo
  echo "experiment 1 = cross-session, config $CFG_CS"
  echo "  split   train ses-01+02+03 -> test ses-04+05, same subject, 25 subjects"
  echo "  recipe  SHU 2022 paper: Adam 1e-3, batch 16, NLLLoss; 300 epochs (paper: 1500)"
  echo "  early   monitor accuracy, patience 25; val = 1/9 of the training pool (8:1)"
  echo "  norm    none (paper feeds the preprocessed data directly)"
  echo "  jobs    ${CS_IDS[*]:-none}"
  echo
  echo "experiment 2 = cross-subject, config $CFG_XS"
  echo "  split   5-fold over 25 subjects, 16 train / 4 val / 5 test subjects"
  echo "  seeds   split_seeds 41-45 (partition), training seed fixed at 0"
  echo "  recipe  Adam 1e-4, batch 128, 500 epochs; monitor accuracy, patience 25"
  echo "  norm    per_sample_zscore"
  echo "  jobs    ${XS_IDS[*]:-none}"
  echo
  echo "both     curves=true (train/val/test per epoch; test never used for selection)"
  echo "         checkpoints best+last only, resume on re-submit"
} | tee "$RUN_CS/parallel_job_ids.txt" > "$RUN_XS/parallel_job_ids.txt"

squeue -u "$USER" -o '%.10i %.9P %.20j %.2t %.10M %.12l %R'
