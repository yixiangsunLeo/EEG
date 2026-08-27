#!/bin/bash
# Submit the cross-subject benchmark: 17 model entries split over 4 GPUs, each GPU's work
# chained across several jobs so it can exceed the 48 h wall clock.
#
#   bash scripts/slurm/submit_bench.sh wbci_2c        # WBCIC-SHU 2C (51 subjects)
#   bash scripts/slurm/submit_bench.sh shu            # SHU 2022 (25 subjects)
#   bash scripts/slurm/submit_bench.sh wbci_2c 6      # 6 relay links instead of 4
#   DRY_RUN=1 bash scripts/slurm/submit_bench.sh shu  # print the plan, submit nothing
#
# WHY RELAY CHAINS
#   `dualcd_transformer` needs ~10 h per cell x 10 cells = ~100 h on WBCIC-2C, but the QOS
#   caps a job at 48 h. Each group is therefore submitted N times with
#   `--dependency=afterany:<previous>`; every link runs the SAME command, and the resume
#   logic (finished cells have a `result.json`, an interrupted cell continues from `last.pt`)
#   makes link k+1 pick up where link k was cut off.
#
#   Over-provisioning links is nearly free: a link whose work is already done skips every
#   cell and exits in seconds. Under-provisioning silently leaves cells unfinished. So the
#   default is generous.
#
# GROUPING
#   Balanced by measured per-cell cost (§7 of the plan). The three DINO/DualCD variants cost
#   ~73% of the total, so they get a GPU each; everything cheap shares the fourth.
#
# WHY NOT ONE JOB WITH 4 GPUs
#   `run_cross_subject` is single-process. Four independent single-GPU jobs is what actually
#   uses the 4-GPU allowance, and it is also what the 821 run did.

set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

DATASET="${1:?usage: submit_bench.sh <wbci_2c|shu> [n_links]}"
N_LINKS="${2:-4}"
DRY_RUN="${DRY_RUN:-0}"

case "$DATASET" in
  wbci_2c) CONFIG="code/configs/experiments/wbci_2c_cross_subject_bench.yaml"; PREFIX="b2c" ;;
  shu)     CONFIG="code/configs/experiments/shu_cross_subject_bench.yaml";     PREFIX="bshu" ;;
  *) echo "ERROR: unknown dataset '$DATASET' (use wbci_2c | shu)" >&2; exit 2 ;;
esac
[ -f "$CONFIG" ] || { echo "ERROR: missing $CONFIG" >&2; exit 2; }

# group name -> comma-separated model entries. Estimated WBCIC-2C hours in the comment.
GROUP_NAMES=(g1 g2 g3 g4)
GROUP_MODELS=(
  "dualcd_transformer"                                                              # ~100 h
  "dualcd_s4_flatten"                                                               #  ~77 h
  "dualcd_s4_timepatch,atcnet"                                                      #  ~90 h
  "eegnet_official,eegnex,eeg_deformer,deepconvnet,fbcnet,s4erp,s4erp_dw@f64d2,s4erp_dw@f16d8,s4erp_dw@f16d2p,atcnet_s4,atcnet@tcn8,atcnet@nwin1,atcnet@noattn"  # ~120 h
)

mkdir -p logs/slurm outputs/experiments
IDS_FILE="logs/slurm/bench_${DATASET}_job_ids.txt"
: > "$IDS_FILE"

echo "config   : $CONFIG"
echo "groups   : ${#GROUP_NAMES[@]} (one GPU each; QOS allows 4 concurrent)"
echo "links    : $N_LINKS per group (48 h wall each -> up to $((N_LINKS * 47)) h per group)"
echo "job ids  : $IDS_FILE"
echo

ALL_IDS=()
for i in "${!GROUP_NAMES[@]}"; do
  g="${GROUP_NAMES[$i]}"
  models="${GROUP_MODELS[$i]}"
  prev=""
  for link in $(seq 1 "$N_LINKS"); do
    name="${PREFIX}_${g}_${link}"
    dep=()
    [ -n "$prev" ] && dep=(--dependency="afterany:${prev}")
    cmd=(sbatch -J "$name" "${dep[@]}" scripts/slurm/bench_gpu.sbatch
         "$CONFIG" --models "$models" --tag-suffix "$g")
    if [ "$DRY_RUN" = "1" ]; then
      echo "[dry-run] ${cmd[*]}"
      prev="DRYID_${name}"
      continue
    fi
    out="$(  "${cmd[@]}" )"
    id="${out##* }"
    echo "$name -> $id ${prev:+(after $prev)}"
    echo "$name $id" >> "$IDS_FILE"
    ALL_IDS+=("$id")
    prev="$id"
  done
done

if [ "$DRY_RUN" = "1" ]; then
  echo
  echo "[dry-run] would also submit the summarize job after all links."
  exit 0
fi

# Summarize once everything has terminated (afterany: also fires if a link failed, so the
# tables reflect whatever completed and `failures.json` says what did not).
DEP="$(IFS=:; echo "${ALL_IDS[*]}")"
SUM_ID="$(sbatch -J "${PREFIX}_sum" --dependency="afterany:${DEP}" \
            -o logs/slurm/%x-%j.out -e logs/slurm/%x-%j.err \
            -p gpu2node -c 4 --mem=16G -t 02:00:00 \
            --wrap "cd $PWD && source scripts/slurm/_common.sh && mi_activate_conda && \
                    python -u scripts/summarize_cross_subject.py --run \$(python -c \
                    \"import yaml,sys;print(yaml.safe_load(open('$CONFIG'))['output']['output_dir'])\") && \
                    sacct -j ${DEP} --units=G \
                      --format=JobID%20,JobName%18,State%13,Elapsed%11,ReqMem%9,MaxRSS%9 \
                      > logs/slurm/sacct_bench_${DATASET}.txt" )"
SUM_ID="${SUM_ID##* }"
echo "${PREFIX}_sum $SUM_ID" >> "$IDS_FILE"
echo
echo "summarize -> $SUM_ID (after all ${#ALL_IDS[@]} links)"
echo
echo "monitor:"
echo "  squeue -u \$USER -o '%.10i %.14j %.2t %.11M %.11L %R'"
echo "  tail -f logs/slurm/${PREFIX}_g4_1-*.out"
echo "if gpu2node is full and a job is PENDING (Resources):"
echo "  scontrol update jobid=<id> partition=gpu3node"
