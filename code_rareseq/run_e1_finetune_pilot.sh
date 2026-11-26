#!/bin/bash
set -uo pipefail
PY=/root/miniconda3/envs/llr_env/bin/python
export PYTHONPATH=/root/rareseq/pydeps:/root/rnaft/pypath
export HF_HOME=/root/hf_home HF_HUB_OFFLINE=1
W=/root/rareseq/work
E0=/root/rareseq/data/e0
ROOT=/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/e1_rnafm_finetune_strict3train_p20
LOGDIR=/home/ProxyFunctionGap/logs
STATUS=$LOGDIR/e1_finetune_pilot.status
mkdir -p "$ROOT" "$LOGDIR"
cd "$W" || exit 1
COMMON=(train --windows-dir "$E0/windows" --neg-dir "$E0/neg"
  --positives-audit "$E0/positives_audit.tsv" --split-clstr "$E0/splits/clusters.id90.clstr"
  --exps-file "$E0/e1_smoke_exp.txt" --folds 0 --tracks neg1c,neg2 --seed 0
  --epochs 1000 --patience 20 --min-delta 1e-4 --batch 256
  --data-version e0-v2-id90-strict3train-main --log "$LOGDIR/experiments.jsonl")

resource_gate() {
  local MIN_FREE=$1 FREE_MIB AVAIL_KIB ATTEMPT
  for ATTEMPT in $(seq 1 36); do
    FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)
    AVAIL_KIB=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
    if [ "$FREE_MIB" -ge "$MIN_FREE" ] && [ "$AVAIL_KIB" -ge 12582912 ]; then
      echo "RESOURCE_OK free_mib=$FREE_MIB mem_available_kib=$AVAIL_KIB"
      return 0
    fi
    echo "RESOURCE_WAIT attempt=$ATTEMPT free_mib=$FREE_MIB mem_available_kib=$AVAIL_KIB"
    sleep 300
  done
  echo "RESOURCE_BLOCKED" >&2
  return 1
}

run_pilot() {
  local MODE=$1 LR=3e-4 MIN_FREE=9000 DIR="$ROOT/pilot_cuda_$1" LOG="$LOGDIR/e1_finetune_pilot_$1.log"
  if [ "$MODE" = full ]; then LR=1e-5; MIN_FREE=14000; fi
  if [ -f "$DIR/.train_complete" ]; then echo "SKIP pilot $MODE"; return 0; fi
  resource_gate "$MIN_FREE" || return 1
  "$PY" e1_finetune.py "${COMMON[@]}" --mode "$MODE" --encoder-lr "$LR" --lora-r 8 --out "$DIR" >"$LOG" 2>&1
}

echo "RUNNING $(date -Is)" > "$STATUS"
run_pilot lora & P0=$!
run_pilot full & P1=$!
RC=0
wait "$P0" || RC=1
wait "$P1" || RC=1
if [ "$RC" -ne 0 ]; then echo "FAILED $(date -Is)" > "$STATUS"; exit 1; fi
"$PY" e1_finetune_verify.py --runs "$ROOT/pilot_cuda_lora,$ROOT/pilot_cuda_full" \
  --out "$ROOT/pilot_cuda_verification.json" >>"$LOGDIR/e1_finetune_pilot_driver.log" 2>&1 || {
  echo "VERIFY_FAILED $(date -Is)" > "$STATUS"; exit 1; }
echo "PILOT_PASSED $(date -Is)" > "$STATUS"
