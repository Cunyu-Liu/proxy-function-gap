#!/bin/bash
set -euo pipefail
PY=/root/miniconda3/envs/llr_env/bin/python
export PYTHONPATH=/root/rareseq/pydeps
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/root/rareseq/work
E0=/root/rareseq/data/e0
ROOT=/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/e5
RUN_ROOT=$ROOT/holdout_convergence
LOG_DIR=/home/ProxyFunctionGap/logs
LOG=$LOG_DIR/experiments.jsonl
STATUS=$LOG_DIR/e5_holdout_convergence.status
HOLDOUTS=HNRNPC,PTBP1,U2AF2,SRSF1,RBFOX2,QKI,IGF2BP1,LIN28B,TAF15,UPF1,CSTF2T,YBX3
mkdir -p "$RUN_ROOT" "$LOG_DIR"
cd "$W"
echo "RUNNING:$(date -Is)" > "$STATUS"
trap 'code=$?; echo "FAILED:${code}:$(date -Is)" > "$STATUS"; exit "$code"' ERR
COMMON=(--windows-dir "$E0/windows" --neg-dir "$E0/neg" --emb-dir "$E0/emb"
  --positives-audit "$E0/positives_audit.tsv" --split-clstr "$E0/splits/clusters.id90.clstr"
  --exps-file "$E0/main_exps.txt" --holdouts "$HOLDOUTS" --folds 0
  --max-epochs 1000 --patience 20 --min-delta 1e-4 --log "$LOG")
run_track() {
  local seed=$1
  local track=$2
  local out=$RUN_ROOT/holdout_convergence_s${seed}/${track}
  local log=$LOG_DIR/e5_holdout_convergence_s${seed}_${track}.log
  local extra=()
  if [[ -f "$out/config.json" ]]; then extra=(--resume); fi
  printf '\n===== %s seed=%s track=%s =====\n' "$(date -Is)" "$seed" "$track" >> "$log"
  "$PY" e5_rbp_holdout.py "${COMMON[@]}" --tracks "$track" --seed "$seed" --out "$out" "${extra[@]}" >> "$log" 2>&1
}
for S in 0 1 2; do
  run_track "$S" neg1c & P1=$!
  run_track "$S" neg2 & P2=$!
  set +e
  wait "$P1"; R1=$?
  wait "$P2"; R2=$?
  set -e
  if [[ $R1 -ne 0 || $R2 -ne 0 ]]; then
    echo "Seed $S failed: neg1c=$R1 neg2=$R2" >&2
    false
  fi
  "$PY" e5_holdout_ext_summary.py --run-root "$RUN_ROOT" --out "$ROOT/holdout_convergence_summary" \
    --seeds 0,1,2 --allow-partial --log "$LOG"
done
"$PY" e5_holdout_ext_summary.py --run-root "$RUN_ROOT" --out "$ROOT/holdout_convergence_summary" --log "$LOG"
echo "COMPLETE:$(date -Is)" > "$STATUS"
