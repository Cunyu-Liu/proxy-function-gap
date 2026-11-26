#!/bin/bash
set -uo pipefail
PY=/root/miniconda3/envs/llr_env/bin/python
export PYTHONPATH=/root/rareseq/pydeps:/root/rnaft/pypath
export HF_HOME=/root/hf_home HF_HUB_OFFLINE=1
W=/root/rareseq/work
E0=/root/rareseq/data/e0
ROOT=/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/e1_rnafm_finetune_strict3train_p20
LOGDIR=/home/ProxyFunctionGap/logs
LOG=$LOGDIR/experiments.jsonl
STATUS=$LOGDIR/e1_finetune_main.status
SELECTION=$ROOT/fold0_lora_rank_selection.json
CUDA_LOCK=$LOGDIR/e1_finetune_cuda_train.lock
mkdir -p "$ROOT" "$LOGDIR"
cd "$W" || exit 1
COMMON=(train --windows-dir "$E0/windows" --neg-dir "$E0/neg"
  --positives-audit "$E0/positives_audit.tsv" --split-clstr "$E0/splits/clusters.id90.clstr"
  --exps-file "$E0/main_exps.txt" --tracks neg1c,neg2
  --epochs 1000 --patience 20 --min-delta 1e-4 --batch 256
  --data-version e0-v2-id90-strict3train-main --log "$LOG")

resource_gate() {
  local LABEL=$1 MIN_FREE=$2 FREE_MIB AVAIL_KIB ATTEMPT
  for ATTEMPT in $(seq 1 36); do
    FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)
    AVAIL_KIB=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
    if [ "$FREE_MIB" -ge "$MIN_FREE" ] && [ "$AVAIL_KIB" -ge 12582912 ]; then
      echo "RESOURCE_OK $LABEL free_mib=$FREE_MIB mem_available_kib=$AVAIL_KIB"
      return 0
    fi
    echo "RESOURCE_WAIT $LABEL attempt=$ATTEMPT free_mib=$FREE_MIB mem_available_kib=$AVAIL_KIB"
    sleep 300
  done
  echo "RESOURCE_BLOCKED $LABEL" >&2
  return 1
}

cuda_lock_acquire() {
  local LABEL=$1 ATTEMPT
  for ATTEMPT in $(seq 1 36); do
    if mkdir "$CUDA_LOCK" 2>/dev/null; then
      echo "$LABEL $HOSTNAME $$" > "$CUDA_LOCK/owner"
      return 0
    fi
    echo "CUDA_LOCK_WAIT $LABEL attempt=$ATTEMPT owner=$(cat "$CUDA_LOCK/owner" 2>/dev/null)"
    sleep 300
  done
  return 1
}

storage_gate() {
  local MODE=$1 EXPS=$2 ATTEMPT TASKS PER FREE REQUIRED
  TASKS=$(( $(wc -l < "$EXPS") * 2 ))
  if [ "$MODE" = full ]; then PER=420000000; else PER=4000000; fi
  REQUIRED=$((TASKS * PER + 21474836480))
  for ATTEMPT in $(seq 1 36); do
    FREE=$(df -B1 "$ROOT" | awk 'NR==2 {print $4}')
    if [ "$FREE" -ge "$REQUIRED" ]; then
      echo "STORAGE_OK tasks=$TASKS free=$FREE required=$REQUIRED"
      return 0
    fi
    echo "STORAGE_WAIT tasks=$TASKS attempt=$ATTEMPT free=$FREE required=$REQUIRED"
    sleep 300
  done
  return 1
}

train_run() {
  local MODE=$1 FOLD=$2 SEED=$3 RANK=$4 DIR=$5 LR=3e-4 MIN_FREE=14000 LOGFILE=$6 EXPS=${7:-$E0/main_exps.txt}
  if [ "$MODE" = full ]; then LR=1e-5; MIN_FREE=16000; fi
  if [ -f "$DIR/.train_complete" ]; then echo "SKIP train $DIR"; return 0; fi
  storage_gate "$MODE" "$EXPS" || return 1
  resource_gate "train_f${FOLD}_s${SEED}_${MODE}" "$MIN_FREE" || return 1
  cuda_lock_acquire "train_f${FOLD}_s${SEED}_${MODE}" || return 1
  if ! resource_gate "locked_train_f${FOLD}_s${SEED}_${MODE}" "$MIN_FREE"; then
    rm -f "$CUDA_LOCK/owner"; rmdir "$CUDA_LOCK" 2>/dev/null; return 1
  fi
  echo "TRAIN_START $DIR $(date -Is)"
  "$PY" e1_finetune.py "${COMMON[@]}" --mode "$MODE" --folds "$FOLD" --seed "$SEED" \
    --encoder-lr "$LR" --lora-r "$RANK" --exps-file "$EXPS" --out "$DIR" >"$LOGFILE" 2>&1
  local RC=$?
  rm -f "$CUDA_LOCK/owner"; rmdir "$CUDA_LOCK" 2>/dev/null || RC=1
  echo "TRAIN_END $DIR rc=$RC $(date -Is)"
  return "$RC"
}

validate_run() {
  local DIR=$1 LOGFILE=$2 RC
  if [ ! -f "$DIR/.independent_validation_complete" ]; then
    resource_gate "validate_${DIR##*/}" 12000 || return 1
    cuda_lock_acquire "validate_${DIR##*/}" || return 1
    if ! resource_gate "locked_validate_${DIR##*/}" 12000; then
      rm -f "$CUDA_LOCK/owner"; rmdir "$CUDA_LOCK" 2>/dev/null; return 1
    fi
    "$PY" e1_finetune_validate.py --out "$DIR" --log "$LOG" >"$LOGFILE" 2>&1
    RC=$?
    rm -f "$CUDA_LOCK/owner"; rmdir "$CUDA_LOCK" 2>/dev/null || RC=1
    [ "$RC" -eq 0 ] || return "$RC"
  fi
  "$PY" e1_finetune_verify.py --runs "$DIR" --require-independent-validation \
    --out "$DIR/validation_only_verification.json" >"$LOGFILE.verify" 2>&1
}

eval_run() {
  local DIR=$1 LOGFILE=$2 REPRESENTATIVE_FILE=${3:-} RC
  if [ ! -f "$DIR/.eval_complete" ]; then
    resource_gate "eval_${DIR##*/}" 12000 || return 1
    cuda_lock_acquire "eval_${DIR##*/}" || return 1
    if ! resource_gate "locked_eval_${DIR##*/}" 12000; then
      rm -f "$CUDA_LOCK/owner"; rmdir "$CUDA_LOCK" 2>/dev/null; return 1
    fi
    echo "EVAL_START $DIR $(date -Is)"
    if [[ "$DIR" == *lora* ]]; then
      "$PY" e1_finetune.py evaluate --out "$DIR" --selection "$SELECTION" --log "$LOG" >"$LOGFILE" 2>&1
    else
      "$PY" e1_finetune.py evaluate --out "$DIR" --log "$LOG" >"$LOGFILE" 2>&1
    fi
    RC=$?
    rm -f "$CUDA_LOCK/owner"; rmdir "$CUDA_LOCK" 2>/dev/null || RC=1
    echo "EVAL_END $DIR rc=$RC $(date -Is)"
    [ "$RC" -eq 0 ] || return "$RC"
  fi
  if [ -f "$DIR/.retention_complete" ]; then echo "SKIP retention $DIR"; return 0; fi
  "$PY" e1_finetune_verify.py --runs "$DIR" --require-eval \
    --out "$DIR/pre_retention_verification.json" >"$LOGFILE.pre_retention_verify" 2>&1 || return 1
  local REP_ARGS=()
  if [ -n "$REPRESENTATIVE_FILE" ]; then REP_ARGS=(--representative-file "$REPRESENTATIVE_FILE"); fi
  if [[ "$DIR" == *lora* ]]; then
    "$PY" e1_finetune_retention.py --run "$DIR" --verification "$DIR/pre_retention_verification.json" \
      --selection "$SELECTION" "${REP_ARGS[@]}" --log "$LOG" >"$LOGFILE.retention" 2>&1 || return 1
  else
    "$PY" e1_finetune_retention.py --run "$DIR" --verification "$DIR/pre_retention_verification.json" \
      "${REP_ARGS[@]}" --log "$LOG" >"$LOGFILE.retention" 2>&1 || return 1
  fi
}

lora_fold0_dir() {
  local SEED=$1 RANK
  if [ "$SEED" -eq 0 ]; then
    RANK=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_rank"])' "$SELECTION")
    echo "$ROOT/f0_lora_r${RANK}_s0"
  else
    echo "$ROOT/f0_lora_s${SEED}"
  fi
}

lora_worker() {
  local R DIR LOGFILE SEED FOLD RANK
  for R in 4 8 16; do
    DIR=$ROOT/f0_lora_r${R}_s0
    LOGFILE=$LOGDIR/e1_finetune_f0_lora_r${R}_s0.log
    train_run lora 0 0 "$R" "$DIR" "$LOGFILE" || return 1
    validate_run "$DIR" "$LOGFILE.independent_validation" || return 1
  done
  "$PY" e1_finetune_verify.py --runs "$ROOT/f0_lora_r4_s0,$ROOT/f0_lora_r8_s0,$ROOT/f0_lora_r16_s0" \
    --require-independent-validation --out "$ROOT/fold0_lora_rank_search_verification.json" \
    >"$LOGDIR/e1_finetune_rank_search_verification.log" 2>&1 || return 1
  "$PY" e1_finetune_select.py --rank4 "$ROOT/f0_lora_r4_s0" --rank8 "$ROOT/f0_lora_r8_s0" \
    --rank16 "$ROOT/f0_lora_r16_s0" --out "$SELECTION" >"$LOGDIR/e1_finetune_rank_selection.log" 2>&1 || return 1
  for SEED in 0 1 2; do
    RANK=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_rank"])' "$SELECTION")
    DIR=$(lora_fold0_dir "$SEED")
    LOGFILE=$LOGDIR/e1_finetune_f0_lora_s${SEED}.log
    train_run lora 0 "$SEED" "$RANK" "$DIR" "$LOGFILE.train" || return 1
    eval_run "$DIR" "$LOGFILE.eval" || return 1
  done
  echo "LORA_FOLD0_DONE $(date -Is)"
}

chunked_full_run() {
  local FOLD=$1 SEED=$2 PLAN_DIR=$ROOT/chunk_plan_8 EXPS IDX DIR LOGFILE REP
  if [ ! -f "$PLAN_DIR/chunk_manifest.json" ]; then
    "$PY" e1_finetune_chunk_plan.py --windows-dir "$E0/windows" --neg-dir "$E0/neg" \
      --positives-audit "$E0/positives_audit.tsv" --exps-file "$E0/main_exps.txt" \
      --folds 0 --seed 0 --tracks neg1c,neg2 --chunk-experiments 8 --out "$PLAN_DIR" \
      >"$LOGDIR/e1_finetune_chunk_plan.log" 2>&1 || return 1
  fi
  for EXPS in $("$PY" -c 'import json,sys; print(" ".join(x["exps_file"] for x in json.load(open(sys.argv[1]))["chunks"]))' "$PLAN_DIR/chunk_manifest.json"); do
    IDX=$(basename "$EXPS" .exps.txt)
    DIR=$ROOT/f${FOLD}_full_s${SEED}/chunks/$IDX
    LOGFILE=$LOGDIR/e1_finetune_f${FOLD}_full_s${SEED}_${IDX}.log
    REP=""
    if [ "$FOLD" -eq 0 ] && [ "$SEED" -eq 0 ]; then REP=$PLAN_DIR/representatives.json; fi
    train_run full "$FOLD" "$SEED" 8 "$DIR" "$LOGFILE.train" "$EXPS" || return 1
    eval_run "$DIR" "$LOGFILE.eval" "$REP" || return 1
  done
  date -Is > "$ROOT/f${FOLD}_full_s${SEED}/.chunks_complete"
}

full_chunk_runs() {
  local FOLD=$1 SEED=$2
  "$PY" -c 'import json,sys,os; p=sys.argv[1]; root=sys.argv[2]; print(",".join(os.path.join(root,"chunks","chunk_%03d"%x["index"]) for x in json.load(open(p))["chunks"]))' \
    "$ROOT/chunk_plan_8/chunk_manifest.json" "$ROOT/f${FOLD}_full_s${SEED}"
}

full_worker() {
  local SEED DIR LOGFILE
  for SEED in 0 1 2; do
    chunked_full_run 0 "$SEED" || return 1
  done
}

expansion_worker() {
  local MODE=$1 FOLD SEED DIR LOGFILE RANK
  RANK=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_rank"])' "$SELECTION")
  for FOLD in 1 2 3 4; do
    for SEED in 0 1 2; do
      if [ "$MODE" = full ]; then
        chunked_full_run "$FOLD" "$SEED" || return 1
      else
        DIR=$ROOT/f${FOLD}_${MODE}_s${SEED}
        LOGFILE=$LOGDIR/e1_finetune_f${FOLD}_${MODE}_s${SEED}.log
        train_run "$MODE" "$FOLD" "$SEED" "$RANK" "$DIR" "$LOGFILE.train" || return 1
        eval_run "$DIR" "$LOGFILE.eval" || return 1
      fi
    done
  done
}

if [ "${E1_FINETUNE_CHUNK_SMOKE:-0}" = 1 ]; then
  SMOKE_DIR=$ROOT/chunk_retention_smoke_full_f1
  train_run full 1 0 8 "$SMOKE_DIR" "$LOGDIR/e1_finetune_chunk_smoke.log.train" "$E0/e1_smoke_exp.txt" || exit 1
  eval_run "$SMOKE_DIR" "$LOGDIR/e1_finetune_chunk_smoke.log.eval" || exit 1
  "$PY" e1_finetune_verify.py --runs "$SMOKE_DIR" --require-eval \
    --out "$SMOKE_DIR/post_retention_verification.json" >"$LOGDIR/e1_finetune_chunk_smoke.log.verify" 2>&1 || exit 1
  echo "CHUNK_SMOKE_PASSED $(date -Is)" > "$STATUS"
  exit 0
fi

if [ ! -f "$ROOT/pilot_cuda_verification.json" ]; then
  echo "PILOT_GATE_MISSING" > "$STATUS"
  exit 1
fi
echo "FOLD0_RUNNING $(date -Is)" > "$STATUS"
lora_worker & P0=$!
full_worker & P1=$!
RC=0
wait "$P0" || RC=1
wait "$P1" || RC=1
if [ "$RC" -ne 0 ]; then
  date -Is > "$ROOT/.fold0_gate_failed"
  echo "FOLD0_FAILED_OR_BLOCKED $(date -Is)" > "$STATUS"
  exit 1
fi
RANK=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_rank"])' "$SELECTION")
L0=$(lora_fold0_dir 0)
FULL_RUNS="$(full_chunk_runs 0 0),$(full_chunk_runs 0 1),$(full_chunk_runs 0 2)"
RUNS="$L0,$ROOT/f0_lora_s1,$ROOT/f0_lora_s2,$FULL_RUNS"
"$PY" e1_finetune_verify.py --runs "$RUNS" --require-eval --out "$ROOT/fold0_gate_verification.json" \
  >"$LOGDIR/e1_finetune_fold0_gate.log" 2>&1 || {
  date -Is > "$ROOT/.fold0_gate_failed"; echo "FOLD0_VERIFY_FAILED $(date -Is)" > "$STATUS"; exit 1; }
date -Is > "$ROOT/.fold0_gate_passed"
echo "FOLD0_PASSED_EXPANSION_RUNNING $(date -Is)" > "$STATUS"
expansion_worker lora & P0=$!
expansion_worker full & P1=$!
wait "$P0" || RC=1
wait "$P1" || RC=1
if [ "$RC" -ne 0 ]; then echo "EXPANSION_FAILED_OR_BLOCKED $(date -Is)" > "$STATUS"; exit 1; fi
ALL_RUNS=""
for F in 0 1 2 3 4; do
  for S in 0 1 2; do
    if [ "$F" -eq 0 ] && [ "$S" -eq 0 ]; then D=$L0; else D=$ROOT/f${F}_lora_s${S}; fi
    ALL_RUNS="$ALL_RUNS,$D"
    ALL_RUNS="$ALL_RUNS,$(full_chunk_runs "$F" "$S")"
  done
done
ALL_RUNS=${ALL_RUNS#,}
"$PY" e1_finetune_verify.py --runs "$ALL_RUNS" --require-eval --out "$ROOT/final_matrix_verification.json" \
  >"$LOGDIR/e1_finetune_final_gate.log" 2>&1 || { echo "FINAL_VERIFY_FAILED $(date -Is)" > "$STATUS"; exit 1; }
"$PY" e1_finetune_summary.py --root "$ROOT" --log "$LOG" >"$LOGDIR/e1_finetune_summary.log" 2>&1 || {
  echo "SUMMARY_FAILED $(date -Is)" > "$STATUS"; exit 1; }
echo "ALL_DONE $(date -Is)" > "$STATUS"
