#!/bin/bash
set -uo pipefail
MODE=${1:-fold0}
PY=/root/miniconda3/envs/lucaone_cu128/bin/python
W=/home/ProxyFunctionGap/code_rareseq
FROZEN=/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/external_benchmarks/rnacompete/rnacompete-ray2013-v1.0-id90-strict3train
INPUT=/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/external_benchmarks/rnacompete/ray2013_nature12311
ROOT=/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/rnacompete_ray2013_model_baselines_v1
CONTRACT=/home/ProxyFunctionGap/预注册/RNAcompete_Ray2013_模型基线预注册合同_v1.0_2026-11-26.md
LOGDIR=/home/ProxyFunctionGap/logs/rnacompete_ray2013
LOG=$LOGDIR/experiments.jsonl
STATUS=$LOGDIR/driver.status
LOCK=$LOGDIR/driver.lock
MODEL=$ROOT/model/rnafm-7d6e73ad3b48e042b378f9a788a56ccb4d573a27
REVISION=7d6e73ad3b48e042b378f9a788a56ccb4d573a27
export HF_HOME=/root/hf_home HF_HUB_OFFLINE=1
export PYTHONPATH=/root/rnaft/pypath
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$ROOT" "$LOGDIR"
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "DRIVER_ALREADY_RUNNING $LOCK"
  exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT
cd "$W" || exit 1

resource_gate() {
  local LABEL=$1 MIN_FREE=$2 MIN_RAM_KIB=$3 FREE AVAIL ATTEMPT
  for ATTEMPT in $(seq 1 288); do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)
    AVAIL=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
    if [ "$FREE" -ge "$MIN_FREE" ] && [ "$AVAIL" -ge "$MIN_RAM_KIB" ]; then
      echo "RESOURCE_OK $LABEL free_mib=$FREE mem_available_kib=$AVAIL $(date -Is)"
      return 0
    fi
    echo "RESOURCE_WAIT $LABEL attempt=$ATTEMPT free_mib=$FREE mem_available_kib=$AVAIL $(date -Is)"
    sleep 300
  done
  echo "RESOURCE_BLOCKED $LABEL $(date -Is)" >&2
  return 1
}

run_cuda() {
  local LABEL=$1 MIN_FREE=$2 LOGFILE=$3
  shift 3
  resource_gate "$LABEL" "$MIN_FREE" 12582912 || return 1
  echo "START $LABEL $(date -Is)"
  "$PY" "$@" >"$LOGFILE" 2>&1
  local RC=$?
  echo "END $LABEL rc=$RC $(date -Is)"
  return "$RC"
}

run_cpu() {
  local LABEL=$1 LOGFILE=$2
  shift 2
  local AVAIL
  AVAIL=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
  if [ "$AVAIL" -lt 12582912 ]; then
    echo "RAM_BLOCKED $LABEL available_kib=$AVAIL" >&2
    return 1
  fi
  echo "START $LABEL $(date -Is)"
  "$PY" "$@" >"$LOGFILE" 2>&1
  local RC=$?
  echo "END $LABEL rc=$RC $(date -Is)"
  return "$RC"
}

BASE=(--frozen-root "$FROZEN" --out "$ROOT" --contract "$CONTRACT" --log "$LOG")
echo "RUNNING $MODE $(date -Is)" >"$STATUS"
MODEL_READY=0
for ATTEMPT in $(seq 1 60); do
  if [ -f "$MODEL/pytorch_model.bin" ] && [ "$(stat -c %s "$MODEL/pytorch_model.bin")" -eq 398175831 ]; then
    MODEL_READY=1
    echo "MODEL_SNAPSHOT_OK revision=$REVISION bytes=398175831"
    break
  fi
  echo "MODEL_SNAPSHOT_WAIT attempt=$ATTEMPT $(date -Is)"
  sleep 60
done
if [ "$MODEL_READY" -ne 1 ]; then
  echo "MODEL_SNAPSHOT_BLOCKED $(date -Is)" >"$STATUS"
  exit 1
fi
if [ ! -f "$ROOT/smoke/smoke_cuda.json" ]; then
  run_cuda smoke 2500 "$LOGDIR/smoke.log" rnacompete_ray2013_rnafm_embed.py \
    --frozen-root "$FROZEN" --out "$ROOT/smoke" --log "$LOG" --batch 8 --limit 16 \
    --model-path "$MODEL" --revision "$REVISION" || {
      echo "SMOKE_FAILED $(date -Is)" >"$STATUS"; exit 1; }
fi
if [ ! -f "$ROOT/embeddings/embedding_manifest.json" ]; then
  run_cuda shared_embedding 2500 "$LOGDIR/embedding.log" rnacompete_ray2013_rnafm_embed.py \
    --frozen-root "$FROZEN" --out "$ROOT/embeddings" --log "$LOG" --batch 128 \
    --model-path "$MODEL" --revision "$REVISION" || {
      echo "EMBEDDING_FAILED_OR_BLOCKED $(date -Is)" >"$STATUS"; exit 1; }
fi
if [ ! -f "$ROOT/label_manifest.json" ]; then
  run_cpu label_cache "$LOGDIR/label_cache.log" rnacompete_ray2013_baselines.py prepare-labels \
    --input-root "$INPUT" "${BASE[@]}" || { echo "LABEL_CACHE_FAILED $(date -Is)" >"$STATUS"; exit 1; }
fi

for TRACK in sequence rbp_ood; do
  for ARCH in linear mlp; do
    if [ ! -f "$ROOT/selections/${TRACK}_${ARCH}.json" ]; then
      run_cuda "select_${TRACK}_${ARCH}" 2500 "$LOGDIR/select_${TRACK}_${ARCH}.log" \
        rnacompete_ray2013_baselines.py select --track "$TRACK" --arch "$ARCH" --fold 0 --seed 0 "${BASE[@]}" || {
          echo "SELECTION_FAILED ${TRACK}_${ARCH} $(date -Is)" >"$STATUS"; exit 1; }
    fi
  done
  if [ ! -f "$ROOT/selections/${TRACK}_knn.json" ]; then
    run_cuda "select_${TRACK}_knn" 2500 "$LOGDIR/select_${TRACK}_knn.log" \
      rnacompete_ray2013_baselines.py knn-select --track "$TRACK" --fold 0 "${BASE[@]}" || {
        echo "KNN_SELECTION_FAILED $TRACK $(date -Is)" >"$STATUS"; exit 1; }
  fi
done

run_fold() {
  local FOLD=$1 TRACK ARCH SEED RUN
  for TRACK in sequence rbp_ood; do
    for ARCH in linear mlp; do
      for SEED in 0 1 2; do
        RUN=$ROOT/runs/${TRACK}_${ARCH}_f${FOLD}_s${SEED}
        if [ ! -f "$RUN/.train_complete" ]; then
          run_cuda "train_${TRACK}_${ARCH}_f${FOLD}_s${SEED}" 2500 \
            "$LOGDIR/train_${TRACK}_${ARCH}_f${FOLD}_s${SEED}.log" \
            rnacompete_ray2013_baselines.py train --track "$TRACK" --arch "$ARCH" \
            --fold "$FOLD" --seed "$SEED" "${BASE[@]}" || return 1
        fi
        if [ ! -f "$RUN/.eval_complete" ]; then
          run_cuda "evaluate_${TRACK}_${ARCH}_f${FOLD}_s${SEED}" 2500 \
            "$LOGDIR/evaluate_${TRACK}_${ARCH}_f${FOLD}_s${SEED}.log" \
            rnacompete_ray2013_baselines.py evaluate --track "$TRACK" --arch "$ARCH" \
            --fold "$FOLD" --seed "$SEED" "${BASE[@]}" || return 1
        fi
      done
    done
    if [ ! -f "$ROOT/knn/${TRACK}_f${FOLD}/.eval_complete" ]; then
      run_cuda "knn_evaluate_${TRACK}_f${FOLD}" 2500 "$LOGDIR/knn_evaluate_${TRACK}_f${FOLD}.log" \
        rnacompete_ray2013_baselines.py knn-evaluate --track "$TRACK" --fold "$FOLD" "${BASE[@]}" || return 1
    fi
  done
}

run_fold 0 || { echo "FOLD0_FAILED_OR_BLOCKED $(date -Is)" >"$STATUS"; exit 1; }
run_cpu verify_fold0 "$LOGDIR/verify_fold0.log" rnacompete_ray2013_baselines.py verify-fold0 "${BASE[@]}" || {
  echo "FOLD0_VERIFY_FAILED $(date -Is)" >"$STATUS"; exit 1; }
echo "FOLD0_GATE_PASSED $(date -Is)" >"$STATUS"
if [ "$MODE" = fold0 ]; then
  exit 0
fi
if [ "$MODE" != all ]; then
  echo "INVALID_MODE $MODE" >&2
  exit 1
fi
for FOLD in 1 2 3 4; do
  run_fold "$FOLD" || { echo "EXPANSION_FAILED_OR_BLOCKED fold=$FOLD $(date -Is)" >"$STATUS"; exit 1; }
done
run_cpu verify_final "$LOGDIR/verify_final.log" rnacompete_ray2013_baselines.py verify-final "${BASE[@]}" || {
  echo "FINAL_VERIFY_FAILED $(date -Is)" >"$STATUS"; exit 1; }
run_cpu summarize "$LOGDIR/summarize.log" rnacompete_ray2013_baselines.py summarize "${BASE[@]}" || {
  echo "SUMMARY_FAILED $(date -Is)" >"$STATUS"; exit 1; }
echo "ALL_DONE $(date -Is)" >"$STATUS"
