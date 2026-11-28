#!/bin/bash
# RNAcompete Ray2013 frozen RiNALMo-giga baselines (protocol addendum 2026-11-28).
# Mirrors run_rnacompete_ray2013_baselines.sh; separate artifact root; RNA-FM tree untouched.
set -uo pipefail
MODE=${1:-all}
PY=/root/miniconda3/envs/llr_env/bin/python
W=/home/ProxyFunctionGap/code_rareseq
FROZEN=/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/external_benchmarks/rnacompete/rnacompete-ray2013-v1.0-id90-strict3train
INPUT=/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/external_benchmarks/rnacompete/ray2013_nature12311
ROOT=/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/rnacompete_ray2013_rinalmo_baselines_v1
CONTRACT=/home/ProxyFunctionGap/预注册/RNAcompete_Ray2013_RiNALMo扩展协议_2026-11-28.md
LOGDIR=/home/ProxyFunctionGap/logs/rnacompete_ray2013_rinalmo
LOG=$LOGDIR/experiments.jsonl
STATUS=$LOGDIR/driver.status
LOCK=$LOGDIR/driver.lock
MODEL=/root/hf_home/hub/models--multimolecule--rinalmo-giga/snapshots/2a71f6f98fb41dd2e6542a5e131d3778111d1468
REVISION=2a71f6f98fb41dd2e6542a5e131d3778111d1468
export HF_HOME=/root/hf_home HF_HUB_OFFLINE=1
export PYTHONPATH=/root/rnaft/pypath:/root/rareseq/pydeps
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
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
  echo "RESOURCE_BLOCKED $LABEL" >&2
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

BASE=(--frozen-root "$FROZEN" --out "$ROOT" --contract "$CONTRACT" --log "$LOG" --encoder rinalmo --emb-dim 1280)
echo "RUNNING $MODE $(date -Is)" >"$STATUS"
if [ ! -f "$MODEL/pytorch_model.bin" ] && [ ! -f "$MODEL/model.safetensors" ]; then
  echo "MODEL_SNAPSHOT_BLOCKED $(date -Is)" >"$STATUS"
  exit 1
fi
echo "MODEL_SNAPSHOT_OK revision=$REVISION"
if [ ! -f "$ROOT/smoke/smoke_cuda.json" ]; then
  run_cuda smoke 4000 "$LOGDIR/smoke.log" rnacompete_ray2013_rinalmo_embed.py \
    --frozen-root "$FROZEN" --out "$ROOT/smoke" --log "$LOG" --batch 8 --limit 16 \
    --model-path "$MODEL" --revision "$REVISION" || {
      echo "SMOKE_FAILED $(date -Is)" >"$STATUS"; exit 1; }
fi
if [ ! -f "$ROOT/embeddings/embedding_manifest.json" ]; then
  run_cuda shared_embedding 9000 "$LOGDIR/embedding.log" rnacompete_ray2013_rinalmo_embed.py \
    --frozen-root "$FROZEN" --out "$ROOT/embeddings" --log "$LOG" --batch 64 \
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
      run_cuda "select_${TRACK}_${ARCH}" 2000 "$LOGDIR/select_${TRACK}_${ARCH}.log" \
        rnacompete_ray2013_baselines.py select --track "$TRACK" --arch "$ARCH" --fold 0 --seed 0 "${BASE[@]}" || {
          echo "SELECTION_FAILED ${TRACK}_${ARCH} $(date -Is)" >"$STATUS"; exit 1; }
    fi
  done
  if [ ! -f "$ROOT/selections/${TRACK}_knn.json" ]; then
    run_cuda "select_${TRACK}_knn" 2000 "$LOGDIR/select_${TRACK}_knn.log" \
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
          run_cuda "train_${TRACK}_${ARCH}_f${FOLD}_s${SEED}" 2000 \
            "$LOGDIR/train_${TRACK}_${ARCH}_f${FOLD}_s${SEED}.log" \
            rnacompete_ray2013_baselines.py train --track "$TRACK" --arch "$ARCH" \
            --fold "$FOLD" --seed "$SEED" "${BASE[@]}" || return 1
        fi
        if [ ! -f "$RUN/.eval_complete" ]; then
          run_cuda "evaluate_${TRACK}_${ARCH}_f${FOLD}_s${SEED}" 2000 \
            "$LOGDIR/evaluate_${TRACK}_${ARCH}_f${FOLD}_s${SEED}.log" \
            rnacompete_ray2013_baselines.py evaluate --track "$TRACK" --arch "$ARCH" \
            --fold "$FOLD" --seed "$SEED" "${BASE[@]}" || return 1
        fi
      done
    done
    if [ ! -f "$ROOT/knn/${TRACK}_f${FOLD}/.eval_complete" ]; then
      run_cuda "knn_evaluate_${TRACK}_f${FOLD}" 2000 "$LOGDIR/knn_evaluate_${TRACK}_f${FOLD}.log" \
        rnacompete_ray2013_baselines.py knn-evaluate --track "$TRACK" --fold "$FOLD" "${BASE[@]}" || return 1
    fi
  done
}

run_fold 0 || { echo "FOLD0_FAILED_OR_BLOCKED $(date -Is)" >"$STATUS"; exit 1; }
run_cuda verify_fold0 2000 "$LOGDIR/verify_fold0.log" rnacompete_ray2013_baselines.py verify-fold0 "${BASE[@]}" || {
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
run_cuda verify_final 2000 "$LOGDIR/verify_final.log" rnacompete_ray2013_baselines.py verify-final "${BASE[@]}" || {
  echo "FINAL_VERIFY_FAILED $(date -Is)" >"$STATUS"; exit 1; }
run_cpu summarize "$LOGDIR/summarize.log" rnacompete_ray2013_baselines.py summarize "${BASE[@]}" || {
  echo "SUMMARY_FAILED $(date -Is)" >"$STATUS"; exit 1; }
echo "ALL_DONE $(date -Is)" >"$STATUS"