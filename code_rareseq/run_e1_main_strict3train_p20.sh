#!/bin/bash
set -uo pipefail
PY=/root/miniconda3/envs/llr_env/bin/python
export PYTHONPATH=/root/rareseq/pydeps
W=/root/rareseq/work
E0=/root/rareseq/data/e0
OUT=/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/e1_main_strict3train_p20_v2
LOGDIR=/home/ProxyFunctionGap/logs
LOG=$LOGDIR/experiments.jsonl
STATUS=$LOGDIR/e1_main_strict3train_p20_driver.status
mkdir -p "$OUT" "$LOGDIR"
cd "$W" || exit 1
COMMON=(--windows-dir "$E0/windows" --neg-dir "$E0/neg" --emb-dir "$E0/emb"
  --positives-audit "$E0/positives_audit.tsv" --split-clstr "$E0/splits/clusters.id90.clstr"
  --exps-file "$E0/main_exps.txt" --tracks neg1c,neg2 --epochs 1000 --patience 20 --min-delta 1e-4
  --log "$LOG" --data-version e0-v2-id90-strict3train-main)
PAIRS=()
for F in 0 1 2 3 4; do for S in 0 1 2; do PAIRS+=("$F:$S"); done; done

run_modern() {
  local F=$1 S=$2 DIR="$OUT/f${1}_modern_bce_s${2}" DONE LOCK L
  DONE="$DIR/.complete"
  if [ -s "$DIR/modern_results.tsv" ] && [ -f "$DONE" ]; then echo "SKIP modern f$F s$S"; return 0; fi
  LOCK="$DIR.lock"
  if ! mkdir "$LOCK" 2>/dev/null; then echo "LOCKED modern f$F s$S" >&2; return 1; fi
  L=$LOGDIR/e1_strict3train_p20_f${F}_s${S}.log
  echo "START modern f$F s$S $(date -Is)"
  "$PY" e1_frozen_baselines.py modern "${COMMON[@]}" --folds "$F" --seed "$S" --loss bce --out "$DIR" >"$L" 2>&1
  local RC=$?
  if [ "$RC" -eq 0 ] && [ -s "$DIR/modern_results.tsv" ]; then
    date -Is > "$DONE"
    echo "DONE modern f$F s$S"
  else
    echo "FAILED modern f$F s$S rc=$RC" >&2
  fi
  rmdir "$LOCK" || return 1
  return "$RC"
}

modern_worker() {
  local I F S PAIR
  I=$1
  while [ "$I" -lt "${#PAIRS[@]}" ]; do
    PAIR=${PAIRS[$I]}; F=${PAIR%%:*}; S=${PAIR##*:}
    run_modern "$F" "$S" || return 1
    I=$((I + 2))
  done
}

run_knn() {
  local F=$1 DIR="$OUT/f${1}_knn" DONE LOCK L
  DONE="$DIR/.complete"
  if [ -s "$DIR/knn_results.tsv" ] && [ -f "$DONE" ]; then echo "SKIP knn f$F"; return 0; fi
  LOCK="$DIR.lock"
  if ! mkdir "$LOCK" 2>/dev/null; then echo "LOCKED knn f$F" >&2; return 1; fi
  L=$LOGDIR/e1_strict3train_p20_f${F}_knn.log
  echo "START knn f$F $(date -Is)"
  "$PY" e1_frozen_baselines.py knn "${COMMON[@]}" --folds "$F" --seed 0 --out "$DIR" >"$L" 2>&1
  local RC=$?
  if [ "$RC" -eq 0 ] && [ -s "$DIR/knn_results.tsv" ]; then
    date -Is > "$DONE"
    echo "DONE knn f$F"
  else
    echo "FAILED knn f$F rc=$RC" >&2
  fi
  rmdir "$LOCK" || return 1
  return "$RC"
}

knn_worker() {
  local F
  for F in "$@"; do run_knn "$F" || return 1; done
}

echo "RUNNING $(date -Is)" > "$STATUS"
modern_worker 0 & P0=$!
modern_worker 1 & P1=$!
RC=0
wait "$P0" || RC=1
wait "$P1" || RC=1
if [ "$RC" -ne 0 ]; then echo "FAILED modern $(date -Is)" > "$STATUS"; exit 1; fi

KNN_JOBS=${KNN_JOBS:-2}
if [ "$KNN_JOBS" = 2 ]; then
  knn_worker 0 2 4 & K0=$!
  knn_worker 1 3 & K1=$!
  wait "$K0" || RC=1
  wait "$K1" || RC=1
elif [ "$KNN_JOBS" = 1 ]; then
  knn_worker 0 1 2 3 4 || RC=1
else
  echo "INVALID KNN_JOBS=$KNN_JOBS" >&2
  exit 1
fi
if [ "$RC" -ne 0 ]; then echo "FAILED knn $(date -Is)" > "$STATUS"; exit 1; fi
echo "ALL_DONE $(date -Is)" > "$STATUS"
