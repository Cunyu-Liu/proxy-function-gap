#!/bin/bash
set -euo pipefail
PY=/root/miniconda3/envs/llr_env/bin/python
W=/home/ProxyFunctionGap/code_rareseq
E0=/root/rareseq/data/e0
PILOT_EMB=/root/rareseq/data/e1/emb_rinalmo
BASE=/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/e1_rinalmo_main_strict3train_p20_v1
EMB=$BASE/embeddings
SMOKE=$BASE/smoke
ROOT=$BASE/main
LOGDIR=/home/ProxyFunctionGap/logs/e1_rinalmo_main
LOG=$LOGDIR/experiments.jsonl
STATUS=$LOGDIR/driver.status
SMOKE_EXPS=$LOGDIR/smoke_tail_exp.txt
MAIN_EXPS=$E0/main_exps.txt
DATA_VERSION=e0-v2-id90-strict3train-main
export PYTHONPATH=/root/rareseq/pydeps
export HF_HOME=/root/hf_home
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
mkdir -p "$EMB" "$SMOKE" "$ROOT" "$LOGDIR"
cd "$W"
exec 9>"$LOGDIR/driver.lock"
if ! flock -n 9; then
  echo "another RiNALMo driver holds the lock" >&2
  exit 0
fi
on_exit() {
  local rc=$?
  if [ "$rc" -ne 0 ]; then echo "FAILED rc=$rc $(date -Is)" > "$STATUS"; fi
}
trap on_exit EXIT
set_status() { echo "$1 $(date -Is)" > "$STATUS"; }

gpu_wait() {
  local required_free=$1 required_ram=$2 free used total ram
  while true; do
    read -r total used < <(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i 0)
    total=${total//,/}
    used=${used//,/}
    free=$((total - used))
    ram=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    if [ "$free" -ge "$required_free" ] && [ "$ram" -ge "$required_ram" ]; then
      echo "RESOURCE_OK required_free=${required_free}MiB actual_free=${free}MiB required_ram=${required_ram}GiB available_ram=${ram}GiB $(date -Is)"
      return 0
    fi
    echo "RESOURCE_WAIT required_free=${required_free}MiB actual_free=${free}MiB required_ram=${required_ram}GiB available_ram=${ram}GiB $(date -Is)"
    sleep 300
  done
}

storage_wait() {
  local required=$1 available
  while true; do
    available=$(df -k "$BASE" | awk 'NR==2{print int($4/1048576)}')
    if [ "$available" -ge "$required" ]; then
      echo "STORAGE_OK required_free=${required}GiB available=${available}GiB $(date -Is)"
      return 0
    fi
    echo "STORAGE_WAIT required_free=${required}GiB available=${available}GiB $(date -Is)"
    set_status "WAITING storage required=${required}GiB available=${available}GiB"
    sleep 300
  done
}

run_embed() {
  local exps=$1 log=$2
  storage_wait 3
  set_status "RUNNING embedding $(basename "$exps")"
  gpu_wait 9000 16
  PYTHONPATH=/root/rnaft/pypath:/root/rareseq/pydeps "$PY" e1_rinalmo_embed.py \
    --windows-dir "$E0/windows" --neg-dir "$E0/neg" --exps-file "$exps" \
    --out "$EMB" --reuse-from "$PILOT_EMB" --batch 64 --max-length 103 --dim 1280 \
    --log "$LOG" --data-version "$DATA_VERSION" > "$log" 2>&1
}

run_modern() {
  local exps=$1 root=$2 tag=$3 fold=$4 seed=$5
  local dir=$root/f${fold}_modern_bce_s${seed}
  local done=$dir/.complete lock=$dir.lock log=$LOGDIR/${tag}_f${fold}_s${seed}.log rc
  if [ -s "$dir/modern_results.tsv" ] && [ -f "$done" ]; then echo "SKIP modern $tag f$fold s$seed"; return 0; fi
  if ! mkdir "$lock" 2>/dev/null; then echo "LOCKED modern $tag f$fold s$seed" >&2; return 1; fi
  set_status "RUNNING modern $tag f$fold s$seed"
  storage_wait 1
  set_status "RUNNING modern $tag f$fold s$seed"
  gpu_wait 6000 12
  if "$PY" e1_rinalmo_frozen_baselines.py modern --windows-dir "$E0/windows" --neg-dir "$E0/neg" \
      --emb-dir "$EMB" --positives-audit "$E0/positives_audit.tsv" \
      --split-clstr "$E0/splits/clusters.id90.clstr" --exps-file "$exps" \
      --folds "$fold" --tracks neg1c,neg2 --seed "$seed" --model rinalmo-giga --loss bce \
      --epochs 1000 --patience 20 --min-delta 1e-4 --batch 512 --out "$dir" \
      --log "$LOG" --data-version "$DATA_VERSION" > "$log" 2>&1; then rc=0; else rc=$?; fi
  if [ "$rc" -eq 0 ] && [ -s "$dir/modern_results.tsv" ]; then
    date -Is > "$done"
  else
    rmdir "$lock"
    if [ "$rc" -eq 0 ]; then rc=1; fi
    return "$rc"
  fi
  rmdir "$lock"
  echo "DONE modern $tag f$fold s$seed"
}

run_knn() {
  local exps=$1 root=$2 tag=$3 fold=$4
  local dir=$root/f${fold}_knn
  local done=$dir/.complete lock=$dir.lock log=$LOGDIR/${tag}_f${fold}_knn.log rc
  if [ -s "$dir/knn_results.tsv" ] && [ -f "$done" ]; then echo "SKIP knn $tag f$fold"; return 0; fi
  if ! mkdir "$lock" 2>/dev/null; then echo "LOCKED knn $tag f$fold" >&2; return 1; fi
  set_status "RUNNING knn $tag f$fold"
  storage_wait 1
  set_status "RUNNING knn $tag f$fold"
  gpu_wait 8000 18
  if "$PY" e1_rinalmo_knn.py --windows-dir "$E0/windows" --neg-dir "$E0/neg" \
      --emb-dir "$EMB" --positives-audit "$E0/positives_audit.tsv" \
      --split-clstr "$E0/splits/clusters.id90.clstr" --exps-file "$exps" \
      --folds "$fold" --tracks neg1c,neg2 --seed 0 --model rinalmo-giga --ks 5,10,20 \
      --out "$dir" --log "$LOG" --data-version "$DATA_VERSION" > "$log" 2>&1; then rc=0; else rc=$?; fi
  if [ "$rc" -eq 0 ] && [ -s "$dir/knn_results.tsv" ]; then
    date -Is > "$done"
  else
    rmdir "$lock"
    if [ "$rc" -eq 0 ]; then rc=1; fi
    return "$rc"
  fi
  rmdir "$lock"
  echo "DONE knn $tag f$fold"
}

run_audit() {
  local exps=$1 root=$2 out=$3 folds=$4 seeds=$5 log=$6
  "$PY" e1_rinalmo_audit.py --root "$root" --windows-dir "$E0/windows" --neg-dir "$E0/neg" \
    --positives-audit "$E0/positives_audit.tsv" --exps-file "$exps" --folds "$folds" --seeds "$seeds" \
    --log "$LOG" --out "$out" > "$log" 2>&1
}

if [ ! -s "$SMOKE_EXPS" ]; then
  "$PY" - "$E0" "$SMOKE_EXPS" <<'PY'
import os
import sys
import pandas as pd
from e1_rinalmo_frozen_baselines import load_tiers, rbp_of
root, out = sys.argv[1:]
main = [x.strip() for x in open(os.path.join(root, "main_exps.txt")) if x.strip()]
pilot = {x.strip() for x in open(os.path.join(root, "pilot_exps.txt")) if x.strip()}
tiers = load_tiers(os.path.join(root, "positives_audit.tsv"))
candidates = []
for experiment in main:
    if experiment in pilot or tiers[rbp_of(experiment)] != "tail":
        continue
    rows = len(pd.read_csv(os.path.join(root, "windows", experiment + ".windows.tsv"), sep="\t"))
    rows += len(pd.read_csv(os.path.join(root, "neg", experiment + ".negatives.tsv"), sep="\t"))
    candidates.append((rows, experiment))
if not candidates:
    raise RuntimeError("no non-pilot tail experiment for CUDA smoke")
rows, experiment = min(candidates)
with open(out, "w") as handle:
    handle.write(experiment + "\n")
print("SMOKE_EXPERIMENT", experiment, rows)
PY
fi

set_status "RUNNING smoke embedding $(cat "$SMOKE_EXPS")"
run_embed "$SMOKE_EXPS" "$LOGDIR/smoke_embed.log"
run_modern "$SMOKE_EXPS" "$SMOKE" smoke 0 0
run_knn "$SMOKE_EXPS" "$SMOKE" smoke 0
run_audit "$SMOKE_EXPS" "$SMOKE" "$SMOKE/audit" 0 0 "$LOGDIR/smoke_audit.log"
echo "SMOKE_ACCEPTED $(date -Is) $(cat "$SMOKE_EXPS")" > "$SMOKE/ACCEPTED"
set_status "SMOKE_ACCEPTED $(cat "$SMOKE_EXPS")"

set_status "RUNNING main embedding 146 experiments"
run_embed "$MAIN_EXPS" "$LOGDIR/main_embed.log"
run_modern "$MAIN_EXPS" "$ROOT" main 0 0
run_knn "$MAIN_EXPS" "$ROOT" main 0
run_audit "$MAIN_EXPS" "$ROOT" "$BASE/acceptance/fold0_seed0" 0 0 "$LOGDIR/main_fold0_seed0_audit.log"
echo "FIRST_MAIN_ACCEPTED $(date -Is)" > "$BASE/acceptance/first_main_accepted"

PAIRS=()
for fold in 0 1 2 3 4; do
  for seed in 0 1 2; do
    if [ "$fold:$seed" != 0:0 ]; then PAIRS+=("$fold:$seed"); fi
  done
done
modern_worker() {
  local index=$1 pair fold seed
  while [ "$index" -lt "${#PAIRS[@]}" ]; do
    pair=${PAIRS[$index]}; fold=${pair%%:*}; seed=${pair##*:}
    run_modern "$MAIN_EXPS" "$ROOT" main "$fold" "$seed" || return 1
    index=$((index + 2))
  done
}
set_status "RUNNING remaining BCE matrix"
modern_worker 0 & worker0=$!
modern_worker 1 & worker1=$!
rc=0
wait "$worker0" || rc=1
wait "$worker1" || rc=1
if [ "$rc" -ne 0 ]; then exit 1; fi
for fold in 1 2 3 4; do run_knn "$MAIN_EXPS" "$ROOT" main "$fold"; done

set_status "RUNNING full independent audit and summaries"
run_audit "$MAIN_EXPS" "$ROOT" "$ROOT/audit" 0,1,2,3,4 0,1,2 "$LOGDIR/main_full_audit.log"
"$PY" e1_combine_cv_predictions.py --e1-root "$ROOT" --prefix "" --out "$ROOT/cv_combined" \
  --folds 0,1,2,3,4 --seeds 0,1,2 --log "$LOG" --data-version "$DATA_VERSION" > "$LOGDIR/main_combine.log" 2>&1
"$PY" e1_cv_summary.py --e1-root "$ROOT" --prefix "" --out "$ROOT/cv_summary" \
  --folds 0,1,2,3,4 --seeds 0,1,2 --log "$LOG" --data-version "$DATA_VERSION" > "$LOGDIR/main_summary.log" 2>&1
set_status "ALL_DONE"
echo "E1_RINALMO_MAIN_ALL_DONE $(date -Is)"
