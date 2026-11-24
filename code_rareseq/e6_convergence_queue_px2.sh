#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/ProxyFunctionGap
BASE=/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/e6
PY=/root/miniconda3/envs/codonflow/bin/python
cd "$ROOT"
for splits in "1 2" "3 4" "5"; do
  pids=""
  for split in $splits; do
    if [[ $split -eq 1 ]]; then
      data=/root/rareseq/data/e6/grimm_ec_v2/split_1
      emb=/root/rareseq/data/e6/esmc300m
    else
      data="$BASE/grimm_ec_v2/split_$split"
      emb="$BASE/esmc300m_split$split"
    fi
    out="$BASE/results_split${split}_convergence"
    [[ ! -e $out ]] || { echo "Refusing to overwrite $out" >&2; exit 1; }
    PYTHONPATH=/root/rareseq/pydeps "$PY" code_rareseq/e6_grimm_baselines.py \
      --data-dir "$data" --emb-dir "$emb" --out "$out" --seeds 0,1,2 \
      --epochs 1000 --patience 20 --min-delta 0.0001 --bootstrap 10000 \
      --log "$ROOT/logs/experiments.jsonl" >"$ROOT/logs/e6_baseline_s${split}_convergence.log" 2>&1 &
    pids="$pids $!"
  done
  for pid in $pids; do wait "$pid"; done
done
echo ALL_GRIMM_CONVERGENCE_RUNS_DONE
