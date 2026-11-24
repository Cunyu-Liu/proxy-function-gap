#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/ProxyFunctionGap
OUT=/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq
PY=/root/miniconda3/envs/codonflow/bin/python
E0=/root/rareseq/data/e0
cd "$ROOT"
for seeds in "0 1" "2"; do
  for seed in $seeds; do
    out="$OUT/m1a_shared_per_task_s${seed}"
    [[ ! -e $out ]] || { echo "Refusing to overwrite $out" >&2; exit 1; }
    PYTHONPATH=/root/rareseq/pydeps "$PY" code_rareseq/m1_function_heads.py \
      --arch shared --normalization per_task --lr 3e-4 --seed "$seed" --fold 0 --tracks neg1c,neg2 \
      --windows-dir "$E0/windows" --neg-dir "$E0/neg" --emb-dir "$E0/emb" \
      --positives-audit "$E0/positives_audit.tsv" --split-clstr "$E0/splits/clusters.id90.clstr" \
      --exps-file "$E0/pilot_exps.txt" --out "$out" --log "$ROOT/logs/experiments.jsonl" \
      >"$ROOT/logs/m1a_shared_per_task_s${seed}.log" 2>&1 &
  done
  wait
done
