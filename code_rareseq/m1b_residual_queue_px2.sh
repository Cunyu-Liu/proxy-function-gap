#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/ProxyFunctionGap
OUT=/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq
PY=/root/miniconda3/envs/codonflow/bin/python
E0=/root/rareseq/data/e0
cd "$ROOT"
for rank in 4 8 16; do
  for shrinkage in 0.1 1 10; do
    for seeds in "0 1" "2"; do
      for seed in $seeds; do
        out="$OUT/m1b_residual_r${rank}_l${shrinkage}_s${seed}"
        if [[ -e $out ]]; then
          echo "Refusing to overwrite $out" >&2
          exit 1
        fi
        PYTHONPATH=/root/rareseq/pydeps "$PY" code_rareseq/m1_function_heads.py \
          --arch residual --rank "$rank" --shrinkage "$shrinkage" --seed "$seed" \
          --fold 0 --tracks neg1c,neg2 \
          --windows-dir "$E0/windows" --neg-dir "$E0/neg" --emb-dir "$E0/emb" \
          --positives-audit "$E0/positives_audit.tsv" --split-clstr "$E0/splits/clusters.id90.clstr" \
          --exps-file "$E0/pilot_exps.txt" --out "$out" --log "$ROOT/logs/experiments.jsonl" \
          >"$ROOT/logs/m1b_residual_r${rank}_l${shrinkage}_s${seed}.log" 2>&1 &
      done
      wait
    done
  done
done
