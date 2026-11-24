#!/bin/bash
set -euo pipefail
cd /root/rareseq/work
PY=/root/miniconda3/envs/llr_env/bin/python
export PYTHONPATH=/root/rareseq/pydeps
E0=/root/rareseq/data/e0
E1=/root/rareseq/data/e1
COMMON=(--windows-dir "$E0/windows" --neg-dir "$E0/neg" --emb-dir "$E0/emb"
  --positives-audit "$E0/positives_audit.tsv" --split-clstr "$E0/splits/clusters.id90.clstr"
  --exps-file "$E0/pilot_exps.txt" --tracks neg1c,neg2)
for FOLD in 2 3 4; do
  for SEED in 0 1 2; do
    "$PY" e1_frozen_baselines.py modern "${COMMON[@]}" --folds "$FOLD" --seed "$SEED" \
      --loss bce --out "$E1/pilot_f${FOLD}_modern_bce_s${SEED}"
  done
  "$PY" e1_frozen_baselines.py knn "${COMMON[@]}" --folds "$FOLD" --seed 0 \
    --out "$E1/pilot_f${FOLD}_knn"
done
