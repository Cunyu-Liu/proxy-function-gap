#!/bin/bash
set -euo pipefail
cd /root/rareseq/work
PY=/root/miniconda3/envs/llr_env/bin/python
export PYTHONPATH=/root/rareseq/pydeps
E0=/root/rareseq/data/e0
E4=/root/rareseq/data/e4
for PI in 0.05 0.1 0.2; do
  "$PY" e1_frozen_baselines.py modern --stage E4 --windows-dir "$E0/windows" --neg-dir "$E0/neg" \
    --emb-dir "$E0/emb" --positives-audit "$E0/positives_audit.tsv" \
    --split-clstr "$E0/splits/clusters.id90.clstr" --exps-file "$E0/pilot_exps.txt" \
    --folds 0 --tracks neg1c,neg2 --seed 0 --loss nnpu --pi "$PI" \
    --out "$E4/nnpu_pi${PI}_s0"
done
