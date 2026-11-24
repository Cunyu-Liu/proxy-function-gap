#!/bin/bash
set -euo pipefail
cd /root/rareseq/work
PY=/root/miniconda3/envs/llr_env/bin/python
export PYTHONPATH=/root/rareseq/pydeps
E0=/root/rareseq/data/e0
E1=/root/rareseq/data/e1
COMMON=(--windows-dir "$E0/windows" --neg-dir "$E0/neg" --emb-dir "$E1/emb_rinalmo"
  --positives-audit "$E0/positives_audit.tsv" --split-clstr "$E0/splits/clusters.id90.clstr"
  --exps-file "$E0/pilot_exps.txt" --folds 0 --tracks neg1c,neg2 --seed 0 --model rinalmo-giga)
"$PY" e1_frozen_baselines.py modern "${COMMON[@]}" --loss bce --out "$E1/pilot_f0_rinalmo_bce_s0"
"$PY" e1_frozen_baselines.py knn "${COMMON[@]}" --out "$E1/pilot_f0_rinalmo_knn"
