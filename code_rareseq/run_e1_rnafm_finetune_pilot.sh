#!/bin/bash
set -euo pipefail
cd /root/rareseq/work
PY=/root/miniconda3/envs/llr_env/bin/python
export PYTHONPATH=/root/rareseq/pydeps:/root/rnaft/pypath
export HF_HOME=/root/hf_home
export HF_HUB_OFFLINE=1
E0=/root/rareseq/data/e0
E1=/root/rareseq/data/e1
COMMON=(--windows-dir "$E0/windows" --neg-dir "$E0/neg"
  --positives-audit "$E0/positives_audit.tsv" --split-clstr "$E0/splits/clusters.id90.clstr"
  --exps-file "$E0/pilot_exps.txt" --folds 0 --tracks neg1c,neg2 --epochs 3)
for MODE in lora full; do
  EXTRA=(--encoder-lr 3e-4)
  if [[ "$MODE" == full ]]; then EXTRA=(--encoder-lr 1e-5); fi
  "$PY" e1_finetune.py "${COMMON[@]}" --mode "$MODE" "${EXTRA[@]}" --batch 256 --seed 0 \
    --out "$E1/pilot_f0_${MODE}_s0"
done
