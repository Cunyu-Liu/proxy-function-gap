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
  --exps-file "$E0/e1_smoke_exp.txt" --folds 0 --tracks neg1c,neg2 --seed 0 --epochs 1)
"$PY" e1_finetune.py "${COMMON[@]}" --mode lora --batch 256 --out "$E1/smoke_lora"
"$PY" e1_finetune.py "${COMMON[@]}" --mode full --batch 256 --encoder-lr 1e-5 --out "$E1/smoke_full"
