#!/bin/bash
set -euo pipefail
PY=/root/miniconda3/envs/llr_env/bin/python
export PYTHONPATH=/root/rareseq/pydeps
W=/root/rareseq/work
E1=/root/rareseq/data/e1
ROOT=/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/e5
LOG=/home/ProxyFunctionGap/logs/experiments.jsonl
cd "$W"
"$PY" e5_low_hom_multiseed.py \
  --modern 0="$E1/cv_all_predictions/modern_bce_s0.tsv.gz" \
  --modern 1="$E1/cv_all_predictions/modern_bce_s1.tsv.gz" \
  --modern 2="$E1/cv_all_predictions/modern_bce_s2.tsv.gz" \
  --knn "$E1/cv_all_predictions/knn.tsv.gz" \
  --identity /root/rareseq/data/e0/identity/per_window_identity.tsv \
  --out "$ROOT/low_hom_multiseed" --low-hom-max 0.60 --expected-rows 2419417 --log "$LOG"
