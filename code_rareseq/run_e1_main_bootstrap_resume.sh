#!/bin/bash
set -euo pipefail
PY=/root/miniconda3/envs/llr_env/bin/python
export PYTHONPATH=/root/rareseq/pydeps
ROOT=/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/e1_main
LOG=/home/ProxyFunctionGap/logs/experiments.jsonl
cd /root/rareseq/work
$PY e1_paired_bootstrap.py \
  --modern 0="$ROOT/cv_combined/modern_bce_s0.tsv.gz" \
  --modern 1="$ROOT/cv_combined/modern_bce_s1.tsv.gz" \
  --modern 2="$ROOT/cv_combined/modern_bce_s2.tsv.gz" \
  --reference "$ROOT/cv_combined/knn.tsv.gz" \
  --out "$ROOT/cv_bootstrap" --tiers tail,mid,head --tracks neg1c,neg2 \
  --n-boot 10000 --jobs 16 --seed 20261123 --resume \
  --log "$LOG" --data-version e0-v1-id90-frozen-main
echo "E1_SUMMARY_DONE $(date) resumed-bootstrap"
