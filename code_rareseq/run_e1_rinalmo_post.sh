#!/bin/bash
set -euo pipefail
PY=/root/miniconda3/envs/llr_env/bin/python
export PYTHONPATH=/root/rareseq/pydeps
W=/root/rareseq/work
E0=/root/rareseq/data/e0
ROOT=/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/e1_rinalmo_main_strict3train_p20_v1
LOGDIR=/home/ProxyFunctionGap/logs/e1_rinalmo_main
LOG=$LOGDIR/experiments.jsonl
STATUS=$LOGDIR/driver.status
L=$LOGDIR/post.log
DATA_VERSION=e0-v2-id90-strict3train-main
cd "$W"
for i in $(seq 1 2880); do
  grep -q ALL_DONE "$STATUS" 2>/dev/null && break
  if grep -q FAILED "$STATUS" 2>/dev/null; then echo "DRIVER_FAILED"; exit 1; fi
  sleep 60
done
grep -q ALL_DONE "$STATUS"
echo "=== BOOTSTRAP $(date) ===" >> "$L"
$PY e1_paired_bootstrap.py \
  --modern 0="$ROOT/main/cv_combined/modern_bce_s0.tsv.gz" \
  --modern 1="$ROOT/main/cv_combined/modern_bce_s1.tsv.gz" \
  --modern 2="$ROOT/main/cv_combined/modern_bce_s2.tsv.gz" \
  --reference "$ROOT/main/cv_combined/knn.tsv.gz" \
  --out "$ROOT/main/cv_bootstrap" --tiers tail,mid,head --tracks neg1c,neg2 \
  --n-boot 10000 --jobs 8 --seed 20261123 --resume --log "$LOG" --data-version "$DATA_VERSION" >> "$L" 2>&1
echo "=== STRATIFY $(date) ===" >> "$L"
$PY e1_stratify.py \
  --run knn="$ROOT/main/cv_combined/knn.tsv.gz" \
  --run modern_s0="$ROOT/main/cv_combined/modern_bce_s0.tsv.gz" \
  --run modern_s1="$ROOT/main/cv_combined/modern_bce_s1.tsv.gz" \
  --run modern_s2="$ROOT/main/cv_combined/modern_bce_s2.tsv.gz" \
  --identity "$E0/identity/per_window_identity.tsv" --out "$ROOT/main/cv_stratified" \
  --log "$LOG" --data-version "$DATA_VERSION" >> "$L" 2>&1
echo "=== VERIFY $(date) ===" >> "$L"
$PY e1_verify_main_results.py --root "$ROOT/main" --identity "$E0/identity/per_window_identity.tsv" \
  --log "$LOG" --out "$ROOT/main/verification" >> "$L" 2>&1
echo "E1_RINALMO_ANALYSIS_DONE $(date)" >> "$L"
