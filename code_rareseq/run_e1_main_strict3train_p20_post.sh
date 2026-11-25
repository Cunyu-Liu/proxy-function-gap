#!/bin/bash
set -euo pipefail
PY=/root/miniconda3/envs/llr_env/bin/python
export PYTHONPATH=/root/rareseq/pydeps
W=/root/rareseq/work
E0=/root/rareseq/data/e0
ROOT=/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/e1_main_strict3train_p20_v2
LOGDIR=/home/ProxyFunctionGap/logs
LOG=$LOGDIR/experiments.jsonl
STATUS=$LOGDIR/e1_main_strict3train_p20_driver.status
L=$LOGDIR/e1_main_strict3train_p20_post.log
DATA_VERSION=e0-v2-id90-strict3train-main
cd "$W"
for i in $(seq 1 2880); do
  grep -q ALL_DONE "$STATUS" 2>/dev/null && break
  if grep -q FAILED "$STATUS" 2>/dev/null; then echo "TRAINING_FAILED"; exit 1; fi
  sleep 60
done
grep -q ALL_DONE "$STATUS"
echo "=== AUDIT $(date) ===" >> "$L"
$PY e1_audit_main.py --root "$ROOT" --windows-dir "$E0/windows" --neg-dir "$E0/neg" \
  --positives-audit "$E0/positives_audit.tsv" --exps-file "$E0/main_exps.txt" --log "$LOG" \
  --out "$ROOT/audit" >> "$L" 2>&1
echo "=== COMBINE $(date) ===" >> "$L"
$PY e1_combine_cv_predictions.py --e1-root "$ROOT" --prefix "" --out "$ROOT/cv_combined" \
  --folds 0,1,2,3,4 --seeds 0,1,2 --log "$LOG" --data-version "$DATA_VERSION" >> "$L" 2>&1
echo "=== CV_SUMMARY $(date) ===" >> "$L"
$PY e1_cv_summary.py --e1-root "$ROOT" --prefix "" --out "$ROOT/cv_summary" \
  --folds 0,1,2,3,4 --seeds 0,1,2 --log "$LOG" --data-version "$DATA_VERSION" >> "$L" 2>&1
echo "=== BOOTSTRAP $(date) ===" >> "$L"
$PY e1_paired_bootstrap.py \
  --modern 0="$ROOT/cv_combined/modern_bce_s0.tsv.gz" \
  --modern 1="$ROOT/cv_combined/modern_bce_s1.tsv.gz" \
  --modern 2="$ROOT/cv_combined/modern_bce_s2.tsv.gz" \
  --reference "$ROOT/cv_combined/knn.tsv.gz" \
  --out "$ROOT/cv_bootstrap" --tiers tail,mid,head --tracks neg1c,neg2 \
  --n-boot 10000 --jobs 16 --seed 20261123 --resume --log "$LOG" --data-version "$DATA_VERSION" >> "$L" 2>&1
echo "=== STRATIFY $(date) ===" >> "$L"
$PY e1_stratify.py \
  --run knn="$ROOT/cv_combined/knn.tsv.gz" \
  --run modern_s0="$ROOT/cv_combined/modern_bce_s0.tsv.gz" \
  --run modern_s1="$ROOT/cv_combined/modern_bce_s1.tsv.gz" \
  --run modern_s2="$ROOT/cv_combined/modern_bce_s2.tsv.gz" \
  --identity "$E0/identity/per_window_identity.tsv" --out "$ROOT/cv_stratified" \
  --log "$LOG" --data-version "$DATA_VERSION" >> "$L" 2>&1
echo "=== VERIFY $(date) ===" >> "$L"
$PY e1_verify_main_results.py --root "$ROOT" --identity "$E0/identity/per_window_identity.tsv" \
  --log "$LOG" --out "$ROOT/verification" >> "$L" 2>&1
echo "E1_STRICT3TRAIN_ANALYSIS_DONE $(date)" >> "$L"
