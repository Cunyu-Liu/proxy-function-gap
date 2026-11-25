#!/bin/bash
# E1 主集五折三种子汇总：合并预测 → CV 分层 → 配对 bootstrap。等训练 ALL_DONE 后运行。
set -euo pipefail
PY=/root/miniconda3/envs/llr_env/bin/python
export PYTHONPATH=/root/rareseq/pydeps
W=/root/rareseq/work
ROOT=/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/e1_main
LOG=/home/ProxyFunctionGap/logs/experiments.jsonl
L=/home/ProxyFunctionGap/logs/e1_main_summary.log
cd "$W"
for i in $(seq 1 360); do
  grep -q ALL_DONE /home/ProxyFunctionGap/logs/e1_main_driver.status 2>/dev/null && break
  sleep 60
done
grep -q ALL_DONE /home/ProxyFunctionGap/logs/e1_main_driver.status
echo "=== COMBINE $(date) ===" >> "$L"
$PY e1_combine_cv_predictions.py --e1-root "$ROOT" --prefix "" --out "$ROOT/cv_combined" \
  --folds 0,1,2,3,4 --seeds 0,1,2 --log "$LOG" --data-version e0-v1-id90-frozen-main >> "$L" 2>&1
echo "=== CV_SUMMARY $(date) ===" >> "$L"
$PY e1_cv_summary.py --e1-root "$ROOT" --prefix "" --out "$ROOT/cv_summary" \
  --folds 0,1,2,3,4 --seeds 0,1,2 --log "$LOG" --data-version e0-v1-id90-frozen-main >> "$L" 2>&1
echo "=== BOOTSTRAP $(date) ===" >> "$L"
$PY e1_paired_bootstrap.py \
  --modern 0="$ROOT/cv_combined/modern_bce_s0.tsv.gz" \
  --modern 1="$ROOT/cv_combined/modern_bce_s1.tsv.gz" \
  --modern 2="$ROOT/cv_combined/modern_bce_s2.tsv.gz" \
  --reference "$ROOT/cv_combined/knn.tsv.gz" \
  --out "$ROOT/cv_bootstrap" --tiers tail,mid,head --n-boot 10000 --jobs 8 \
  --log "$LOG" --data-version e0-v1-id90-frozen-main >> "$L" 2>&1
echo "E1_SUMMARY_DONE $(date)" >> "$L"