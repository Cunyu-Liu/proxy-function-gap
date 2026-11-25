#!/bin/bash
set -euo pipefail
PY=/root/miniconda3/envs/llr_env/bin/python
export PYTHONPATH=/root/rareseq/pydeps
W=/root/rareseq/work
ROOT=/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/e1_main
LOG=/home/ProxyFunctionGap/logs/experiments.jsonl
L=/home/ProxyFunctionGap/logs/e1_main_analysis.log
cd "$W"
for i in $(seq 1 360); do
  grep -q E1_SUMMARY_DONE /home/ProxyFunctionGap/logs/e1_main_summary.log 2>/dev/null && break
  if test -f /home/ProxyFunctionGap/logs/e1_main_summary.blocked; then
    echo SUMMARY_BLOCKED
    exit 1
  fi
  sleep 60
done
grep -q E1_SUMMARY_DONE /home/ProxyFunctionGap/logs/e1_main_summary.log
test -s "$ROOT/cv_bootstrap/paired_bootstrap.tsv"
echo "=== STRATIFY $(date) ===" >> "$L"
$PY e1_stratify.py \
  --run knn="$ROOT/cv_combined/knn.tsv.gz" \
  --run modern_s0="$ROOT/cv_combined/modern_bce_s0.tsv.gz" \
  --run modern_s1="$ROOT/cv_combined/modern_bce_s1.tsv.gz" \
  --run modern_s2="$ROOT/cv_combined/modern_bce_s2.tsv.gz" \
  --identity /root/rareseq/data/e0/identity/per_window_identity.tsv \
  --out "$ROOT/cv_stratified" --log "$LOG" --data-version e0-v1-id90-frozen-main >> "$L" 2>&1
echo "=== VERIFY $(date) ===" >> "$L"
$PY e1_verify_main_results.py --root "$ROOT" --log "$LOG" --out "$ROOT/verification" >> "$L" 2>&1
echo "E1_ANALYSIS_DONE $(date)" >> "$L"
