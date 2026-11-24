#!/bin/bash
set -euo pipefail
cd /root/rareseq/work
PY=/root/miniconda3/envs/llr_env/bin/python
E1=/root/rareseq/data/e1
"$PY" e1_combine_cv_predictions.py --e1-root "$E1" --out "$E1/cv_all_predictions"
"$PY" e1_cv_summary.py --e1-root "$E1" --out "$E1/cv_all_summary"
BOOT=(--reference "$E1/cv_all_predictions/knn.tsv.gz" --out "$E1/cv_all_bootstrap" --n-boot 10000 --jobs 16)
for SEED in 0 1 2; do
  BOOT+=(--modern "$SEED=$E1/cv_all_predictions/modern_bce_s${SEED}.tsv.gz")
done
"$PY" e1_paired_bootstrap.py "${BOOT[@]}"
"$PY" e1_stratify.py --run "bce=$E1/cv_all_predictions/modern_bce_s0.tsv.gz" \
  --run "knn=$E1/cv_all_predictions/knn.tsv.gz" \
  --identity /root/rareseq/data/e0/identity/per_window_identity.tsv \
  --out "$E1/cv_all_stratified_s0" --log /root/rareseq/logs/experiments.jsonl
