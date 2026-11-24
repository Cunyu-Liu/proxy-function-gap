#!/bin/bash
set -euo pipefail
cd /root/rareseq/work
PY=/root/miniconda3/envs/llr_env/bin/python
E1=/root/rareseq/data/e1
RESULTS=(
  --result "rnafm_bce=$E1/pilot_f0_modern_bce_s0/modern_results.tsv"
  --result "rnafm_knn=$E1/pilot_f0_knn/knn_results.tsv"
  --result "rnafm_lora=$E1/pilot_f0_lora_s0/finetune_results.tsv"
  --result "rnafm_full=$E1/pilot_f0_full_s0/finetune_results.tsv"
  --result "rinalmo_bce=$E1/pilot_f0_rinalmo_bce_s0/modern_results.tsv"
  --result "rinalmo_knn=$E1/pilot_f0_rinalmo_knn/knn_results.tsv"
)
PREDICTIONS=(
  --prediction "rnafm_bce=$E1/pilot_f0_modern_bce_s0/modern_test_predictions.tsv.gz"
  --prediction "rnafm_knn=$E1/pilot_f0_knn/knn_test_predictions.tsv.gz"
  --prediction "rnafm_lora=$E1/pilot_f0_lora_s0/finetune_test_predictions.tsv.gz"
  --prediction "rnafm_full=$E1/pilot_f0_full_s0/finetune_test_predictions.tsv.gz"
  --prediction "rinalmo_bce=$E1/pilot_f0_rinalmo_bce_s0/modern_test_predictions.tsv.gz"
  --prediction "rinalmo_knn=$E1/pilot_f0_rinalmo_knn/knn_test_predictions.tsv.gz"
)
"$PY" e1_discrete_router.py "${RESULTS[@]}" "${PREDICTIONS[@]}" --out "$E1/pilot_f0_discrete_router"
"$PY" e1_mode_compare.py --run "frozen=$E1/pilot_f0_modern_bce_s0/modern_results.tsv" \
  --run "full=$E1/pilot_f0_full_s0/finetune_results.tsv" \
  --run "router=$E1/pilot_f0_discrete_router/router_results.tsv" \
  --reference frozen --out "$E1/pilot_f0_router_comparison"
