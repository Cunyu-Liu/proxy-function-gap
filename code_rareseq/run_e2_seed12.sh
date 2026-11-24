#!/bin/bash
set -euo pipefail
cd /root/rareseq/work
PY=/root/miniconda3/envs/llr_env/bin/python
export PYTHONPATH=/root/rareseq/pydeps
E0=/root/rareseq/data/e0
E2=/root/rareseq/data/e2
for SEED in 1 2; do
  COMMON=(--stage E2 --windows-dir "$E0/windows" --neg-dir "$E0/neg" --emb-dir "$E0/emb"
    --positives-audit "$E0/positives_audit.tsv" --split-clstr "$E0/splits/clusters.id90.clstr"
    --exps-file "$E0/pilot_exps.txt" --folds 0 --tracks neg1c,neg2 --seed "$SEED")
  "$PY" e1_frozen_baselines.py modern "${COMMON[@]}" --loss bce --out "$E2/frozen_bce_s${SEED}"
  "$PY" e1_frozen_baselines.py knn "${COMMON[@]}" --out "$E2/frozen_knn_s${SEED}"
  "$PY" e2_evidence_router.py \
    --val-bce "$E2/frozen_bce_s${SEED}/modern_val_predictions.tsv.gz" \
    --val-knn "$E2/frozen_knn_s${SEED}/knn_val_predictions.tsv.gz" \
    --test-bce "$E2/frozen_bce_s${SEED}/modern_test_predictions.tsv.gz" \
    --test-knn "$E2/frozen_knn_s${SEED}/knn_test_predictions.tsv.gz" \
    --identity "$E0/identity/per_window_identity.tsv" --out "$E2/fold0_router_calibrated_s${SEED}"
done
