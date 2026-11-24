#!/usr/bin/env bash
# E0 续跑：步骤3（跳过已完成预计算）→ 4-9
set -euo pipefail
W=/root/rareseq/work
E=/root/rareseq/data/e0
R=/root/rareseq/data/bbad307_repo/Benchmark-RBP
GENOME=/root/rareseq/data/ref/genome/hg38.fa
PYC="/root/miniconda3/envs/codonflow/bin/python"
PYL="/root/miniconda3/envs/llr_env/bin/python"
export PYTHONPATH=/root/rareseq/pydeps
cd "$W"

echo "=== [3/9] negatives RESUME (skip-precompute) $(date -u +%H:%M) ==="
$PYC e0_negatives.py --repo "$R" --genome "$GENOME" --folds-dir "$E/splits" \
  --windows-dir "$E/windows" --out "$E/neg" --workers 10 --skip-precompute

echo "=== [4/9] embeddings rnafm (main) $(date -u +%H:%M) ==="
HF_HOME=/root/hf_home HF_HUB_OFFLINE=1 PYTHONPATH=/root/rareseq/pydeps:/root/rnaft/pypath \
  $PYL e0_g0_models.py embed --windows-dir "$E/windows" --neg-dir "$E/neg" \
  --out "$E/emb" --model rnafm --batch 256 --exps-file "$E/main_exps.txt"

echo "=== [5/9] OverfitNN (main, 5 folds, 3 tracks) $(date -u +%H:%M) ==="
HF_HOME=/root/hf_home HF_HUB_OFFLINE=1 PYTHONPATH=/root/rareseq/pydeps:/root/rnaft/pypath \
  $PYL e0_g0_models.py overfitnn --windows-dir "$E/windows" --neg-dir "$E/neg" \
  --emb-dir "$E/emb" --out "$E/g0" --model rnafm --exps-file "$E/main_exps.txt" \
  --folds 5 --tracks neg1c,neg2,neg1f

echo "=== [6/9] probes (pilot 5-fold + main fold0) $(date -u +%H:%M) ==="
$PYC e0_g0_models.py probe --windows-dir "$E/windows" --neg-dir "$E/neg" \
  --emb-dir "$E/emb" --out "$E/g0" --model rnafm --exps-file "$E/pilot_exps.txt" \
  --folds 5 --tracks neg1c,neg2
$PYC e0_g0_models.py probe --windows-dir "$E/windows" --neg-dir "$E/neg" \
  --emb-dir "$E/emb" --out "$E/g0_mainfold0" --model rnafm --exps-file "$E/main_exps.txt" \
  --folds 1 --tracks neg1c,neg2

echo "=== [7/9] leakage quant (main) $(date -u +%H:%M) ==="
HF_HOME=/root/hf_home HF_HUB_OFFLINE=1 PYTHONPATH=/root/rareseq/pydeps:/root/rnaft/pypath \
  $PYL e0_leakage.py --windows-dir "$E/windows" --neg-dir "$E/neg" --folds-dir "$E/splits" \
  --emb-dir "$E/emb" --model rnafm --out "$E/leakage" --exps-file "$E/main_exps.txt" \
  --tracks neg1c,neg2

echo "=== [8/9] per-window d_hom identity (main) $(date -u +%H:%M) ==="
HF_HOME=/root/hf_home HF_HUB_OFFLINE=1 PYTHONPATH=/root/rareseq/pydeps:/root/rnaft/pypath \
  $PYL e0_window_identity.py --windows-dir "$E/windows" --neg-dir "$E/neg" \
  --emb-dir "$E/emb" --model rnafm --out "$E/identity" --exps-file "$E/main_exps.txt" \
  || echo "STEP8_WINDOW_IDENTITY_FAILED"

echo "=== [9/9] RBP family freeze $(date -u +%H:%M) ==="
$PYC e0_rbp_families.py --windows-dir "$E/windows" --out "$E/families" \
  --exps-file "$E/main_exps.txt" || echo "STEP9_FAMILY_FAILED"

echo "=== E0 RESUME CHAIN DONE $(date -u +%H:%M) ==="