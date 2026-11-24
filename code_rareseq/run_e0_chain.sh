#!/usr/bin/env bash
# RARE-Seq E0 执行链（基因组就绪后启动）: 窗口 → 聚类划分 → 负例 → 嵌入 → OverfitNN/探针/泄漏
set -euo pipefail
W=/root/rareseq/work
E=/root/rareseq/data/e0
R=/root/rareseq/data/bbad307_repo/Benchmark-RBP
G=/root/rareseq/data/ref/gencode_v42_parsed
GENOME=${1:-/root/rareseq/data/ref/genome/hg38.fa}
PYC="/root/miniconda3/envs/codonflow/bin/python"
PYL="/root/miniconda3/envs/llr_env/bin/python"
export PYTHONPATH=/root/rareseq/pydeps
cd "$W"

echo "=== [1/7] positive windows $(date -u +%H:%M) ==="
$PYC e0_windows.py --repo "$R" --genome "$GENOME" --out "$E/windows"

echo "=== [2/7] cluster splits id90 (main 146) $(date -u +%H:%M) ==="
$PYC e0_splits.py --windows-dir "$E/windows" --regions "$G/basic_merged_regions.bed" \
  --regions-tsv "$G/basic_merged_regions.tsv" --out "$E/splits" --identity 0.9 \
  --exps-file "$E/main_exps.txt" --threads 16 --mem-mb 24000

echo "=== [3/7] negatives (3 tracks, all 223) $(date -u +%H:%M) ==="
$PYC e0_negatives.py --repo "$R" --genome "$GENOME" --folds-dir "$E/splits" \
  --windows-dir "$E/windows" --out "$E/neg" --workers 10

echo "=== [4/7] embeddings rnafm (main) $(date -u +%H:%M) ==="
HF_HOME=/root/hf_home HF_HUB_OFFLINE=1 PYTHONPATH=/root/rareseq/pydeps:/root/rnaft/pypath \
  $PYL e0_g0_models.py embed --windows-dir "$E/windows" --neg-dir "$E/neg" \
  --out "$E/emb" --model rnafm --batch 256 --exps-file "$E/main_exps.txt"

echo "=== [5/7] OverfitNN (main, 5 folds, 3 tracks) $(date -u +%H:%M) ==="
HF_HOME=/root/hf_home HF_HUB_OFFLINE=1 PYTHONPATH=/root/rareseq/pydeps:/root/rnaft/pypath \
  $PYL e0_g0_models.py overfitnn --windows-dir "$E/windows" --neg-dir "$E/neg" \
  --emb-dir "$E/emb" --out "$E/g0" --model rnafm --exps-file "$E/main_exps.txt" \
  --folds 5 --tracks neg1c,neg2,neg1f

echo "=== [6/7] probes (pilot 5-fold + main fold0) $(date -u +%H:%M) ==="
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

echo "=== E0 CHAIN DONE $(date -u +%H:%M) ==="