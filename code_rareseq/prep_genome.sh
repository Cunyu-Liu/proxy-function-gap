#!/usr/bin/env bash
# RARE-Seq E0 — 基因组准备：解压 → chr 命名对齐 GENCODE → faidx。
# 用法: bash prep_genome.sh <tar.gz 或 fa.gz 或 fa> <输出目录>
set -euo pipefail
SRC="$1"
OUT="$2"
mkdir -p "$OUT"
PY=/root/miniconda3/envs/codonflow/bin/python
export PYTHONPATH=/root/rareseq/pydeps

WORK="$OUT/_extract"
mkdir -p "$WORK"
case "$SRC" in
  *.tar.gz) tar xzf "$SRC" -C "$WORK" ;;
  *.fa.gz|*.fasta.gz|*.fna.gz) gunzip -c "$SRC" > "$WORK/genome.fa" ;;
  *.fa|*.fasta|*.fna) cp "$SRC" "$WORK/genome.fa" ;;
  *) echo "unknown format: $SRC"; exit 1 ;;
esac
FA=$(find "$WORK" -type f \( -name "*.fa" -o -name "*.fasta" -o -name "*.fna" -o -name "*.dna*" \) | head -1)
[ -n "$FA" ] || { echo "no fasta found in $SRC"; exit 1; }
echo "extracted: $FA ($(du -h "$FA" | cut -f1))"
FIRST=$(head -1 "$FA")
echo "first header: $FIRST"
FINAL="$OUT/hg38.fa"
if [[ "$FIRST" == ">chr"* ]]; then
  mv "$FA" "$FINAL"
else
  echo "renaming Ensembl chroms -> chr prefix (MT->chrM)"
  sed -E 's/^>MT([ \t]|$)/>chrM\1/; s/^>([0-9XY]+)([ \t]|$)/>chr\1\2/' "$FA" > "$FINAL"
  rm -f "$FA"
fi
rmdir "$WORK" 2>/dev/null || true
$PY -c "import pysam; pysam.faidx('$FINAL')"
echo "--- chrom sizes (first 25) ---"
cut -f1,2 "$FINAL.fai" | head -25
echo "PREP_DONE $FINAL"