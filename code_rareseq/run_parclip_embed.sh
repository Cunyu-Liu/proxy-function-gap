#!/bin/bash
PY=/root/miniconda3/envs/codonflow/bin/python
export PYTHONPATH=/root/rareseq/pydeps
export HF_HOME=/root/hf_home
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
W=/root/rareseq/work
P=/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/parclip
LOG=/home/ProxyFunctionGap/logs/experiments.jsonl
L=/home/ProxyFunctionGap/logs/parclip_embed.log
cd $W
$PY -c "import multimolecule,torch; assert torch.cuda.is_available(); print('MM+CUDA ready',flush=True)" > $L 2>&1 || { echo CUDA_MM_FAIL >> $L; exit 1; }
echo "=== EMBED full $(date) ===" >> $L
$PY e0_g0_models.py embed --windows-dir $P/processed.full_windows_v2 --out $P/emb_v2/full --model rnafm --batch 512 --stage E0 --data-version parclip-hg38-v2 --log $LOG >> $L 2>&1
echo "=== EMBED processed $(date) ===" >> $L
$PY e0_g0_models.py embed --windows-dir $P/processed_windows_v2 --out $P/emb_v2/processed --model rnafm --batch 512 --stage E0 --data-version parclip-hg38-v2 --log $LOG >> $L 2>&1
echo "EMBED_DONE $(date)" >> $L