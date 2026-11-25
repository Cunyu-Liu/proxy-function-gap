#!/bin/bash
set -euo pipefail
PY=/root/miniconda3/envs/codonflow/bin/python
export PYTHONPATH=/root/rareseq/pydeps
CODE=/home/ProxyFunctionGap/code_rareseq
E0=/root/rareseq/data/e0
P=/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/parclip
BASE=/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/e5/parclip_cross_protocol_contract_p20
LOGROOT=/home/ProxyFunctionGap/logs
mkdir -p "$BASE" "$LOGROOT"
cd "$CODE"
"$PY" -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), flush=True)"
for SEED in 0 1 2; do
  "$PY" e5_parclip_eval.py \
    --e0-dir "$E0" \
    --parclip-emb-dir "$P/emb_v2/full" \
    --parclip-windows-dir "$P/processed.full_windows_v2" \
    --out "$BASE/seed$SEED" \
    --seed "$SEED" \
    --epochs 1000 \
    --patience 20 \
    --min-delta 1e-4 \
    --device cuda:0 \
    --parclip-neg2 \
    --log "$LOGROOT/experiments.jsonl" \
    > "$LOGROOT/parclip_cross_protocol_contract_p20_s$SEED.log" 2>&1
  "$PY" -c "import pandas as pd; p='$BASE/seed$SEED/cross_protocol_results.tsv'; d=pd.read_csv(p, sep='\t'); assert len(d)==27; print(d.groupby('eval_protocol').auroc.mean())"
done
echo "PARCLIP_CONTRACT_P20_ALL_DONE $(date -u +%FT%TZ)"
