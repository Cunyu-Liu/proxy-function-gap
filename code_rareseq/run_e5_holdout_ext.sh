#!/bin/bash
set -u
PY=/root/miniconda3/envs/llr_env/bin/python
export PYTHONPATH=/root/rareseq/pydeps
W=/root/rareseq/work
E0=/root/rareseq/data/e0
OUT=/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/e5
LOG=/home/ProxyFunctionGap/logs/experiments.jsonl
mkdir -p $OUT
cd $W
HOLDOUTS=HNRNPC,PTBP1,U2AF2,SRSF1,RBFOX2,QKI,IGF2BP1,LIN28B,TAF15,UPF1,CSTF2T,YBX3
COMMON=(--windows-dir "$E0/windows" --neg-dir "$E0/neg" --emb-dir "$E0/emb"
  --positives-audit "$E0/positives_audit.tsv" --split-clstr "$E0/splits/clusters.id90.clstr"
  --exps-file "$E0/main_exps.txt" --holdouts "$HOLDOUTS" --folds 0 --tracks neg1c,neg2 --log "$LOG")
for S in 0 1 2; do
  $PY e5_rbp_holdout.py "${COMMON[@]}" --seed $S --out "$OUT/holdout_ext_s${S}" \
    > /home/ProxyFunctionGap/logs/e5_holdout_ext_s${S}.log 2>&1
done
echo E5_EXT_DONE > /home/ProxyFunctionGap/logs/e5_holdout_ext.status