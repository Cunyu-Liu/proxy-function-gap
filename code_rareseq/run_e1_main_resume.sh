#!/bin/bash
# E1 主集断点续跑：跳过已完成 run，2 路并行 modern，再 knn，写 ALL_DONE 触发汇总。
set -u
PY=/root/miniconda3/envs/llr_env/bin/python
export PYTHONPATH=/root/rareseq/pydeps
W=/root/rareseq/work
E0=/root/rareseq/data/e0
OUT=/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/e1_main
LOG=/home/ProxyFunctionGap/logs/experiments.jsonl
mkdir -p $OUT
cd $W
COMMON=(--windows-dir "$E0/windows" --neg-dir "$E0/neg" --emb-dir "$E0/emb"
  --positives-audit "$E0/positives_audit.tsv" --split-clstr "$E0/splits/clusters.id90.clstr"
  --exps-file "$E0/main_exps.txt" --tracks neg1c,neg2 --epochs 30 --log "$LOG")
run(){ local F=$1 S=$2
  if [ -f "$OUT/f${F}_modern_bce_s${S}/modern_results.tsv" ]; then return; fi
  "$PY" e1_frozen_baselines.py modern "${COMMON[@]}" --folds "$F" --seed "$S" --loss bce \
    --out "$OUT/f${F}_modern_bce_s${S}" > /home/ProxyFunctionGap/logs/e1_main_f${F}_s${S}.log 2>&1
}
for F in 0 1 2 3 4; do
  for S in 0 1 2; do
    while [ "$(jobs -rp | wc -l)" -ge 2 ]; do sleep 20; done
    run "$F" "$S" &
  done
done
wait
for F in 0 1 2 3 4; do
  if [ -f "$OUT/f${F}_knn/knn_results.tsv" ]; then continue; fi
  "$PY" e1_frozen_baselines.py knn "${COMMON[@]}" --folds "$F" --seed 0 \
    --out "$OUT/f${F}_knn" > /home/ProxyFunctionGap/logs/e1_main_f${F}_knn.log 2>&1
done
echo ALL_DONE > /home/ProxyFunctionGap/logs/e1_main_driver.status