#!/bin/bash
set -uo pipefail
C=/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/e1_rnafm_finetune_strict3train_p20/loader_cache
L=/home/ProxyFunctionGap/logs/e1_finetune_loader_cache_verify.log
cd /root/rareseq/work
for i in $(seq 1 2880); do
  n=$(ls "$C"/tasks_*.pkl 2>/dev/null | wc -l)
  [ "$n" -ge 5 ] && break
  sleep 60
done
n=$(ls "$C"/tasks_*.pkl 2>/dev/null | wc -l)
if [ "$n" -lt 5 ]; then echo "CACHE_TIMEOUT n=$n $(date -Is)" >> "$L"; exit 1; fi
echo "CACHE_READY n=$n $(date -Is)" >> "$L"
PYTHONPATH=/root/rareseq/pydeps:/root/rnaft/pypath /root/miniconda3/envs/llr_env/bin/python verify_loader_cache.py >> "$L" 2>&1
echo "VERIFY_EXIT rc=$? $(date -Is)" >> "$L"