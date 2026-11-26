#!/bin/bash
set -uo pipefail
LOGDIR=/home/ProxyFunctionGap/logs
PILOT=$LOGDIR/e1_finetune_pilot.status
CHAIN=$LOGDIR/e1_finetune_chain.status
LOCK=$LOGDIR/e1_finetune_chain.lock
if ! mkdir "$LOCK" 2>/dev/null; then echo "CHAIN_ALREADY_RUNNING"; exit 0; fi
cd /root/rareseq/work || exit 1
echo "WAITING_PILOT $(date -Is)" > "$CHAIN"
for ATTEMPT in $(seq 1 72); do
  if grep -q '^PILOT_PASSED' "$PILOT" 2>/dev/null; then
    echo "LAUNCHING_MAIN $(date -Is)" > "$CHAIN"
    setsid -f nohup bash run_e1_finetune_strict3train_p20.sh \
      > "$LOGDIR/e1_finetune_main_driver.log" 2>&1 < /dev/null
    echo "MAIN_LAUNCHED $(date -Is)" > "$CHAIN"
    exit 0
  fi
  if grep -qE '^(FAILED|VERIFY_FAILED|RESOURCE_BLOCKED)' "$PILOT" 2>/dev/null; then
    echo "PILOT_FAILED $(date -Is)" > "$CHAIN"
    exit 1
  fi
  sleep 300
done
echo "PILOT_WAIT_TIMEOUT $(date -Is)" > "$CHAIN"
exit 1
