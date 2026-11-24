#!/usr/bin/env bash
set -euo pipefail

CODE_HOME=${CODE_HOME:-/home/cunyuliu/proxy-function-gap}
DATA_ROOT=${DATA_ROOT:-/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap}
GPUS=${GPUS:-0,1,2,3}
EPOCHS=${MAX_EPOCHS:-1000}
IFS=, read -ra DEVICES <<< "$GPUS"

if [[ $# -ne 1 || $1 != m1a ]]; then
  echo "Usage: GPUS=0,1,2,3 $0 m1a" >&2
  exit 2
fi
for required in WINDOWS_DIR NEG_DIR EMB_DIR POSITIVES_AUDIT SPLIT_CLSTR EXPS_FILE; do
  if [[ -z ${!required:-} ]]; then
    echo "Missing environment variable: $required" >&2
    exit 2
  fi
done

command -v nvidia-smi >/dev/null
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
mkdir -p "$DATA_ROOT/m1a" "$CODE_HOME/logs"
index=0
for arch in independent shared; do
  for seed in 0 1 2; do
    gpu=${DEVICES[$((index % ${#DEVICES[@]}))]}
    out="$DATA_ROOT/m1a/${arch}_s${seed}"
    log="$CODE_HOME/logs/m1a_${arch}_s${seed}.log"
    if [[ -e $out ]]; then
      echo "Refusing to overwrite existing $out" >&2
      exit 1
    fi
    CUDA_VISIBLE_DEVICES=$gpu python3 "$CODE_HOME/code_rareseq/m1_function_heads.py" \
      --arch "$arch" --seed "$seed" --fold 0 --tracks neg1c,neg2 \
      --windows-dir "$WINDOWS_DIR" --neg-dir "$NEG_DIR" --emb-dir "$EMB_DIR" \
      --positives-audit "$POSITIVES_AUDIT" --split-clstr "$SPLIT_CLSTR" --exps-file "$EXPS_FILE" \
      --out "$out" --max-epochs "$EPOCHS" --log "$CODE_HOME/logs/experiments.jsonl" \
      >"$log" 2>&1 &
    echo "Submitted arch=$arch seed=$seed gpu=$gpu pid=$! out=$out"
    index=$((index + 1))
  done
done
wait
