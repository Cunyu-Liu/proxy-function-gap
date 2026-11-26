#!/bin/bash
set -euo pipefail
ROOT=/home/ProxyFunctionGap
DATA=/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/external_benchmarks/rnacompete
VERSION=rnacompete-ray2013-v1.0-id90-strict3train
export PYTHONHASHSEED=0
CODE_COMMIT=${CODE_COMMIT:-$(git -C "$ROOT" rev-parse HEAD)}
cd "$ROOT"
python3 code_rareseq/rnacompete_ray2013_contract_test.py
python3 code_rareseq/rnacompete_ray2013_freeze.py \
  --input-root "$DATA/ray2013_nature12311" \
  --out "$DATA/$VERSION" \
  --contract "$ROOT/预注册/RNAcompete_Ray2013_预注册数据合同_v1.0_2026-11-26.md" \
  --code-commit "$CODE_COMMIT"
python3 code_rareseq/rnacompete_ray2013_freeze.py --out "$DATA/$VERSION" --verify-only
