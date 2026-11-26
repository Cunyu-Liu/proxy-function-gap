#!/usr/bin/env bash
set -euo pipefail
PY=/root/miniconda3/envs/codonflow/bin/python
ROOT=/home/ProxyFunctionGap
W=$ROOT/code_rareseq
DATA=/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/external_benchmarks/cafa/cafa3_timeholdout_v1
BASE=/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/cafa3_timeholdout_esmc300m_p20_v1
EMB=$BASE/embeddings
SMOKE=$BASE/smoke
RUNS=$BASE/runs
LOGDIR=$ROOT/logs/cafa3_esmc300m_p20
STATUS=$LOGDIR/driver.status
export PYTHONPATH=/root/rareseq/pydeps
export HF_HOME=/root/hf_home
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
mkdir -p "$EMB" "$SMOKE" "$RUNS" "$LOGDIR"
cd "$W"
exec 9>"$LOGDIR/driver.lock"
if ! flock -n 9; then
  echo "another CAFA3 driver holds the lock" >&2
  exit 0
fi
on_exit() {
  local rc=$?
  if [ "$rc" -ne 0 ]; then echo "FAILED rc=$rc $(date -Is)" > "$STATUS"; fi
}
trap on_exit EXIT
set_status() { echo "$1 $(date -Is)" > "$STATUS"; }

gpu_wait() {
  local required_free=$1 required_ram=$2 free used total ram
  while true; do
    read -r total used < <(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i 0)
    total=${total//,/}; used=${used//,/}; free=$((total - used))
    ram=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    if [ "$free" -ge "$required_free" ] && [ "$ram" -ge "$required_ram" ]; then
      echo "RESOURCE_OK required_free=${required_free}MiB actual_free=${free}MiB required_ram=${required_ram}GiB available_ram=${ram}GiB $(date -Is)"
      return 0
    fi
    echo "RESOURCE_WAIT required_free=${required_free}MiB actual_free=${free}MiB required_ram=${required_ram}GiB available_ram=${ram}GiB $(date -Is)"
    sleep 300
  done
}

set_status "RUNNING CUDA embedding smoke"
gpu_wait 10000 16
"$PY" cafa3_esmc_embed.py --data-dir "$DATA" --out "$SMOKE/embeddings" \
  --splits t0,t1 --limit 4 --batch 2 --max-length 512 > "$LOGDIR/smoke_embed.log" 2>&1
"$PY" - "$DATA" "$SMOKE/embeddings" > "$LOGDIR/smoke_verify.log" 2>&1 <<'PY'
import json
import sys
import numpy as np
import torch
from cafa3_esmc_embed import read_fasta
root, emb = sys.argv[1:]
report = {"cuda": torch.cuda.is_available(), "device": torch.cuda.get_device_name(0), "splits": {}}
assert report["cuda"]
for split, fasta in (("t0", "t0.fasta"), ("t1", "t1_benchmark.fasta")):
    expected, _ = read_fasta(f"{root}/{fasta}")
    saved = open(f"{emb}/{split}_ids.tsv").read().splitlines()
    array = np.load(f"{emb}/{split}.npy")
    assert saved == expected[:4]
    assert array.shape == (4, 960) and np.isfinite(array).all()
    report["splits"][split] = {"ids": saved, "shape": list(array.shape), "all_finite": True}
x = torch.from_numpy(np.load(f"{emb}/t0.npy").astype(np.float32)).cuda()
y = torch.zeros((4, 3), device="cuda"); y[:, 0] = 1
head = torch.nn.Linear(960, 3).cuda()
loss = torch.nn.functional.binary_cross_entropy_with_logits(head(x), y)
loss.backward()
assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
report["bce_backward"] = {"loss": float(loss.detach()), "finite_gradients": True}
with open(f"{emb}/../cuda_smoke.json", "w") as handle:
    json.dump(report, handle, indent=2)
print(json.dumps(report))
PY
date -Is > "$SMOKE/CUDA_ACCEPTED"
set_status "CUDA_SMOKE_ACCEPTED; RUNNING full t0+t1 embedding"
gpu_wait 10000 24
"$PY" cafa3_esmc_embed.py --data-dir "$DATA" --out "$EMB" --splits t0,t1 \
  --batch 16 --max-length 512 > "$LOGDIR/full_embed.log" 2>&1
date -Is > "$EMB/.complete"

for fold in 0 1 2; do
  for track in main id30 id50 id90; do
    for ontology in MFO BPO CCO; do
      out=$RUNS/fold${fold}/${track}/${ontology}
      log=$LOGDIR/fold${fold}_${track}_${ontology}.log
      if [ -f "$out/.complete" ] && [ -s "$out/summary.json" ]; then
        echo "SKIP fold=$fold track=$track ontology=$ontology"
        continue
      fi
      set_status "RUNNING fold=$fold track=$track ontology=$ontology"
      gpu_wait 12000 40
      "$PY" cafa3_esmc_baselines.py --data-dir "$DATA" --emb-dir "$EMB" --out "$out" \
        --fold "$fold" --track "$track" --ontology "$ontology" --seeds 0,1,2 \
        --epochs 1000 --patience 20 --min-delta 0.0001 --batch 512 \
        > "$log" 2>&1
      test -s "$out/summary.json"
      date -Is > "$out/.complete"
    done
  done
done
set_status "RUNNING independent audit"
"$PY" cafa3_esmc_audit.py --run-root "$BASE" --data-dir "$DATA" \
  > "$LOGDIR/final_audit.log" 2>&1
set_status "ALL_DONE"
echo "CAFA3_ESMC_BASELINES_ALL_DONE $(date -Is)"
