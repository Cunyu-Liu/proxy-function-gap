"""Verify loader disk cache equals fresh loader output (engineering check)."""
import pickle, hashlib, os
from types import SimpleNamespace
import e1_finetune as F
args = SimpleNamespace(windows_dir="/root/rareseq/data/e0/windows", neg_dir="/root/rareseq/data/e0/neg",
    positives_audit="/root/rareseq/data/e0/positives_audit.tsv",
    split_clstr="/root/rareseq/data/e0/splits/clusters.id90.clstr",
    exps_file="/root/rareseq/data/e0/main_exps.txt", folds="0", tracks="neg1c,neg2")
key = F._loader_cache_key(args)
path = os.path.join(F.LOADER_CACHE_ROOT, f"tasks_{key}_f0.pkl")
with open(path, "rb") as h:
    cached = pickle.load(h)["tasks"]
fresh = list(F.SequenceLoader(args).tasks())
assert len(cached) == len(fresh) == 292, (len(cached), len(fresh))
for c, f in zip(cached, fresh):
    assert c["experiment"] == f["experiment"] and c["track"] == f["track"] and c["fold"] == f["fold"]
    assert c["fold_counts"] == f["fold_counts"]
    for (ci, cy), (fi, fy) in zip(c["splits"], f["splits"]):
        assert ci == fi and (cy == fy).all()
print("LOADER_CACHE_VERIFY_OK 292 tasks, splits/labels/fold_counts identical")