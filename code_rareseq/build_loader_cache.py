"""One-off builder for the finetune loader disk cache (engineering only)."""
from types import SimpleNamespace
import e1_finetune as F
args = SimpleNamespace(windows_dir="/root/rareseq/data/e0/windows", neg_dir="/root/rareseq/data/e0/neg",
    positives_audit="/root/rareseq/data/e0/positives_audit.tsv",
    split_clstr="/root/rareseq/data/e0/splits/clusters.id90.clstr",
    exps_file="/root/rareseq/data/e0/main_exps.txt", folds="0", tracks="neg1c,neg2")
loader = F.SequenceLoader(args)
tasks = F.cached_tasks(loader, args)
print("BUILDER_DONE", len(tasks))
