"""Engineering-only RSS/GPU probe for CAFA3 baseline resource gates. No metrics, no training."""
import argparse, json, os, resource, sys
import numpy as np
import torch
from cafa3_esmc_baselines import Cafa3Run

def rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576

args = argparse.Namespace(
    data_dir="/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/external_benchmarks/cafa/cafa3_timeholdout_v1",
    emb_dir="/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/cafa3_timeholdout_esmc300m_p20_v1/embeddings",
    out="/home/ProxyFunctionGap/logs/cafa3_esmc300m_p20/resource_probe_out",
    fold=0, track="id50", ontology=sys.argv[1] if len(sys.argv) > 1 else "BPO",
    batch=512, k=20)
os.makedirs(args.out, exist_ok=True)
run = Cafa3Run(args, None, None)
init_rss = rss_gib()
n_active = int(run.active.sum())
train_x = torch.from_numpy(np.asarray(run.x["t0"][run.train_index], dtype=np.float32)).cuda()
train_y = torch.from_numpy(run.y_train[:, run.active].toarray().astype(np.float32)).cuda()
model = torch.nn.Linear(960, n_active).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
batch = train_x[:512]
target = torch.from_numpy(run.y_train[run.train_index[:512]][:, run.active].toarray().astype(np.float32)).cuda()
loss = torch.nn.functional.binary_cross_entropy_with_logits(model(batch), target)
loss.backward(); opt.step()
norm = torch.nn.functional.normalize(train_x, dim=1)
sim = (norm[:512] @ norm.T).topk(20, dim=1)
torch.cuda.synchronize()
report = {"ontology": args.ontology, "n_train": len(run.train_index), "n_val": len(run.val_index),
          "n_labels": len(run.labels), "n_active": n_active,
          "init_peak_rss_gib": round(init_rss, 3), "total_peak_rss_gib": round(rss_gib(), 3),
          "gpu_peak_mib": round(torch.cuda.max_memory_allocated() / 1048576, 1),
          "gpu_reserved_mib": round(torch.cuda.max_memory_reserved() / 1048576, 1),
          "train_y_dense_gib": round(train_y.numel() * 4 / 1073741824, 3)}
print(json.dumps(report, indent=2))
with open(os.path.join(args.out, f"resource_probe_{args.ontology}.json"), "w") as h:
    json.dump(report, h, indent=2)
