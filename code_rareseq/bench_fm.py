#!/usr/bin/env python3
"""E0 吞吐实测：RNA-FM / RiNALMo 在 RTX 5090 上的冻结前向、冻结+头训练、LoRA、全参。
合成 101nt 随机序列（固定种子），测 seq/s 与峰值显存，替换执行文档 §7 假设。
用法: HF_HOME=/root/hf_home HF_HUB_OFFLINE=1 PYTHONPATH=/root/rareseq/pydeps:/root/rnaft/pypath \
      /root/miniconda3/envs/llr_env/bin/python bench_fm.py
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/root/rareseq/work")
from log_run import RunLogger  # noqa: E402

SEQ_LEN = 101
N_BATCHES = 12          # 计时批数（前2批 warmup 不计）
WARMUP = 2
SEED = 20261122
OUT = "/root/rareseq/data/e0/bench_fm.json"

MODELS = [
    ("rnafm", "multimolecule/rnafm"),
    ("rinalmo-giga", "multimolecule/rinalmo-giga"),
]
BATCHES = {
    "rnafm": [64, 128, 256, 512],
    "rinalmo-giga": [16, 32, 64, 128],
}


def make_seqs(n, seed):
    rng = np.random.default_rng(seed)
    bases = np.array(list("ACGU"))
    return ["".join(bases[rng.integers(0, 4, SEQ_LEN)]) for _ in range(n)]


def load_model(name, repo):
    from multimolecule import AutoModel, AutoTokenizer  # noqa
    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    model = AutoModel.from_pretrained(repo, trust_remote_code=True)
    return tok, model


def encode_batch(tok, seqs, device):
    enc = tok(seqs, return_tensors="pt", padding=True, truncation=True, max_length=SEQ_LEN + 2)
    return {k: v.to(device) for k, v in enc.items()}


def find_lora_targets(model):
    cands = {"query", "key", "value", "q_proj", "k_proj", "v_proj", "qkv", "Wqkv"}
    found = set()
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            leaf = n.split(".")[-1]
            if leaf in cands:
                found.add(leaf)
    for trio in (["query", "key", "value"], ["q_proj", "k_proj", "v_proj"]):
        if all(t in found for t in trio):
            return trio
    for single in ("qkv", "Wqkv"):
        if single in found:
            return [single]
    return sorted(found) or ["query"]


def run_mode(tok, model, device, mode, bs, dtype):
    seqs = make_seqs(bs * (N_BATCHES + WARMUP), SEED)
    batches = [encode_batch(tok, seqs[i * bs:(i + 1) * bs], device)
               for i in range(N_BATCHES + WARMUP)]
    model.train(mode != "frozen-fwd")
    for p in model.parameters():
        p.requires_grad_(mode == "full")
    head = None
    opt = None
    if mode in ("frozen-head", "lora", "full"):
        hidden = model.config.hidden_size
        head = torch.nn.Linear(hidden, 1).to(device)
        params = list(head.parameters())
        if mode == "lora":
            from peft import LoraConfig, get_peft_model
            targets = find_lora_targets(model)
            cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                             target_modules=targets, bias="none")
            model = get_peft_model(model, cfg)
            params += [p for p in model.parameters() if p.requires_grad]
        elif mode == "full":
            params += list(model.parameters())
        opt = torch.optim.AdamW(params, lr=1e-4)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    t0 = time.time()
    nseq = 0
    for i, batch in enumerate(batches):
        if i == WARMUP:
            torch.cuda.synchronize()
            t0 = time.time()
            nseq = 0
        with torch.autocast("cuda", dtype=dtype):
            if mode == "frozen-fwd":
                with torch.no_grad():
                    out = model(**batch).last_hidden_state
            else:
                out = model(**batch).last_hidden_state
                logits = head(out[:, 0])
                loss = logits.float().square().mean()
        if mode != "frozen-fwd":
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        nseq += bs
    torch.cuda.synchronize()
    dt = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
    return {"seq_per_s": nseq / dt, "peak_vram_gb": round(peak_gb, 2),
            "s_per_batch": dt / N_BATCHES, "batch_size": bs, "mode": mode}


def main():
    device = "cuda"
    dtype = torch.bfloat16
    results = {}
    for name, repo in MODELS:
        print(f"=== {name} ({repo}) ===", flush=True)
        try:
            tok, model = load_model(name, repo)
        except Exception as e:  # noqa: BLE001
            print(f"LOAD FAIL {name}: {type(e).__name__} {str(e)[:200]}", flush=True)
            results[name] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            continue
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"params: {n_params/1e6:.1f}M", flush=True)
        results[name] = {"params_m": round(n_params / 1e6, 1), "modes": {}}
        modes = ["frozen-fwd", "frozen-head", "lora", "full"] if name == "rnafm" else ["frozen-fwd", "lora"]
        for mode in modes:
            for bs in BATCHES[name]:
                tag = f"{mode}-bs{bs}"
                try:
                    if mode == "lora":
                        tok2, m2 = load_model(name, repo)
                        m2 = m2.to(device)
                        r = run_mode(tok2, m2, device, mode, bs, dtype)
                        del m2
                    elif mode == "full":
                        tok2, m2 = load_model(name, repo)
                        m2 = m2.to(device)
                        r = run_mode(tok2, m2, device, mode, bs, dtype)
                        del m2
                    else:
                        r = run_mode(tok, model, device, mode, bs, dtype)
                    results[name]["modes"][tag] = r
                    print(f"{tag}: {r['seq_per_s']:.0f} seq/s  vram {r['peak_vram_gb']}GB", flush=True)
                    lg = RunLogger("/root/rareseq/logs/experiments.jsonl", stage="E0", kind="throughput")
                    lg.start(config={"model": repo, "mode": mode, "batch_size": bs,
                                     "seq_len": SEQ_LEN, "dtype": "bf16", "gpu": "RTX5090"},
                             seed=SEED, data_version="synthetic")
                    lg.finish(metrics=r, outputs=[OUT])
                except torch.cuda.OutOfMemoryError:
                    print(f"{tag}: OOM", flush=True)
                    results[name]["modes"][tag] = {"oom": True}
                    torch.cuda.empty_cache()
                except Exception as e:  # noqa: BLE001
                    print(f"{tag}: FAIL {type(e).__name__} {str(e)[:150]}", flush=True)
                    results[name]["modes"][tag] = {"error": str(e)[:200]}
                    torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print("SAVED", OUT, flush=True)


if __name__ == "__main__":
    main()