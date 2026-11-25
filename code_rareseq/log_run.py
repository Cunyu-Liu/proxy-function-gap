#!/usr/bin/env python3
"""RARE-Seq 统一实验日志工具。

每个 run 追加一条 JSONL 记录到实验日志（本地项目目录与服务器各一份，定期同步）。
用法（脚本内）:
    from log_run import RunLogger
    lg = RunLogger("/root/rareseq/logs/experiments.jsonl", stage="E0", kind="clustering")
    lg.start(config={...}, seed=0, data_version="...", inputs=[...])
    ... 执行 ...
    lg.finish(metrics={...}, outputs=[...], notes="...")
命令行查看: python log_run.py <jsonl路径> [stage过滤]
"""
import json
import os
import socket
import sys
import time
import datetime
import hashlib
import fcntl


def sha256_file(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        if limit:
            h.update(f.read(limit))
        else:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


class RunLogger:
    def __init__(self, path, stage, kind):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.stage = stage
        self.kind = kind
        self.rec = None

    def start(self, config, seed=None, data_version="", inputs=None, code_version="", run_id=None):
        ts = datetime.datetime.now(datetime.timezone.utc)
        rid = run_id or f"{self.stage}-{self.kind}-{ts.strftime('%Y%m%d-%H%M%S')}-{socket.gethostname()[:6]}"
        self.rec = {
            "run_id": rid, "stage": self.stage, "kind": self.kind,
            "start_utc": ts.isoformat(), "host": socket.gethostname(),
            "config": config, "seed": seed, "data_version": data_version,
            "code_version": code_version, "inputs": inputs or [],
            "status": "running",
        }
        self._append()
        self.t0 = time.time()
        return rid

    def finish(self, metrics=None, outputs=None, notes="", status="completed"):
        assert self.rec is not None
        self.rec.update({
            "end_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "wall_seconds": round(time.time() - self.t0, 2),
            "metrics": metrics or {}, "outputs": outputs or [],
            "notes": notes, "status": status,
        })
        self._append()

    def _append(self):
        payload = (json.dumps(self.rec, ensure_ascii=False) + "\n").encode("utf-8")
        with open(self.path + ".lock", "a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            descriptor = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
            try:
                written = os.write(descriptor, payload)
                if written != len(payload):
                    raise OSError(f"Partial JSONL append: {written}/{len(payload)}")
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


if __name__ == "__main__":
    p = sys.argv[1]
    stage = sys.argv[2] if len(sys.argv) > 2 else None
    seen = {}
    for line in open(p):
        r = json.loads(line)
        if stage and r["stage"] != stage:
            continue
        seen[r["run_id"]] = r  # 同 run_id 取最新状态
    for r in seen.values():
        m = json.dumps(r.get("metrics", {}), ensure_ascii=False)[:160]
        print(f"{r['run_id']}  [{r['status']}]  {r.get('wall_seconds','?')}s  {m}")