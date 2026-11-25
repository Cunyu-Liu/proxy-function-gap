import fcntl
import json
import os

from log_run import RunLogger


class E1RunLogger(RunLogger):
    def _append(self):
        payload = (json.dumps(self.rec, ensure_ascii=False) + "\n").encode("utf-8")
        with open(self.path + ".e1.lock", "a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                written = os.write(fd, payload)
                if written != len(payload):
                    raise OSError(f"partial JSONL append: {written}/{len(payload)}")
            finally:
                os.close(fd)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
