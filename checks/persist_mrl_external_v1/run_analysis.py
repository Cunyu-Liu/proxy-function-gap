import json
import time

from persist_mrl import ROOT, dump_json, load_protocol, prepare, verify_inputs, cuda_predict, summarize, write_manifest


def main():
    started = time.time()
    protocol = load_protocol()
    inputs = verify_inputs(protocol)
    dump_json({"status": "PASS", "files": inputs, "prediction_label_metrics_computed": False}, ROOT / "verification" / "round1_raw_inputs.json")
    audit, stopped = prepare(protocol)
    if stopped:
        dump_json({"status": "STOP", "reason": "Below frozen minimum sample", "inclusion": audit}, ROOT / "results" / "stopped.json")
        write_manifest()
        return
    cuda_predict(protocol)
    summary = summarize(protocol)
    manifest = write_manifest()
    dump_json({"status": "PASS", "primary": summary["primary"], "elapsed_seconds": time.time() - started, "artifact_count": len(manifest)}, ROOT / "verification" / "round1_execution.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
