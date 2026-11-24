#!/usr/bin/env python3
"""E6 GRIMM EC_v2 split 数据卡与标签/序列/簇重叠审计。"""
import argparse
import json
import os

import pandas as pd

from log_run import RunLogger, sha256_file


def labels(series):
    return set(x for value in series for x in str(value).split(";") if x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", default="/root/rareseq/logs/experiments.jsonl")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    lg = RunLogger(args.log, stage="E6", kind="grimm-audit")
    lg.start(config={"release": "EC_v2", "data_dir": args.data_dir}, data_version="GRIMM-EC_v2")
    frames = {name: pd.read_csv(os.path.join(args.data_dir, name + ".tsv"), sep="\t")
              for name in ("train", "validation", "test1", "test2")}
    summary, overlaps = {}, []
    for name, frame in frames.items():
        summary[name] = {"rows": len(frame), "unique_entry": frame.Entry.nunique(),
                         "unique_sequence": frame.Sequence.nunique(), "unique_ec_combo": frame["EC number"].nunique(),
                         "unique_ec_labels": len(labels(frame["EC number"])),
                         "unique_uniref50": frame.UniRef50.nunique(),
                         "duplicate_entry_rows": int(frame.Entry.duplicated().sum()),
                         "duplicate_sequence_rows": int(frame.Sequence.duplicated().sum())}
    names = list(frames)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            fa, fb = frames[a], frames[b]
            overlaps.append({"a": a, "b": b, "entry_overlap": len(set(fa.Entry) & set(fb.Entry)),
                             "sequence_overlap": len(set(fa.Sequence) & set(fb.Sequence)),
                             "uniref50_overlap": len(set(fa.UniRef50) & set(fb.UniRef50)),
                             "ec_label_overlap": len(labels(fa["EC number"]) & labels(fb["EC number"]))})
    counts = labels(frames["train"]["EC number"])
    frequencies = frames["train"].assign(ec=frames["train"]["EC number"].str.split(";")).explode("ec").ec.value_counts()
    frequency_table = frequencies.rename_axis("ec").reset_index(name="n_train")
    frequency_table.to_csv(os.path.join(args.out, "ec_train_frequency.tsv"), sep="\t", index=False)
    audit = {"splits": summary, "overlaps": overlaps,
             "train_ec_label_count": len(frequencies),
             "train_label_frequency": {"min": int(frequencies.min()), "median": float(frequencies.median()),
                                       "max": int(frequencies.max()),
                                       "labels_lt10": int((frequencies < 10).sum()),
                                       "labels_lt30": int((frequencies < 30).sum()),
                                       "labels_lt100": int((frequencies < 100).sum())},
             "test2_labels_not_in_train": sorted(labels(frames["test2"]["EC number"]) - counts),
             "test2_label_overlap_with_train": sorted(labels(frames["test2"]["EC number"]) & counts)}
    with open(os.path.join(args.out, "grimm_audit.json"), "w") as handle:
        json.dump(audit, handle, indent=2)
    pd.DataFrame(overlaps).to_csv(os.path.join(args.out, "overlap_summary.tsv"), sep="\t", index=False)
    lg.finish(metrics={"splits": summary, "train_label_frequency": audit["train_label_frequency"],
                       "test2_new_label_count": len(audit["test2_labels_not_in_train"])}, outputs=[args.out])
    print(json.dumps(audit, indent=2)[:5000])


if __name__ == "__main__":
    main()
