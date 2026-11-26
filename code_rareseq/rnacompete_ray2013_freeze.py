#!/usr/bin/env python3
"""Audit and freeze the RNAcompete Ray2013 regression benchmark; no model training."""
import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np
import pandas as pd

VERSION = "rnacompete-ray2013-v1.0-id90-strict3train"
SEED = 20261126
N_FOLDS = 5
MIN_FINITE_PER_FOLD = 100
EXPECTED = {
    "SupplementaryData1_RNAcompete_master_file.xlsx": {
        "bytes": 2677507,
        "sha256": "3da82ac06deae33264ff2d49f83cd5302dc9d12ccb559032b84d93fdaf42f261",
    },
    "RBD_v3_design_unified.txt": {
        "bytes": 47537452,
        "sha256": "47abcc7272ad43ec8b0bcb210d3c1c4529e986ba19ea45a6638617eebd5e536a",
    },
    "norm_data.txt.gz": {
        "bytes": 249998462,
        "sha256": "efa7fade5fe6c3a189ece3f381550f73f3bb3d935598d211d69d29d9e3ae47ff",
    },
}
XML_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_id(sequence):
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def split_roles(test_fold):
    validation = (test_fold + 1) % N_FOLDS
    train = [fold for fold in range(N_FOLDS) if fold not in (test_fold, validation)]
    assert len(train) == 3
    return {"train": train, "validation": [validation], "test": [test_fold]}


def write_table(frame, path):
    if path.endswith(".gz"):
        with open(path, "wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                    frame.to_csv(text, sep="\t", index=False, float_format="%.10g")
    else:
        frame.to_csv(path, sep="\t", index=False, float_format="%.10g")


def read_master_xlsx(path):
    with zipfile_open(path) as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = ["".join(node.text or "" for node in item.iter(XML_NS + "t"))
                      for item in root.findall(XML_NS + "si")]
        root = ET.fromstring(archive.read("xl/worksheets/sheet2.xml"))
        rows = []
        for row in root.findall(".//" + XML_NS + "sheetData/" + XML_NS + "row"):
            values = {}
            for cell in row.findall(XML_NS + "c"):
                reference = cell.get("r")
                letters = re.match(r"([A-Z]+)", reference).group(1)
                index = 0
                for letter in letters:
                    index = index * 26 + ord(letter) - ord("A") + 1
                value_node = cell.find(XML_NS + "v")
                value = "" if value_node is None else value_node.text
                if cell.get("t") == "s" and value:
                    value = shared[int(value)]
                elif cell.get("t") == "inlineStr":
                    value = "".join(node.text or "" for node in cell.iter(XML_NS + "t"))
                values[index - 1] = value or ""
            rows.append(values)
    header = [rows[0].get(index, "") for index in range(max(rows[0]) + 1)]
    records = [[row.get(index, "") for index in range(len(header))] for row in rows[1:]]
    return pd.DataFrame(records, columns=header)


def zipfile_open(path):
    import zipfile
    return zipfile.ZipFile(path)


def read_design(path):
    records = []
    controls = Counter()
    with open(path, newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        expected_header = ["%Meta Row", "Meta Column", "Row", "Col", "ID", "ControlType",
                           "PrimaryAccession", "GeneSymbol", "Accessions", "Description"]
        assert header == expected_header, header
        for line_number, row in enumerate(reader, start=2):
            assert len(row) in (10, 14), (line_number, len(row))
            control_type = row[5]
            controls[control_type] += 1
            if control_type != "false":
                assert len(row) == 10, (line_number, control_type)
                continue
            assert len(row) == 14, (line_number, control_type)
            match = re.search(r"(?:^|\s)(SetA|SetB)(?:_|\s|$)", row[9])
            probe_set = match.group(1) if match else "other"
            records.append({
                "meta_row": int(row[0]), "meta_column": int(row[1]),
                "array_row": int(row[2]), "array_col": int(row[3]),
                "probe_id": row[4], "description": row[9], "probe_set": probe_set,
                "rna_seq": row[11].strip().upper(), "structure": row[12], "mfe": row[13],
            })
    frame = pd.DataFrame(records)
    assert frame.probe_id.is_unique
    assert controls["false"] == 241399
    assert controls["pos"] == 1503 and controls["neg"] == 602
    return frame, dict(controls)


def scan_normalized(path):
    counts = Counter()
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        width = len(header)
        for row in reader:
            counts["rows"] += 1
            assert len(row) == width, (counts["rows"] + 1, len(row), width)
            for value in row[3:]:
                stripped = value.strip()
                lowered = stripped.lower()
                if not stripped:
                    counts["blank"] += 1
                elif lowered == "nan":
                    counts["nan"] += 1
                elif lowered in ("inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"):
                    counts["inf"] += 1
    assert counts["blank"] == 0, dict(counts)
    assert counts["inf"] == 0, dict(counts)
    return header, dict(counts)


def parse_clstr(path):
    mapping = {}
    cluster = None
    with open(path) as handle:
        for line in handle:
            if line.startswith(">Cluster"):
                cluster = int(line.split()[1])
            else:
                assert cluster is not None
                identifier = line.split(">", 1)[1].split("...", 1)[0]
                assert identifier not in mapping, identifier
                mapping[identifier] = cluster
    return mapping


def assign_balanced(weight_by_key, seed=SEED):
    keys = sorted(weight_by_key, key=lambda key: str(key))
    rng = np.random.default_rng(seed)
    jitter = dict(zip(keys, rng.random(len(keys))))
    ordered = sorted(keys, key=lambda key: (-weight_by_key[key], jitter[key], str(key)))
    loads = [0] * N_FOLDS
    assignment = {}
    for key in ordered:
        fold = int(np.argmin(loads))
        assignment[key] = fold
        loads[fold] += int(weight_by_key[key])
    return assignment, loads


def frequency_tiers(experiments):
    eligible = experiments[experiments.sequence_track_included].copy()
    eligible = eligible.sort_values(["n_finite_unique", "Motif_ID"], ascending=[False, True])
    third = int(math.ceil(len(eligible) / 3.0))
    tiers = (["head"] * min(third, len(eligible)) +
             ["mid"] * min(third, max(0, len(eligible) - third)) +
             ["tail"] * max(0, len(eligible) - 2 * third))
    return dict(zip(eligible.Motif_ID, tiers))


def aggregate_sequence_labels(probe_frame, values):
    frame = probe_frame[["sequence_id", "probe_set", "fold"]].copy()
    frame["value"] = np.asarray(values, dtype=np.float64)
    grouped = frame.groupby("sequence_id", sort=False)
    result = grouped.agg(probe_set=("probe_set", "first"), fold=("fold", "first"),
                         label_median=("value", "median"), label_min=("value", "min"),
                         label_max=("value", "max"), n_finite=("value", "count"),
                         n_probe_rows=("value", "size")).reset_index()
    result["n_nan"] = result.n_probe_rows - result.n_finite
    result["duplicate_label_range"] = result.label_max - result.label_min
    return result


def run_cdhit(fasta, output_prefix, threads, memory_mb):
    command = ["cd-hit-est", "-i", fasta, "-o", output_prefix,
               "-c", "0.9", "-n", "8", "-G", "1", "-g", "1", "-r", "1",
               "-aL", "0.9", "-aS", "0.9", "-d", "0",
               "-T", str(threads), "-M", str(memory_mb)]
    version_run = subprocess.run(["cd-hit-est", "-h"], stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, universal_newlines=True)
    version = version_run.stdout.splitlines()[0] if version_run.stdout else "unknown"
    started = time.time()
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               universal_newlines=True)
    if completed.returncode != 0:
        raise RuntimeError("cd-hit-est failed:\n" + completed.stdout[-4000:])
    return command, version, time.time() - started, completed.stdout


def load_experiment_labels(input_root, frozen_root, motif_id):
    """Load one experiment as unique-RNA median labels without imputing NaN."""
    probes = pd.read_csv(os.path.join(frozen_root, "probes.tsv.gz"), sep="\t")
    values = pd.read_csv(os.path.join(input_root, "norm_data.txt.gz"), sep="\t",
                         usecols=["Probe_ID", motif_id])
    merged = probes.merge(values, left_on="probe_id", right_on="Probe_ID",
                          how="left", validate="one_to_one")
    assert merged[motif_id].notna().any(), motif_id
    return aggregate_sequence_labels(probes, merged[motif_id].to_numpy())


def require_teams(path):
    absolute = os.path.abspath(path)
    assert absolute == "/teams" or absolute.startswith("/teams/"), path
    return absolute


def freeze(args):
    input_root = require_teams(args.input_root)
    out = require_teams(args.out)
    manifest_path = os.path.join(out, "split_manifest.json")
    assert not os.path.exists(manifest_path), "Frozen manifest already exists; use --verify-only"
    os.makedirs(out, exist_ok=True)
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started = time.time()
    run_log = {"stage": "RNAcompete", "kind": "data-audit-split-freeze", "version": VERSION,
               "start_utc": started_utc, "status": "running", "model_training": False}
    try:
        inputs = {}
        for name, expected in EXPECTED.items():
            path = os.path.join(input_root, name)
            actual = {"bytes": os.path.getsize(path), "sha256": sha256_file(path)}
            assert actual == expected, (name, actual, expected)
            inputs[name] = actual
        print("INPUT_SHA256_OK", flush=True)

        master = read_master_xlsx(os.path.join(input_root, "SupplementaryData1_RNAcompete_master_file.xlsx"))
        assert len(master) == 244 and master.Motif_ID.is_unique
        assert not master[["Species", "Gene_systematic_ID"]].isna().any().any()
        assert not (master.Species.astype(str).str.strip() == "").any()
        assert not (master.Gene_systematic_ID.astype(str).str.strip() == "").any()
        master["species_gene"] = master.Species + "::" + master.Gene_systematic_ID
        assert master.species_gene.nunique() == 205

        design, controls = read_design(os.path.join(input_root, "RBD_v3_design_unified.txt"))
        norm_path = os.path.join(input_root, "norm_data.txt.gz")
        norm_header, norm_scan = scan_normalized(norm_path)
        matrix = pd.read_csv(norm_path, sep="\t")
        assert matrix.columns.tolist() == norm_header
        assert matrix.columns[:3].tolist() == ["Probe_Set", "RNA_Seq", "Probe_ID"]
        experiments = list(matrix.columns[3:])
        assert len(experiments) == 244 and len(set(experiments)) == 244
        assert set(experiments) == set(master.Motif_ID)
        assert matrix.Probe_ID.is_unique and len(matrix) == 241357
        numeric = matrix[experiments].to_numpy(dtype=np.float64)
        assert not np.isinf(numeric).any()
        assert int(np.isnan(numeric).sum()) == norm_scan["nan"]
        assert norm_scan["nan"] == 787163

        norm_meta = matrix[["Probe_ID", "Probe_Set", "RNA_Seq"]].rename(
            columns={"Probe_ID": "probe_id", "Probe_Set": "matrix_probe_set",
                     "RNA_Seq": "matrix_rna_seq"})
        assert set(norm_meta.probe_id).issubset(set(design.probe_id))
        merged = design.merge(norm_meta, on="probe_id", how="left", validate="one_to_one")
        matched = merged.matrix_probe_set.notna()
        assert (merged.loc[matched, "probe_set"] == merged.loc[matched, "matrix_probe_set"]).all()
        assert (merged.loc[matched, "rna_seq"] == merged.loc[matched, "matrix_rna_seq"]).all()
        merged["exclusion_reason"] = ""
        merged.loc[~matched, "exclusion_reason"] = "absent_from_normalized"
        merged.loc[merged.probe_set == "other", "exclusion_reason"] = "not_SetA_B"
        excluded = merged[merged.exclusion_reason != ""].copy()
        assert len(excluded) == 42
        assert (excluded.exclusion_reason == "not_SetA_B").sum() == 42
        assert (excluded.exclusion_reason == "absent_from_normalized").sum() == 0
        assert merged.loc[matched, "probe_set"].isin(["SetA", "SetB"]).all()
        excluded_columns = ["probe_id", "probe_set", "rna_seq", "description", "exclusion_reason"]
        write_table(excluded[excluded_columns], os.path.join(out, "excluded_probes.tsv"))

        probes = merged[merged.exclusion_reason == ""].copy()
        assert len(probes) == 241357
        unique_sequences = sorted(probes.rna_seq.unique())
        assert len(unique_sequences) == 219990
        assert all(sequence and re.fullmatch(r"[ACGU]+", sequence) for sequence in unique_sequences)
        id_by_sequence = {sequence: sequence_id(sequence) for sequence in unique_sequences}
        assert len(set(id_by_sequence.values())) == len(unique_sequences)
        probes["sequence_id"] = probes.rna_seq.map(id_by_sequence)
        probes["rna_length"] = probes.rna_seq.str.len()
        probes = probes.sort_values("probe_id").reset_index(drop=True)
        matrix = matrix.set_index("Probe_ID").loc[probes.probe_id].reset_index()
        assert (matrix.Probe_ID.to_numpy() == probes.probe_id.to_numpy()).all()

        sequence_table = probes.drop_duplicates("sequence_id").copy()
        sequence_table = sequence_table[["sequence_id", "rna_seq", "rna_length", "probe_set"]]
        sequence_table = sequence_table.sort_values("sequence_id").reset_index(drop=True)
        assert sequence_table.groupby("rna_seq").sequence_id.nunique().max() == 1
        fasta = os.path.join(out, "sequences.id90.fasta")
        with open(fasta, "w") as handle:
            for row in sequence_table.itertuples():
                handle.write(">" + row.sequence_id + "\n" + row.rna_seq.replace("U", "T") + "\n")
        command, cdhit_version, cluster_seconds, cluster_log = run_cdhit(
            fasta, os.path.join(out, "clusters.id90"), args.threads, args.memory_mb)
        with open(os.path.join(out, "cdhit.log"), "w") as handle:
            handle.write(cluster_log)
        cluster_by_sequence = parse_clstr(os.path.join(out, "clusters.id90.clstr"))
        assert set(cluster_by_sequence) == set(sequence_table.sequence_id)
        sequence_table["cluster_id"] = sequence_table.sequence_id.map(cluster_by_sequence)
        cluster_sizes = sequence_table.groupby("cluster_id").size().to_dict()
        cluster_fold, fold_loads = assign_balanced(cluster_sizes)
        sequence_table["fold"] = sequence_table.cluster_id.map(cluster_fold)
        assert sequence_table.fold.notna().all()
        assert sequence_table.groupby("cluster_id").fold.nunique().max() == 1
        assert sorted(sequence_table.groupby("fold").size().to_dict()) == list(range(N_FOLDS))
        assert sequence_table.groupby("fold").size().to_dict() == fold_loads
        print("CDHIT_AND_SEQUENCE_FOLDS_OK", fold_loads, flush=True)

        group_sizes = master.groupby("species_gene").size().to_dict()
        group_fold, rbp_fold_loads = assign_balanced(group_sizes, seed=SEED + 1)
        master["rbp_fold"] = master.species_gene.map(group_fold)
        assert master.rbp_fold.notna().all()
        assert master.groupby("species_gene").rbp_fold.nunique().max() == 1
        rbp_folds = master[["Motif_ID", "Species", "Gene_systematic_ID", "species_gene",
                            "Plasmid_ID", "Hyb_ID", "rbp_fold"]].sort_values("Motif_ID")
        write_table(rbp_folds, os.path.join(out, "rbp_folds.tsv"))

        probes = probes.merge(sequence_table[["sequence_id", "cluster_id", "fold"]],
                              on="sequence_id", how="left", validate="many_to_one")
        assert (matrix.Probe_ID.to_numpy() == probes.probe_id.to_numpy()).all()
        assert probes.fold.notna().all()
        probe_columns = ["probe_id", "probe_set", "sequence_id", "rna_seq", "rna_length",
                         "cluster_id", "fold", "meta_row", "meta_column", "array_row", "array_col",
                         "description", "structure", "mfe"]
        write_table(probes[probe_columns], os.path.join(out, "probes.tsv.gz"))

        sequence_index = pd.Index(sequence_table.sequence_id)
        observed_experiment_count = np.zeros(len(sequence_table), dtype=np.int32)
        availability_rows = []
        duplicate_rows = []
        experiment_stats = {}
        for position, motif_id in enumerate(experiments, start=1):
            aggregated = aggregate_sequence_labels(probes, matrix[motif_id].to_numpy(dtype=np.float64))
            observed = aggregated.n_finite > 0
            observed_experiment_count += observed.to_numpy(dtype=np.int32)[
                sequence_index.get_indexer(aggregated.sequence_id)]
            finite_labels = aggregated.loc[observed, "label_median"]
            fold_counts = aggregated[observed].groupby("fold").size()
            experiment_stats[motif_id] = {
                "n_finite_unique": int(observed.sum()),
                "n_distinct_median_labels": int(finite_labels.nunique()),
                "min_fold_finite_unique": int(fold_counts.min()),
            }
            for fold in range(N_FOLDS):
                for probe_set in ("All", "SetA", "SetB"):
                    subset = aggregated[aggregated.fold == fold]
                    if probe_set != "All":
                        subset = subset[subset.probe_set == probe_set]
                    labels = subset.loc[subset.n_finite > 0, "label_median"]
                    availability_rows.append({
                        "Motif_ID": motif_id, "fold": fold, "probe_set": probe_set,
                        "n_unique_rna": len(subset), "n_finite_unique": int(subset.n_finite.gt(0).sum()),
                        "n_all_nan_unique": int(subset.n_finite.eq(0).sum()),
                        "label_median": float(labels.median()) if len(labels) else np.nan,
                        "label_p05": float(labels.quantile(0.05)) if len(labels) else np.nan,
                        "label_p95": float(labels.quantile(0.95)) if len(labels) else np.nan,
                    })
            duplicated = aggregated[aggregated.n_probe_rows > 1]
            multi_finite = duplicated[duplicated.n_finite > 1]
            ranges = multi_finite.duplicate_label_range
            duplicate_rows.append({
                "Motif_ID": motif_id,
                "n_duplicate_unique_rna": len(duplicated),
                "n_duplicate_probe_rows": int(duplicated.n_probe_rows.sum()),
                "n_duplicate_rna_multi_finite": len(multi_finite),
                "n_duplicate_rna_discordant": int((ranges > 0).sum()),
                "median_duplicate_label_range": float(ranges.median()) if len(ranges) else np.nan,
                "max_duplicate_label_range": float(ranges.max()) if len(ranges) else np.nan,
            })
            if position % 20 == 0 or position == len(experiments):
                print("LABEL_AUDIT", position, "/", len(experiments), flush=True)

        availability = pd.DataFrame(availability_rows)
        duplicate_audit = pd.DataFrame(duplicate_rows)
        write_table(availability, os.path.join(out, "label_availability.tsv.gz"))
        write_table(duplicate_audit, os.path.join(out, "duplicate_audit.tsv"))
        sequence_table["n_probe_rows"] = sequence_table.sequence_id.map(probes.groupby("sequence_id").size())
        sequence_table["n_experiments_observed"] = observed_experiment_count
        sequence_columns = ["sequence_id", "rna_seq", "rna_length", "probe_set", "cluster_id",
                            "fold", "n_probe_rows", "n_experiments_observed"]
        write_table(sequence_table[sequence_columns], os.path.join(out, "sequence_folds.tsv.gz"))

        stats_frame = pd.DataFrame.from_dict(experiment_stats, orient="index")
        stats_frame.index.name = "Motif_ID"
        master = master.merge(stats_frame, on="Motif_ID", how="left", validate="one_to_one")
        master["sequence_track_included"] = (
            (master.min_fold_finite_unique >= MIN_FINITE_PER_FOLD) &
            (master.n_distinct_median_labels >= 2))
        master["exclusion_reason"] = ""
        master.loc[master.min_fold_finite_unique < MIN_FINITE_PER_FOLD,
                   "exclusion_reason"] = "fewer_than_100_finite_unique_in_a_fold"
        master.loc[master.n_distinct_median_labels < 2,
                   "exclusion_reason"] = "fewer_than_2_distinct_median_labels"
        tier_by_experiment = frequency_tiers(master)
        master["frequency_tier"] = master.Motif_ID.map(tier_by_experiment).fillna("excluded")
        experiment_columns = ["Motif_ID", "Gene_systematic_ID", "Protein_name", "Species",
                              "species_gene", "Domains", "Construct_type", "Plasmid_ID", "Hyb_ID",
                              "rbp_fold", "n_finite_unique", "n_distinct_median_labels",
                              "min_fold_finite_unique", "sequence_track_included",
                              "exclusion_reason", "frequency_tier"]
        write_table(master[experiment_columns].sort_values("Motif_ID"),
                    os.path.join(out, "experiments.tsv"))

        leakage_rows = []
        for fold in range(N_FOLDS):
            roles = split_roles(fold)
            fold_frames = {role: sequence_table[sequence_table.fold.isin(folds)]
                           for role, folds in roles.items()}
            for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
                exact = set(fold_frames[left].rna_seq) & set(fold_frames[right].rna_seq)
                clusters = set(fold_frames[left].cluster_id) & set(fold_frames[right].cluster_id)
                leakage_rows.append({"track": "sequence", "fold": fold,
                                     "comparison": left + "__" + right,
                                     "n_left": len(fold_frames[left]), "n_right": len(fold_frames[right]),
                                     "n_exact_overlap": len(exact), "n_cluster_overlap": len(clusters),
                                     "n_group_overlap": 0})
            group_frames = {role: set(master.loc[master.rbp_fold.isin(folds), "species_gene"])
                            for role, folds in roles.items()}
            for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
                overlap = group_frames[left] & group_frames[right]
                leakage_rows.append({"track": "rbp_ood", "fold": fold,
                                     "comparison": left + "__" + right,
                                     "n_left": len(group_frames[left]), "n_right": len(group_frames[right]),
                                     "n_exact_overlap": 0, "n_cluster_overlap": 0,
                                     "n_group_overlap": len(overlap)})
        set_a = set(sequence_table.loc[sequence_table.probe_set == "SetA", "rna_seq"])
        set_b = set(sequence_table.loc[sequence_table.probe_set == "SetB", "rna_seq"])
        leakage_rows.append({"track": "setA_setB", "fold": -1, "comparison": "SetA__SetB",
                             "n_left": len(set_a), "n_right": len(set_b),
                             "n_exact_overlap": len(set_a & set_b), "n_cluster_overlap": np.nan,
                             "n_group_overlap": 0})
        leakage = pd.DataFrame(leakage_rows)
        assert (leakage.n_exact_overlap.fillna(0) == 0).all()
        assert (leakage.n_cluster_overlap.fillna(0) == 0).all()
        assert (leakage.n_group_overlap == 0).all()
        write_table(leakage, os.path.join(out, "leakage_audit.tsv"))

        elapsed = time.time() - started
        run_log.update({"status": "accepted", "end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "elapsed_seconds": elapsed, "code_commit": args.code_commit})
        with open(os.path.join(out, "run_log.json"), "w") as handle:
            json.dump(run_log, handle, indent=2, sort_keys=True)
        output_hashes = {}
        for name in sorted(os.listdir(out)):
            path = os.path.join(out, name)
            if os.path.isfile(path) and name not in ("split_manifest.json", "SHA256SUMS"):
                output_hashes[name] = sha256_file(path)
        manifest = {
            "version": VERSION, "status": "accepted", "kind": "data_audit_and_split_freeze",
            "model_results": False, "created_utc": started_utc, "seed": SEED,
            "code_commit": args.code_commit, "code_sha256": sha256_file(os.path.abspath(__file__)),
            "contract_sha256": sha256_file(args.contract),
            "inputs": inputs, "normalized_scan": norm_scan, "design_controls": controls,
            "experiments_total": len(master),
            "experiments_sequence_track_included": int(master.sequence_track_included.sum()),
            "experiments_excluded": master.loc[~master.sequence_track_included,
                                               ["Motif_ID", "exclusion_reason"]].to_dict("records"),
            "rbp_groups": len(group_sizes), "probes_main": len(probes), "probes_excluded": len(excluded),
            "unique_rna_main": len(sequence_table), "clusters": len(cluster_sizes),
            "sequence_fold_unique_rna": fold_loads, "rbp_fold_experiment_counts": rbp_fold_loads,
            "frequency_tier_counts": master.frequency_tier.value_counts().to_dict(),
            "frequency_quantiles": {str(key): float(value) for key, value in
                                    master.loc[master.sequence_track_included, "n_finite_unique"].quantile(
                                        [0, 0.25, 0.5, 0.75, 1]).items()},
            "cross_experiment_exact_sequence_observation": {
                "observed_in_at_least_2_experiments": int((observed_experiment_count >= 2).sum()),
                "observed_in_all_244_experiments": int((observed_experiment_count == 244).sum()),
                "note": "Shared probes are intended across RBP tasks; RBP-OOD is not sequence-OOD.",
            },
            "split_roles": {str(fold): split_roles(fold) for fold in range(N_FOLDS)},
            "cdhit": {"version": cdhit_version, "command": command,
                      "elapsed_seconds": cluster_seconds},
            "aggregation": "per experiment and unique RNA: median of finite probe values; NaN remains unobserved",
            "minimum_finite_unique_per_fold": MIN_FINITE_PER_FOLD,
            "leakage_checks_passed": True, "output_sha256": output_hashes,
            "elapsed_seconds": elapsed,
        }
        with open(manifest_path, "w") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True, ensure_ascii=False)
        with open(os.path.join(out, "SHA256SUMS"), "w") as handle:
            for name in sorted(output_hashes):
                handle.write(output_hashes[name] + "  " + name + "\n")
            handle.write(sha256_file(manifest_path) + "  split_manifest.json\n")
        print("RNACOMPETE_FREEZE_ACCEPTED", json.dumps({
            "experiments": len(master), "included": int(master.sequence_track_included.sum()),
            "unique_rna": len(sequence_table), "clusters": len(cluster_sizes),
            "folds": fold_loads, "elapsed_seconds": round(elapsed, 1)}), flush=True)
    except Exception as error:
        run_log.update({"status": "blocked", "end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "error": repr(error)})
        with open(os.path.join(out, "run_log.json"), "w") as handle:
            json.dump(run_log, handle, indent=2, sort_keys=True)
        raise


def verify_frozen(out):
    out = require_teams(out)
    with open(os.path.join(out, "split_manifest.json")) as handle:
        manifest = json.load(handle)
    assert manifest["status"] == "accepted" and manifest["model_results"] is False
    assert manifest["leakage_checks_passed"] is True
    with open(os.path.join(out, "SHA256SUMS")) as handle:
        lines = [line.rstrip("\n") for line in handle if line.strip()]
    names = []
    for line in lines:
        expected, name = line.split("  ", 1)
        assert sha256_file(os.path.join(out, name)) == expected, name
        names.append(name)
    assert "split_manifest.json" in names
    for name, expected in manifest["output_sha256"].items():
        assert sha256_file(os.path.join(out, name)) == expected, name
    sequence = pd.read_csv(os.path.join(out, "sequence_folds.tsv.gz"), sep="\t")
    probes = pd.read_csv(os.path.join(out, "probes.tsv.gz"), sep="\t",
                         usecols=["probe_id", "sequence_id", "rna_seq", "fold"])
    experiments = pd.read_csv(os.path.join(out, "experiments.tsv"), sep="\t")
    rbp_folds = pd.read_csv(os.path.join(out, "rbp_folds.tsv"), sep="\t")
    availability = pd.read_csv(os.path.join(out, "label_availability.tsv.gz"), sep="\t")
    leakage = pd.read_csv(os.path.join(out, "leakage_audit.tsv"), sep="\t")
    excluded = pd.read_csv(os.path.join(out, "excluded_probes.tsv"), sep="\t")
    assert len(experiments) == manifest["experiments_total"] == 244
    assert experiments.Motif_ID.is_unique and rbp_folds.Motif_ID.is_unique
    assert len(excluded) == 42 and excluded.exclusion_reason.value_counts().to_dict() == {
        "not_SetA_B": 42}
    assert sequence.sequence_id.is_unique and sequence.rna_seq.is_unique
    assert probes.probe_id.is_unique and len(probes) == 241357
    assert sequence.groupby("cluster_id").fold.nunique().max() == 1
    probe_sequence = probes.merge(sequence[["sequence_id", "rna_seq", "fold"]],
                                  on="sequence_id", suffixes=("", "_frozen"),
                                  how="left", validate="many_to_one")
    assert (probe_sequence.rna_seq == probe_sequence.rna_seq_frozen).all()
    assert (probe_sequence.fold == probe_sequence.fold_frozen).all()
    assert (availability.n_finite_unique + availability.n_all_nan_unique ==
            availability.n_unique_rna).all()
    all_counts = availability[availability.probe_set == "All"]
    set_counts = availability[availability.probe_set.isin(["SetA", "SetB"])].groupby(
        ["Motif_ID", "fold"])[["n_unique_rna", "n_finite_unique", "n_all_nan_unique"]].sum()
    indexed_all = all_counts.set_index(["Motif_ID", "fold"])
    for column in ("n_unique_rna", "n_finite_unique", "n_all_nan_unique"):
        assert (indexed_all[column] == set_counts[column]).all()
    observed = all_counts.groupby("Motif_ID").n_finite_unique.sum()
    expected_observed = experiments.set_index("Motif_ID").n_finite_unique
    assert (observed == expected_observed).all()
    for row in experiments[experiments.sequence_track_included].itertuples():
        counts = all_counts[all_counts.Motif_ID == row.Motif_ID].n_finite_unique
        assert len(counts) == 5 and counts.min() >= MIN_FINITE_PER_FOLD
    assert rbp_folds.groupby("species_gene").rbp_fold.nunique().max() == 1
    assert (leakage.n_exact_overlap.fillna(0) == 0).all()
    assert (leakage.n_cluster_overlap.fillna(0) == 0).all()
    assert (leakage.n_group_overlap == 0).all()
    for fold in range(N_FOLDS):
        roles = manifest["split_roles"][str(fold)]
        assert roles == split_roles(fold)
        frames = {role: sequence[sequence.fold.isin(folds)] for role, folds in roles.items()}
        groups = {role: set(rbp_folds.loc[rbp_folds.rbp_fold.isin(folds), "species_gene"])
                  for role, folds in roles.items()}
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            assert not (set(frames[left].rna_seq) & set(frames[right].rna_seq))
            assert not (set(frames[left].cluster_id) & set(frames[right].cluster_id))
            assert not (groups[left] & groups[right])
    print("RNACOMPETE_FROZEN_VERIFICATION_OK", json.dumps({
        "version": manifest["version"], "files": len(names),
        "experiments": len(experiments), "unique_rna": len(sequence),
        "clusters": int(sequence.cluster_id.nunique())}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/external_benchmarks/rnacompete/ray2013_nature12311")
    parser.add_argument("--out", default="/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/external_benchmarks/rnacompete/" + VERSION)
    parser.add_argument("--contract", default="/home/ProxyFunctionGap/预注册/RNAcompete_Ray2013_预注册数据合同_v1.0_2026-11-26.md")
    parser.add_argument("--code-commit", default="unknown")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--memory-mb", type=int, default=24000)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify_frozen(args.out)
    else:
        assert os.path.isfile(args.contract), args.contract
        freeze(args)


if __name__ == "__main__":
    main()
