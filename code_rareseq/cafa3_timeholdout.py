#!/usr/bin/env python3
import argparse
import collections
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

NAMESPACES = {"F": "MFO", "P": "BPO", "C": "CCO"}
OBO_NAMESPACES = {"molecular_function": "MFO", "biological_process": "BPO", "cellular_component": "CCO"}
ONTOLOGIES = ("MFO", "BPO", "CCO")
PREFIX = {"MFO": "mfo", "BPO": "bpo", "CCO": "cco"}
FREQUENCY_BUCKETS = ((0, 0, "0"), (1, 4, "1-4"), (5, 19, "5-19"), (20, 99, "20-99"), (100, 499, "100-499"), (500, None, ">=500"))
EXPECTED = {
    "MFO": {"leaf_rows": 1765, "NK": 454, "LK": 647},
    "BPO": {"leaf_rows": 3608, "NK": 849, "LK": 1296},
    "CCO": {"leaf_rows": 1800, "NK": 639, "LK": 458},
}
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def open_tsv(path, header):
    raw = open(path, "wb")
    zipped = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
    text = io.TextIOWrapper(zipped, encoding="utf-8", newline="")
    writer = csv.writer(text, delimiter="\t", lineterminator="\n")
    writer.writerow(header)
    return text, writer


def read_tsv(path):
    if hasattr(path, "read"):
        yield from csv.reader(path, delimiter="\t")
        return
    with open(path, "rt", encoding="utf-8", newline="") as handle:
        yield from csv.reader(handle, delimiter="\t")


def parse_fasta_text(lines):
    identifier = None
    description = ""
    sequence = []
    for raw in lines:
        line = raw.rstrip("\r\n")
        if line.startswith(">"):
            if identifier is not None:
                yield identifier, description, "".join(sequence)
            description = line[1:]
            identifier = description.split()[0]
            sequence = []
        elif identifier is not None:
            sequence.append("".join(line.split()).upper())
    if identifier is not None:
        yield identifier, description, "".join(sequence)


def parse_obo(path):
    terms = {}
    current = None
    is_term = False
    with open(path, "rt", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\r\n")
            if line.startswith("["):
                if is_term and current and "id" in current:
                    terms[current["id"]] = current
                is_term = line == "[Term]"
                current = {"alt_ids": [], "parents": set(), "is_obsolete": False, "namespace": ""} if is_term else None
            elif is_term and current is not None:
                if line.startswith("id: "):
                    current["id"] = line[4:].strip()
                elif line.startswith("alt_id: "):
                    current["alt_ids"].append(line[8:].strip())
                elif line.startswith("namespace: "):
                    current["namespace"] = line[11:].strip()
                elif line.startswith("is_obsolete: "):
                    current["is_obsolete"] = line[13:].strip() == "true"
                elif line.startswith("is_a: "):
                    current["parents"].add(line[6:].split()[0])
                elif line.startswith("relationship: part_of "):
                    current["parents"].add(line[22:].split()[0])
        if is_term and current and "id" in current:
            terms[current["id"]] = current
    alt_to_id = {}
    alt_collisions = set()
    for term_id, term in terms.items():
        for alt_id in term["alt_ids"]:
            if alt_id in alt_to_id and alt_to_id[alt_id] != term_id:
                alt_collisions.add(alt_id)
            else:
                alt_to_id[alt_id] = term_id
    for alt_id in alt_collisions:
        del alt_to_id[alt_id]
    for term in terms.values():
        term["namespace"] = OBO_NAMESPACES.get(term["namespace"], term["namespace"])
    return terms, alt_to_id, sorted(alt_collisions)


def canonical_term(go_id, namespace, terms, alt_to_id):
    if go_id in terms:
        term_id = go_id
        mapped = False
    elif go_id in alt_to_id:
        term_id = alt_to_id[go_id]
        mapped = True
    else:
        return None, "unknown_go", False
    term = terms[term_id]
    if term["is_obsolete"]:
        return None, "obsolete_go", mapped
    if term["namespace"] != namespace:
        return None, "namespace_mismatch", mapped
    return term_id, "accepted_alt_id" if mapped else "accepted", mapped


def ancestor_distances(term_id, terms):
    distances = {term_id: 0}
    queue = collections.deque([term_id])
    while queue:
        child = queue.popleft()
        for parent in terms[child]["parents"]:
            if parent not in terms or terms[parent]["is_obsolete"]:
                continue
            if terms[parent]["namespace"] != terms[child]["namespace"]:
                continue
            distance = distances[child] + 1
            if parent not in distances or distance < distances[parent]:
                distances[parent] = distance
                queue.append(parent)
    return distances


def frequency_bucket(count):
    for low, high, label in FREQUENCY_BUCKETS:
        if count >= low and (high is None or count <= high):
            return label
    raise ValueError(count)


def fold_for_group(group_id):
    return int(hashlib.sha256(group_id.encode("utf-8")).hexdigest()[:16], 16) % 3


def read_mapping(archive, name):
    mapping = {}
    with archive.extractfile(name) as handle:
        for row in read_tsv(io.TextIOWrapper(handle, encoding="utf-8")):
            if len(row) >= 2:
                mapping[row[0]] = row[1]
    return mapping


def load_inputs(root):
    training_archive = root / "CAFA3_training_data.tgz"
    target_archive = root / "CAFA3_targets.tgz"
    commit = "910af41cd1e8b7751a68061de2c3fe77cd7c6148"
    assessment = root / commit / "precrec"
    benchmark = assessment / "benchmark" / "CAFA3_benchmarks"
    training = {}
    with tarfile.open(training_archive, "r:gz") as archive:
        with archive.extractfile("CAFA3_training_data/uniprot_sprot_exp.fasta") as handle:
            for identifier, description, sequence in parse_fasta_text(io.TextIOWrapper(handle, encoding="utf-8")):
                if identifier in training:
                    raise ValueError(f"duplicate training ID: {identifier}")
                training[identifier] = {"sequence": sequence, "description": description}
        with archive.extractfile("CAFA3_training_data/uniprot_sprot_exp.txt") as handle:
            annotations = list(read_tsv(io.TextIOWrapper(handle, encoding="utf-8")))
    targets = {}
    mappings = {}
    with tarfile.open(target_archive, "r:gz") as archive:
        names = archive.getnames()
        for name in names:
            if name.startswith("Target files/") and name.endswith(".fasta"):
                taxonomy = "moonlight" if "moonlight" in name else name.rsplit(".", 2)[1]
                with archive.extractfile(name) as handle:
                    for identifier, description, sequence in parse_fasta_text(io.TextIOWrapper(handle, encoding="utf-8")):
                        if identifier in targets:
                            raise ValueError(f"duplicate target ID: {identifier}")
                        targets[identifier] = {"sequence": sequence, "description": description, "taxonomy": taxonomy}
        for name in names:
            if name.startswith("Mapping files/") and name.endswith(".map"):
                for identifier, external_id in read_mapping(archive, name).items():
                    if identifier in mappings and mappings[identifier] != external_id:
                        raise ValueError(f"conflicting target mapping: {identifier}")
                    mappings[identifier] = external_id
    terms, alt_to_id, alt_collisions = parse_obo(assessment / "go_cafa3.obo")
    leaf_rows = {}
    roles = {}
    for ontology in ONTOLOGIES:
        with open(benchmark / "groundtruth" / f"leafonly_{ontology}.txt", "rt", encoding="utf-8") as handle:
            leaf_rows[ontology] = list(read_tsv(handle))
        roles[ontology] = {}
        for role, kind in (("NK", "type1"), ("LK", "type2")):
            path = benchmark / "lists" / f"{PREFIX[ontology]}_all_{kind}.txt"
            identifiers = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(identifiers) != len(set(identifiers)):
                raise ValueError(f"duplicate role IDs: {path}")
            for identifier in identifiers:
                if identifier in roles[ontology]:
                    raise ValueError(f"NK/LK overlap: {ontology} {identifier}")
                roles[ontology][identifier] = role
    return training, annotations, targets, mappings, terms, alt_to_id, alt_collisions, leaf_rows, roles


def build(args):
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    if not str(output_root).startswith("/teams/"):
        raise ValueError("large CAFA outputs must be under /teams")
    output_root.mkdir(parents=True, exist_ok=True)
    training, annotations, targets, mappings, terms, alt_to_id, alt_collisions, leaf_rows, roles = load_inputs(input_root)
    summary = {
        "schema_version": "cafa3-timeholdout-summary-v1",
        "dataset_version": "cafa3_timeholdout_v1",
        "evidence_level": "data_and_protocol_only_no_model_results",
        "ontology": {},
        "folds": {},
        "homology": {},
    }
    summary["ontology"] = {
        "terms": len(terms),
        "alt_ids": len(alt_to_id),
        "ambiguous_alt_ids_excluded": alt_collisions,
        "propagation_edges": "is_a_and_part_of_only_same_namespace_non_obsolete",
    }
    clean_by_protein = collections.defaultdict(list)
    direct_counts = collections.Counter()
    propagated_counts = collections.Counter()
    rejects = collections.Counter()
    clean_seen = set()
    clean_path = output_root / "t0_annotations_direct_clean.tsv.gz"
    reject_path = output_root / "t0_annotations_rejected.tsv.gz"
    clean_handle, clean_writer = open_tsv(clean_path, ["protein_id", "go_id", "ontology", "source_go_id", "normalization"])
    reject_handle, reject_writer = open_tsv(reject_path, ["source_line", "protein_id", "source_go_id", "source_namespace", "reason"])
    for line_number, row in enumerate(annotations, 1):
        if len(row) != 3:
            rejects["malformed_row"] += 1
            reject_writer.writerow([line_number, *row, "malformed_row"])
            continue
        protein_id, source_go, short_namespace = row
        if protein_id not in training:
            reason = "missing_training_sequence"
        elif short_namespace not in NAMESPACES:
            reason = "unknown_namespace"
        else:
            ontology = NAMESPACES[short_namespace]
            go_id, reason, _ = canonical_term(source_go, ontology, terms, alt_to_id)
            if go_id is not None:
                key = (protein_id, go_id, ontology)
                if key in clean_seen:
                    reason = "duplicate_after_normalization"
                else:
                    clean_seen.add(key)
                    clean_by_protein[protein_id].append((go_id, ontology))
                    direct_counts[(ontology, go_id)] += 1
                    clean_writer.writerow([protein_id, go_id, ontology, source_go, reason])
                    continue
        rejects[reason] += 1
        reject_writer.writerow([line_number, protein_id, source_go, short_namespace, reason])
    clean_handle.close()
    reject_handle.close()
    for protein_id, rows in clean_by_protein.items():
        propagated = set()
        for go_id, ontology in rows:
            for ancestor in ancestor_distances(go_id, terms):
                propagated.add((ontology, ancestor))
        propagated_counts.update(propagated)
    summary["training_annotations"] = {
        "source_rows": len(annotations),
        "clean_unique_rows": len(clean_seen),
        "rejected_or_duplicate_rows": sum(rejects.values()),
        "reject_counts": dict(sorted(rejects.items())),
        "annotated_proteins": len(clean_by_protein),
        "evidence_codes": "organizer_prefiltered_experimental_no_per_row_codes_available",
    }
    edge_path = output_root / "ontology_edges.tsv.gz"
    edge_handle, edge_writer = open_tsv(edge_path, ["child_go_id", "parent_go_id", "ontology"])
    for term_id in sorted(terms):
        term = terms[term_id]
        if term["is_obsolete"] or term["namespace"] not in ONTOLOGIES:
            continue
        for parent in sorted(term["parents"]):
            if parent in terms and not terms[parent]["is_obsolete"] and terms[parent]["namespace"] == term["namespace"]:
                edge_writer.writerow([term_id, parent, term["namespace"]])
    edge_handle.close()
    ancestor_cache = {}
    def ancestors(term_id):
        if term_id not in ancestor_cache:
            ancestor_cache[term_id] = ancestor_distances(term_id, terms)
        return ancestor_cache[term_id]
    membership_path = output_root / "t1_evaluation_membership.tsv.gz"
    leaf_path = output_root / "t1_groundtruth_leaf.tsv.gz"
    propagated_path = output_root / "t1_groundtruth_propagated.tsv.gz"
    anomaly_path = output_root / "upstream_anomalies.tsv.gz"
    membership_handle, membership_writer = open_tsv(membership_path, ["protein_id", "ontology", "role", "has_official_leaf_truth", "has_sequence", "sequence_md5"])
    leaf_handle, leaf_writer = open_tsv(leaf_path, ["protein_id", "ontology", "role", "source_go_id", "evaluation_go_id", "canonical_go_id", "status", "t0_direct_frequency", "t0_propagated_frequency", "frequency_bucket"])
    propagated_handle, propagated_writer = open_tsv(propagated_path, ["protein_id", "ontology", "role", "source_leaf_go_id", "ancestor_go_id", "min_distance"])
    anomaly_handle, anomaly_writer = open_tsv(anomaly_path, ["anomaly_type", "protein_id", "ontology", "role", "detail"])
    leaf_by_protein = {ontology: collections.defaultdict(list) for ontology in ONTOLOGIES}
    t1_stats = {}
    for ontology in ONTOLOGIES:
        status_counts = collections.Counter()
        seen_leaf = set()
        for row in leaf_rows[ontology]:
            if len(row) != 2:
                raise ValueError(f"malformed official leaf row: {ontology} {row}")
            protein_id, source_go = row
            role = roles[ontology].get(protein_id, "UNLISTED")
            go_id, ontology_status, _ = canonical_term(source_go, ontology, terms, alt_to_id)
            key = (protein_id, source_go)
            if key in seen_leaf:
                status = "duplicate_official_protein_go"
                status_counts[status] += 1
                leaf_writer.writerow([protein_id, ontology, role, source_go, source_go, go_id or "", status, "", "", ""])
                continue
            seen_leaf.add(key)
            if go_id is None:
                status = f"official_leaf_{ontology_status}_self_only"
                propagation = {source_go: 0}
                canonical_id = ""
                anomaly_writer.writerow(["leaf_ontology_limitation", protein_id, ontology, role, f"{source_go}\t{status}\tlabel_retained"])
            else:
                status = ontology_status
                propagation = ancestors(go_id)
                canonical_id = go_id
                if go_id != source_go:
                    propagation[source_go] = 0
            status_counts[status] += 1
            leaf_by_protein[ontology][protein_id].append((source_go, source_go))
            direct = direct_counts[(ontology, source_go)]
            propagated = propagated_counts[(ontology, source_go)]
            leaf_writer.writerow([protein_id, ontology, role, source_go, source_go, canonical_id, status, direct, propagated, frequency_bucket(direct)])
            for ancestor, distance in sorted(propagation.items()):
                propagated_writer.writerow([protein_id, ontology, role, source_go, ancestor, distance])
        for protein_id, role in sorted(roles[ontology].items()):
            has_truth = protein_id in leaf_by_protein[ontology]
            has_sequence = protein_id in targets
            md5 = hashlib.md5(targets[protein_id]["sequence"].encode()).hexdigest() if has_sequence else ""
            membership_writer.writerow([protein_id, ontology, role, int(has_truth), int(has_sequence), md5])
            if not has_truth:
                anomaly_writer.writerow(["role_without_leaf_truth", protein_id, ontology, role, "retained_in_official_evaluation_membership"])
            if not has_sequence:
                anomaly_writer.writerow(["role_without_target_sequence", protein_id, ontology, role, "retained_in_official_evaluation_membership"])
        for protein_id in sorted(leaf_by_protein[ontology]):
            if protein_id not in roles[ontology]:
                anomaly_writer.writerow(["leaf_truth_without_role", protein_id, ontology, "UNLISTED", "not_assigned_to_NK_or_LK"])
        actual = {
            "leaf_rows": len(leaf_rows[ontology]),
            "unique_official_leaf_pairs": len(seen_leaf),
            "NK": sum(role == "NK" for role in roles[ontology].values()),
            "LK": sum(role == "LK" for role in roles[ontology].values()),
        }
        for key, expected in EXPECTED[ontology].items():
            if actual[key] != expected:
                raise ValueError(f"source regression: {ontology} {key}={actual[key]}, expected {expected}")
        actual["leaf_status_counts"] = dict(sorted(status_counts.items()))
        actual["evaluation_proteins"] = len(roles[ontology])
        actual["leaf_truth_proteins_including_unlisted"] = len(leaf_by_protein[ontology])
        actual["role_proteins_with_leaf_truth"] = sum(protein_id in leaf_by_protein[ontology] for protein_id in roles[ontology])
        actual["roles_without_leaf_truth"] = sorted(set(roles[ontology]) - set(leaf_by_protein[ontology]))
        actual["leaf_truth_proteins_without_role"] = sorted(set(leaf_by_protein[ontology]) - set(roles[ontology]))
        t1_stats[ontology] = actual
    membership_handle.close()
    leaf_handle.close()
    propagated_handle.close()
    anomaly_handle.close()
    summary["t1"] = t1_stats
    summary["bpo_nk_anomaly"] = {
        "protein_id": "T2089630000462",
        "present_in_official_nk_list": "T2089630000462" in roles["BPO"] and roles["BPO"]["T2089630000462"] == "NK",
        "present_in_official_leaf_truth": "T2089630000462" in leaf_by_protein["BPO"],
        "policy": "retain_membership_and_report_zero_leaf_pairs_never_impute_or_silently_delete",
    }
    if not summary["bpo_nk_anomaly"]["present_in_official_nk_list"] or summary["bpo_nk_anomaly"]["present_in_official_leaf_truth"]:
        raise ValueError("expected BPO NK upstream anomaly changed")
    if len(training) != 66841 or len(annotations) != 386197 or len(targets) != 130827:
        raise ValueError("fixed CAFA3 source size regression")
    training_md5 = {identifier: hashlib.md5(record["sequence"].encode()).hexdigest() for identifier, record in training.items()}
    target_md5 = {identifier: hashlib.md5(record["sequence"].encode()).hexdigest() for identifier, record in targets.items()}
    t0_fasta = output_root / "t0.fasta"
    t0_protein_handle, t0_protein_writer = open_tsv(output_root / "t0_proteins.tsv.gz", ["uniprot_accession", "description", "sequence_md5", "sequence_length", "nonstandard_aa_count"])
    with open(t0_fasta, "wt", encoding="utf-8") as handle:
        for identifier in sorted(training):
            sequence = training[identifier]["sequence"]
            t0_protein_writer.writerow([identifier, training[identifier]["description"], training_md5[identifier], len(sequence), sum(aa not in STANDARD_AA for aa in sequence)])
            handle.write(f">{identifier}\n")
            for offset in range(0, len(sequence), 80):
                handle.write(sequence[offset:offset + 80] + "\n")
    t0_protein_handle.close()
    target_handle, target_writer = open_tsv(output_root / "target_proteins.tsv.gz", ["cafa_target_id", "external_id", "taxonomy", "description", "sequence_md5", "sequence_length", "nonstandard_aa_count"])
    for identifier in sorted(targets):
        sequence = targets[identifier]["sequence"]
        target_writer.writerow([identifier, mappings.get(identifier, ""), targets[identifier]["taxonomy"], targets[identifier]["description"], target_md5[identifier], len(sequence), sum(aa not in STANDARD_AA for aa in sequence)])
    target_handle.close()
    benchmark_ids = sorted(set().union(*(set(roles[ontology]) for ontology in ONTOLOGIES)))
    t1_fasta = output_root / "t1_benchmark.fasta"
    with open(t1_fasta, "wt", encoding="utf-8") as handle:
        for identifier in benchmark_ids:
            if identifier not in targets:
                continue
            sequence = targets[identifier]["sequence"]
            handle.write(f">{identifier}\n")
            for offset in range(0, len(sequence), 80):
                handle.write(sequence[offset:offset + 80] + "\n")
    search_path = output_root / "t1_vs_t0_blastp.tsv"
    homology_stats = {"status": "not_run", "coverage_definition": "min(alnlen/qlen,alnlen/tlen)", "identity_thresholds": [30, 50, 90], "stress_coverage_threshold": 0.5}
    clusters = {identifier: f"exact:{training_md5[identifier]}" for identifier in training}
    homology_hits = collections.defaultdict(list)
    if args.blastp and args.makeblastdb:
        blast_version = subprocess.run([args.blastp, "-version"], check=True, text=True, capture_output=True).stdout.strip().splitlines()[0]
        work = output_root / "homology_work"
        work.mkdir(exist_ok=True)
        subprocess.run([args.makeblastdb, "-in", str(t0_fasta), "-dbtype", "prot", "-out", str(work / "t0_blast"), "-parse_seqids"], check=True)
        subprocess.run([args.blastp, "-query", str(t1_fasta), "-db", str(work / "t0_blast"), "-out", str(search_path), "-evalue", "10", "-max_target_seqs", "1000", "-num_threads", str(args.threads), "-outfmt", "6 qseqid sseqid pident length qlen slen evalue bitscore"], check=True)
        with open(search_path, "rt", encoding="utf-8") as handle:
            for row in read_tsv(handle):
                if len(row) != 8:
                    raise ValueError(f"unexpected BLASTP row: {row}")
                query, target, identity, alnlen, qlen, tlen, evalue, bits = row
                coverage = min(int(alnlen) / int(qlen), int(alnlen) / int(tlen))
                homology_hits[query].append((target, float(identity), coverage, float(evalue), float(bits)))
        homology_stats.update({"status": "completed", "blastp_version": blast_version, "blastp_binary_sha256": sha256_file(args.blastp), "makeblastdb_binary_sha256": sha256_file(args.makeblastdb), "fold_cluster_definition": "exact_sequence_MD5_only_no_claim_of_t0_all_vs_all_homology_clustering", "search_definition": "t1_vs_all_t0_BLASTP_evalue_10_max_target_seqs_1000"})
    else:
        if args.require_homology:
            raise ValueError("--require-homology needs --blastp and --makeblastdb")
    for identifier in training:
        clusters.setdefault(identifier, f"exact:{training_md5[identifier]}")
    folds = {identifier: fold_for_group(clusters[identifier]) for identifier in training}
    split_path = output_root / "t0_strict3fold_assignments.tsv.gz"
    split_handle, split_writer = open_tsv(split_path, ["protein_id", "sequence_md5", "homology_group_id", "fold", "sequence_length", "nonstandard_aa_count"])
    for identifier in sorted(training):
        sequence = training[identifier]["sequence"]
        split_writer.writerow([identifier, training_md5[identifier], clusters[identifier], folds[identifier], len(sequence), sum(aa not in STANDARD_AA for aa in sequence)])
    split_handle.close()
    for fold in range(3):
        for partition in ("train", "validation"):
            selected = [identifier for identifier in sorted(training) if (folds[identifier] != fold) == (partition == "train")]
            path = output_root / f"fold{fold}_{partition}_whitelist.txt.gz"
            handle, writer = open_tsv(path, ["protein_id"])
            for identifier in selected:
                writer.writerow([identifier])
            handle.close()
        fold_rows = [identifier for identifier in training if folds[identifier] == fold]
        summary["folds"][str(fold)] = {
            "validation_proteins": len(fold_rows),
            "train_proteins": len(training) - len(fold_rows),
            "validation_annotations": sum(len(clean_by_protein[identifier]) for identifier in fold_rows),
            "train_annotations": len(clean_seen) - sum(len(clean_by_protein[identifier]) for identifier in fold_rows),
        }
    if len(set(folds.values())) != 3 or len({clusters[identifier] for identifier in training if folds[identifier] == 0} & {clusters[identifier] for identifier in training if folds[identifier] == 1}) or len({clusters[identifier] for identifier in training if folds[identifier] == 0} & {clusters[identifier] for identifier in training if folds[identifier] == 2}) or len({clusters[identifier] for identifier in training if folds[identifier] == 1} & {clusters[identifier] for identifier in training if folds[identifier] == 2}):
        raise ValueError("strict three-fold group exclusivity failed")
    sequence_members = collections.defaultdict(list)
    for source, identifiers in (("t0", training), ("target", targets)):
        for identifier in identifiers:
            md5 = training_md5[identifier] if source == "t0" else target_md5[identifier]
            sequence_members[md5].append((source, identifier))
    duplicate_path = output_root / "exact_sequence_groups.tsv.gz"
    duplicate_handle, duplicate_writer = open_tsv(duplicate_path, ["sequence_md5", "group_size", "source", "protein_id", "fold", "ontology_roles"])
    cross_t0_t1_groups = 0
    cross_fold_groups = 0
    for md5, members in sorted(sequence_members.items()):
        sources = {source for source, _ in members}
        member_folds = {folds[identifier] for source, identifier in members if source == "t0"}
        if len(member_folds) > 1:
            cross_fold_groups += 1
        benchmark_members = [identifier for source, identifier in members if identifier in benchmark_ids]
        if "t0" in sources and benchmark_members:
            cross_t0_t1_groups += 1
        if len(members) > 1 or benchmark_members:
            for source, identifier in sorted(members):
                ontology_roles = ";".join(f"{ontology}:{roles[ontology][identifier]}" for ontology in ONTOLOGIES if identifier in roles[ontology])
                duplicate_writer.writerow([md5, len(members), source, identifier, folds.get(identifier, ""), ontology_roles])
    duplicate_handle.close()
    if cross_fold_groups:
        raise ValueError("exact sequence appears in multiple t0 folds")
    summary["sequences"] = {
        "t0_proteins": len(training),
        "target_proteins": len(targets),
        "benchmark_proteins": len(benchmark_ids),
        "t0_unique_sequences": len(set(training_md5.values())),
        "target_unique_sequences": len(set(target_md5.values())),
        "exact_groups_spanning_t0_and_benchmark": cross_t0_t1_groups,
        "exact_groups_crossing_t0_folds": cross_fold_groups,
        "t0_empty_sequences": sum(not record["sequence"] for record in training.values()),
        "target_empty_sequences": sum(not record["sequence"] for record in targets.values()),
        "t0_records_with_nonstandard_aa": sum(any(aa not in STANDARD_AA for aa in record["sequence"]) for record in training.values()),
        "target_records_with_nonstandard_aa": sum(any(aa not in STANDARD_AA for aa in record["sequence"]) for record in targets.values()),
        "target_mapped_external_ids": len(mappings),
        "target_mapping_note": "official_external_ID_preserved_not_assumed_to_be_UniProt_accession",
    }
    homology_path = output_root / "t1_to_t0_homology.tsv.gz"
    homology_handle, homology_writer = open_tsv(homology_path, ["protein_id", "best_t0_id", "max_identity_percent", "bilateral_coverage", "evalue", "bits", "identity_bucket", "exact_sequence_overlap"])
    excluded = {threshold: set() for threshold in (30, 50, 90)}
    bucket_counts = collections.Counter()
    role_bucket_counts = collections.Counter()
    role_exact_counts = collections.Counter()
    t0_by_md5 = collections.defaultdict(list)
    for identifier, md5 in training_md5.items():
        t0_by_md5[md5].append(identifier)
    for identifier in benchmark_ids:
        candidates = [hit for hit in homology_hits.get(identifier, []) if hit[2] >= 0.5]
        exact_targets = t0_by_md5.get(target_md5.get(identifier, ""), [])
        best = max(candidates, key=lambda hit: (hit[1], hit[2], hit[4], hit[0]), default=None)
        if best is None and exact_targets:
            best = (sorted(exact_targets)[0], 100.0, 1.0, 0.0, 0.0)
        if best is None:
            bucket = "no_detected_hit"
            row = [identifier, "", "", "", "", "", bucket, int(bool(exact_targets))]
            bucket_counts[bucket] += 1
        else:
            target, identity, coverage, evalue, bits = best
            if identity == 100:
                bucket = "100"
            elif identity >= 90:
                bucket = "90-<100"
            elif identity >= 50:
                bucket = "50-<90"
            elif identity >= 30:
                bucket = "30-<50"
            else:
                bucket = "20-<30"
            bucket_counts[bucket] += 1
            row = [identifier, target, f"{identity:.4f}", f"{coverage:.6f}", evalue, bits, bucket, int(bool(exact_targets))]
        homology_writer.writerow(row)
        for ontology in ONTOLOGIES:
            if identifier in roles[ontology]:
                role = roles[ontology][identifier]
                role_bucket_counts[(ontology, role, bucket)] += 1
                role_exact_counts[(ontology, role, bool(exact_targets))] += 1
        for target, identity, coverage, _, _ in homology_hits.get(identifier, []):
            if coverage < 0.5:
                continue
            for threshold in excluded:
                if identity >= threshold:
                    excluded[threshold].add(target)
        if exact_targets:
            for threshold in excluded:
                excluded[threshold].update(exact_targets)
    homology_handle.close()
    homology_stats["benchmark_identity_buckets"] = dict(sorted(bucket_counts.items()))
    homology_stats["role_identity_buckets"] = {f"{ontology}:{role}": {bucket: count for (o, r, bucket), count in sorted(role_bucket_counts.items()) if o == ontology and r == role} for ontology in ONTOLOGIES for role in ("NK", "LK")}
    homology_stats["role_exact_sequence_overlap"] = {f"{ontology}:{role}": {str(exact).lower(): count for (o, r, exact), count in sorted(role_exact_counts.items(), key=lambda item: str(item[0])) if o == ontology and r == role} for ontology in ONTOLOGIES for role in ("NK", "LK")}
    homology_stats["approximation_boundary"] = "BLASTP_max_target_seqs_1000_and_heuristic_search_make_this_a_reproducible_approximate_audit_not_a_proof_of_absence_of_homology"
    homology_stats["stress_train_exclusions"] = {}
    for threshold, identifiers in excluded.items():
        homology_stats["stress_train_exclusions"][str(threshold)] = len(identifiers)
        for fold in range(3):
            for partition in ("train", "validation"):
                selected = [identifier for identifier in sorted(training) if ((folds[identifier] != fold) == (partition == "train")) and identifier not in identifiers]
                path = output_root / f"stress_id{threshold}_fold{fold}_{partition}_whitelist.txt.gz"
                handle, writer = open_tsv(path, ["protein_id"])
                for identifier in selected:
                    writer.writerow([identifier])
                handle.close()
    summary["homology"] = homology_stats
    term_path = output_root / "term_frequencies.tsv.gz"
    term_handle, term_writer = open_tsv(term_path, ["go_id", "ontology", "ontology_status", "t0_direct_protein_frequency", "t0_propagated_protein_frequency", "frequency_bucket", "t1_leaf_positive_count_NK", "t1_leaf_positive_count_LK"])
    for ontology in ONTOLOGIES:
        nk_counts = collections.Counter()
        lk_counts = collections.Counter()
        for protein_id, rows in leaf_by_protein[ontology].items():
            for _, go_id in rows:
                if roles[ontology].get(protein_id) == "NK":
                    nk_counts[go_id] += 1
                elif roles[ontology].get(protein_id) == "LK":
                    lk_counts[go_id] += 1
        ontology_terms = sorted(term_id for term_id, term in terms.items() if term["namespace"] == ontology and not term["is_obsolete"])
        for go_id in ontology_terms:
            direct = direct_counts[(ontology, go_id)]
            term_writer.writerow([go_id, ontology, "in_historical_obo", direct, propagated_counts[(ontology, go_id)], frequency_bucket(direct), nk_counts[go_id], lk_counts[go_id]])
        extra_eval_terms = sorted((set(nk_counts) | set(lk_counts)) - set(ontology_terms))
        for go_id in extra_eval_terms:
            direct = direct_counts[(ontology, go_id)]
            term_writer.writerow([go_id, ontology, "official_t1_leaf_not_propagatable_in_historical_obo", direct, propagated_counts[(ontology, go_id)], frequency_bucket(direct), nk_counts[go_id], lk_counts[go_id]])
    term_handle.close()
    frequency_summary = {}
    for ontology in ONTOLOGIES:
        counts = collections.Counter()
        for rows in leaf_by_protein[ontology].values():
            for _, go_id in rows:
                counts[frequency_bucket(direct_counts[(ontology, go_id)])] += 1
        frequency_summary[ontology] = dict(sorted(counts.items()))
    summary["t1_leaf_rows_by_t0_direct_frequency_bucket"] = frequency_summary
    summary["evaluation_contract"] = {
        "primary": "CAFA_protein_centric_Fmax_and_Smin_over_thresholds_0.01_to_1.00_threshold_selected_on_t0_validation_only",
        "macro_auprc": "unweighted_mean_over_labels_with_at_least_one_positive_in_the_reported_slice_report_label_count_never_impute_undefined_to_zero",
        "frequency_buckets": [label for _, _, label in FREQUENCY_BUCKETS],
        "ood_slices": "ontology_x_NK_LK_x_frequency_bucket_x_exact_overlap_x_homology_bucket_with_coverage_and_evaluable_protein_counts",
        "negative_semantics": "absence_of_annotation_is_not_experimental_negative",
        "test_use": "t1_only_after_model_and_threshold_freeze_no_test_selection_unlisted_groundtruth_rows_audited_but_never_assigned_to_NK_or_LK",
        "validation": "strict_three_fold_group_exclusive_t0_proxy_validation_not_a_second_time_split",
        "future_model_requirements": "explicit_CUDA_no_CPU_fallback_patience_20",
        "bpo_anomaly": "official_NK_membership_remains_849_protein_centric_metrics_use_only_848_with_leaf_truth_and_report_coverage_848_of_849_never_synthesize_an_all_zero_truth",
        "grimm_boundary": "GRIMM_EC_v2_is_a_different_task_and_never_a_CAFA3_substitute",
    }
    summary["cleaning_contract"] = {
        "training_filter": "retain_only_wellformed_organizer_experimental_rows_with_training_sequence_and_valid_same_namespace_nonobsolete_GO_alt_IDs_mapped_once_duplicates_audited",
        "leaf_truth": "all_official_leaf_rows_preserved_exact_protein_GO_duplicates_audited_evaluation_labels_never_filtered_by_OBO_unknown_or_nonpropagatable_terms_are_self_only_no_current_GO_or_GOA_backfill",
        "sequence_filter": "no_sequence_record_dropped_empty_and_nonstandard_sequences_counted",
        "role_semantics": "NK_type1_no_prior_experimental_knowledge_in_that_ontology_LK_type2_partial_prior_knowledge_LK_t0_rows_remain_in_faithful_training",
    }
    summary["artifact_policy"] = "large_inputs_outputs_only_under_teams_repository_contains_code_small_summary_and_frozen_manifest"
    input_paths = [input_root / "CAFA3_training_data.tgz", input_root / "CAFA3_targets.tgz", input_root / "SHA256SUMS"]
    input_paths.extend(sorted((input_root / "910af41cd1e8b7751a68061de2c3fe77cd7c6148").rglob("*")))
    input_records = [{"path": str(path.relative_to(input_root)), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in input_paths if path.is_file()]
    write_json(output_root / "cleaning_summary.json", summary)
    outputs = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and not {"mmseqs_work", "homology_work"}.intersection(path.parts) and path.name not in {"frozen_manifest.json", "SHA256SUMS"}:
            outputs.append({"path": str(path.relative_to(output_root)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = {
        "schema_version": "cafa3-frozen-manifest-v1",
        "dataset_version": summary["dataset_version"],
        "freeze_date": args.freeze_date,
        "code": {"path": "code_rareseq/cafa3_timeholdout.py", "sha256": sha256_file(Path(__file__).resolve())},
        "input_root": str(input_root),
        "output_root": str(output_root),
        "inputs": input_records,
        "outputs": outputs,
        "summary_sha256": sha256_file(output_root / "cleaning_summary.json"),
        "model_results": False,
    }
    write_json(output_root / "frozen_manifest.json", manifest)
    checksums = "".join(f"{record['sha256']}  {record['path']}\n" for record in outputs)
    checksums += f"{sha256_file(output_root / 'frozen_manifest.json')}  frozen_manifest.json\n"
    (output_root / "SHA256SUMS").write_text(checksums, encoding="utf-8")
    print(json.dumps({"status": "completed", "output_root": str(output_root), "outputs": len(outputs), "training_rows": len(clean_seen), "benchmark_proteins": len(benchmark_ids)}, indent=2))


def verify(args):
    root = Path(args.output_root).resolve()
    manifest = json.loads((root / "frozen_manifest.json").read_text(encoding="utf-8"))
    failures = []
    for group, base in (("inputs", Path(manifest["input_root"])), ("outputs", root)):
        for record in manifest[group]:
            path = base / record["path"]
            if not path.is_file() or path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
                failures.append(f"{group}:{record['path']}")
    summary = json.loads((root / "cleaning_summary.json").read_text(encoding="utf-8"))
    if summary["bpo_nk_anomaly"]["present_in_official_nk_list"] is not True or summary["bpo_nk_anomaly"]["present_in_official_leaf_truth"] is not False:
        failures.append("BPO_NK_anomaly_policy")
    if summary["sequences"]["exact_groups_crossing_t0_folds"] != 0:
        failures.append("exact_sequence_fold_leakage")
    if summary["homology"]["status"] != "completed" and args.require_homology:
        failures.append("homology_not_completed")
    with gzip.open(root / "t0_strict3fold_assignments.tsv.gz", "rt", encoding="utf-8", newline="") as handle:
        assignments = list(csv.DictReader(handle, delimiter="\t"))
    identifiers = {row["protein_id"] for row in assignments}
    folds = {row["protein_id"]: int(row["fold"]) for row in assignments}
    group_folds = collections.defaultdict(set)
    md5_folds = collections.defaultdict(set)
    for row in assignments:
        group_folds[row["homology_group_id"]].add(int(row["fold"]))
        md5_folds[row["sequence_md5"]].add(int(row["fold"]))
    if len(identifiers) != 66841 or len(assignments) != len(identifiers) or set(folds.values()) != {0, 1, 2}:
        failures.append("assignment_ID_or_fold_coverage")
    if any(len(values) != 1 for values in group_folds.values()):
        failures.append("homology_group_fold_leakage")
    if any(len(values) != 1 for values in md5_folds.values()):
        failures.append("md5_fold_leakage")
    checked_whitelists = 0
    for fold in range(3):
        partition_ids = {}
        for partition in ("train", "validation"):
            with gzip.open(root / f"fold{fold}_{partition}_whitelist.txt.gz", "rt", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            partition_ids[partition] = {row["protein_id"] for row in rows}
            if len(rows) != len(partition_ids[partition]):
                failures.append(f"duplicate_whitelist_ID_fold{fold}_{partition}")
            expected = {identifier for identifier, value in folds.items() if (value != fold) == (partition == "train")}
            if partition_ids[partition] != expected:
                failures.append(f"whitelist_contract_fold{fold}_{partition}")
            checked_whitelists += 1
        if partition_ids["train"] & partition_ids["validation"] or partition_ids["train"] | partition_ids["validation"] != identifiers:
            failures.append(f"partition_overlap_or_incomplete_fold{fold}")
    if summary["t1"]["BPO"]["role_proteins_with_leaf_truth"] != 2144 or summary["t1"]["BPO"]["leaf_truth_proteins_without_role"] != ["T2089630003090"]:
        failures.append("BPO_role_coverage_or_unlisted_groundtruth")
    result = {"status": "passed" if not failures else "failed", "checked_inputs": len(manifest["inputs"]), "checked_outputs": len(manifest["outputs"]), "checked_whitelists": checked_whitelists, "checked_assignment_groups": len(group_folds), "failures": failures, "model_results": False}
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description="Freeze CAFA3 t0/t1 time-holdout data and audit artifacts; no model training or evaluation.")
    parser.add_argument("--input-root", default="/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/external_benchmarks/cafa/cafa3_2016_2017")
    parser.add_argument("--output-root", default="/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/external_benchmarks/cafa/cafa3_timeholdout_v1")
    parser.add_argument("--blastp")
    parser.add_argument("--makeblastdb")
    parser.add_argument("--require-homology", action="store_true")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--freeze-date", default="2026-11-26")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify(args)
    else:
        build(args)


if __name__ == "__main__":
    main()
