#!/usr/bin/env python3
"""CAFA3 frozen-ESMC multilabel BCE and deep-kNN baselines."""
import argparse
import csv
import gzip
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit
from sklearn.metrics import average_precision_score

FREQUENCY_BUCKETS = ["0", "1-4", "5-19", "20-99", "100-499", ">=500"]
THRESHOLDS = np.round(np.arange(0.01, 1.001, 0.01), 2)


def read_tsv(path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def read_whitelist(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        identifiers = [line.strip() for line in handle if line.strip()]
    if identifiers and identifiers[0] == "protein_id":
        identifiers = identifiers[1:]
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError(f"Duplicate identifiers in {path}")
    return identifiers


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frequency_bucket(count):
    if count == 0:
        return "0"
    if count <= 4:
        return "1-4"
    if count <= 19:
        return "5-19"
    if count <= 99:
        return "20-99"
    if count <= 499:
        return "100-499"
    return ">=500"


def sparse_matrix(rows, lookup, shape):
    row_index, column_index = [], []
    for i, terms in enumerate(rows):
        for term in terms:
            j = lookup.get(term)
            if j is not None:
                row_index.append(i)
                column_index.append(j)
    matrix = sparse.csr_matrix((np.ones(len(row_index), dtype=np.uint8), (row_index, column_index)),
                               shape=shape, dtype=np.uint8)
    matrix.sum_duplicates()
    matrix.data[:] = 1
    return matrix


def ancestor_closure(edges):
    cache = {}

    def ancestors(term, visiting=frozenset()):
        if term in cache:
            return cache[term]
        if term in visiting:
            raise ValueError(f"Cycle in frozen ontology at {term}")
        result = {term}
        for parent in edges.get(term, ()):
            result.update(ancestors(parent, visiting | {term}))
        cache[term] = result
        return result

    return ancestors


def metric_at_threshold(y, probability, information_content, threshold, batch=128):
    if probability.shape != y.shape or not np.isfinite(threshold):
        raise ValueError("Misaligned metric arrays or invalid threshold")
    f_values, coverage, remaining, misinformation = [], [], [], []
    for start in range(0, y.shape[0], batch):
        target = y[start:start + batch].toarray().astype(np.float32)
        scores = np.asarray(probability[start:start + batch], dtype=np.float32)
        if not np.isfinite(scores).all():
            raise ValueError("NaN/Inf probability encountered")
        pred = scores >= threshold
        n_pred = pred.sum(axis=1, dtype=np.float64)
        n_true = target.sum(axis=1, dtype=np.float64)
        tp = (pred * target).sum(axis=1, dtype=np.float64)
        denominator = n_pred + n_true
        f_values.append(np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0))
        coverage.append(n_pred > 0)
        true_ic = target * information_content
        remaining.append(true_ic.sum(axis=1) - (pred * true_ic).sum(axis=1))
        misinformation.append((pred * (1 - target) * information_content).sum(axis=1))
    if not f_values:
        raise ValueError("Cannot evaluate an empty protein slice")
    ru = float(np.concatenate(remaining).mean())
    mi = float(np.concatenate(misinformation).mean())
    return {"fmax_at_val_threshold": float(np.concatenate(f_values).mean()),
            "coverage": float(np.concatenate(coverage).mean()), "ru": ru, "mi": mi,
            "smin_at_val_threshold": float(np.sqrt(ru * ru + mi * mi))}


def validation_thresholds(y, probability, information_content):
    rows = []
    for threshold in THRESHOLDS:
        rows.append({"threshold": float(threshold), **metric_at_threshold(y, probability, information_content, threshold)})
    frame = pd.DataFrame(rows)
    fmax_index = frame.fmax_at_val_threshold.idxmax()
    smin_index = frame.smin_at_val_threshold.idxmin()
    return (float(frame.loc[fmax_index, "threshold"]), float(frame.loc[smin_index, "threshold"]), frame)


def macro_auprc(y, probability, direct_frequency, active=None, tau=0.0, log_prior=None,
                logits=False, return_table=False):
    if y.shape != probability.shape:
        raise ValueError("AUPRC labels and scores are misaligned")
    rows = []
    for j in range(y.shape[1]):
        target = y[:, j].toarray().ravel()
        positives = int(target.sum())
        if positives == 0:
            ap = np.nan
        else:
            if logits and (active is None or not active[j]):
                score = np.zeros(y.shape[0], dtype=np.float32)
            else:
                score = np.asarray(probability[:, j]).ravel().astype(np.float32)
                if logits:
                    if tau and log_prior is None:
                        raise ValueError("log_prior is required for nonzero tau")
                    score = expit(score if log_prior is None else score - tau * log_prior[j])
            if not np.isfinite(score).all():
                raise ValueError("NaN/Inf AUPRC score encountered")
            ap = float(average_precision_score(target, score))
        rows.append({"label_index": j, "n_positive": positives, "auprc": ap,
                     "frequency_bucket": frequency_bucket(int(direct_frequency[j]))})
    table = pd.DataFrame(rows)
    values = {"overall": (float(table.auprc.mean()), int(table.auprc.notna().sum()))}
    for bucket in FREQUENCY_BUCKETS:
        selected = table[table.frequency_bucket == bucket].auprc
        values[bucket] = (float(selected.mean()), int(selected.notna().sum()))
    return (values, table) if return_table else values


def write_metrics(handle, model, seed, split, ontology, role, slice_type, slice_value,
                  protein_metrics, auprc, fmax_threshold, smin_threshold, n_proteins, official_count):
    for bucket in ["overall"] + FREQUENCY_BUCKETS:
        macro, n_labels = auprc[bucket]
        row = {"model": model, "seed": seed, "split": split, "ontology": ontology, "role": role,
               "slice_type": slice_type, "slice_value": slice_value, "frequency_bucket": bucket,
               "n_proteins": n_proteins, "official_role_proteins": official_count,
               "macro_auprc": macro, "n_labels": n_labels,
               "fmax_threshold": fmax_threshold, "smin_threshold": smin_threshold}
        if bucket == "overall":
            row.update(protein_metrics)
        handle.writerow(row)


def subset_rows(matrix, indices):
    return matrix[indices] if len(indices) else matrix[:0]


class Cafa3Run:
    def __init__(self, args):
        self.args = args
        self.data = Path(args.data_dir)
        self.emb = Path(args.emb_dir)
        self.out = Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.ontology = args.ontology
        self.fold = args.fold
        self.track = args.track
        manifest = json.loads((self.data / "frozen_manifest.json").read_text(encoding="utf-8"))
        frozen_outputs = {item["path"]: item["sha256"] for item in manifest["outputs"]}
        prefix = "" if self.track == "main" else f"stress_{self.track}_"
        required = ["t0_annotations_direct_clean.tsv.gz", "ontology_edges.tsv.gz", "term_frequencies.tsv.gz",
                    "t1_groundtruth_propagated.tsv.gz", "t1_evaluation_membership.tsv.gz",
                    "t1_to_t0_homology.tsv.gz", f"{prefix}fold{self.fold}_train_whitelist.txt.gz",
                    f"{prefix}fold{self.fold}_validation_whitelist.txt.gz"]
        for name in required:
            if sha256_file(self.data / name) != frozen_outputs[name]:
                raise ValueError(f"Frozen input SHA mismatch: {name}")
        self.t0_ids = (self.emb / "t0_ids.tsv").read_text(encoding="utf-8").splitlines()
        self.t1_ids = (self.emb / "t1_ids.tsv").read_text(encoding="utf-8").splitlines()
        self.t0_lookup = {value: i for i, value in enumerate(self.t0_ids)}
        self.t1_lookup = {value: i for i, value in enumerate(self.t1_ids)}
        self.x = {"t0": np.load(self.emb / "t0.npy", mmap_mode="r"),
                  "t1": np.load(self.emb / "t1.npy", mmap_mode="r")}
        if self.x["t0"].shape != (len(self.t0_ids), 960) or self.x["t1"].shape != (len(self.t1_ids), 960):
            raise ValueError("Embedding ID/dimension mismatch")
        if not np.isfinite(self.x["t0"]).all() or not np.isfinite(self.x["t1"]).all():
            raise ValueError("Embedding NaN/Inf")
        train_ids = read_whitelist(self.data / f"{prefix}fold{self.fold}_train_whitelist.txt.gz")
        val_ids = read_whitelist(self.data / f"{prefix}fold{self.fold}_validation_whitelist.txt.gz")
        if set(train_ids) & set(val_ids):
            raise ValueError("Frozen train/validation whitelist overlap")
        self.train_index = np.array([self.t0_lookup[value] for value in train_ids], dtype=np.int64)
        self.val_index = np.array([self.t0_lookup[value] for value in val_ids], dtype=np.int64)
        self.train_ids, self.val_ids = train_ids, val_ids
        self._load_labels()

    def _load_labels(self):
        direct_frequency, propagated_frequency = {}, {}
        for row in read_tsv(self.data / "term_frequencies.tsv.gz"):
            if row["ontology"] == self.ontology:
                term = row["go_id"]
                direct_frequency[term] = int(row["t0_direct_protein_frequency"])
                propagated_frequency[term] = int(row["t0_propagated_protein_frequency"])
        t1_pairs = defaultdict(set)
        for row in read_tsv(self.data / "t1_groundtruth_propagated.tsv.gz"):
            if row["ontology"] == self.ontology:
                t1_pairs[row["protein_id"]].add(row["ancestor_go_id"])
                direct_frequency.setdefault(row["ancestor_go_id"], 0)
                propagated_frequency.setdefault(row["ancestor_go_id"], 0)
        self.labels = sorted(direct_frequency)
        self.label_lookup = {value: i for i, value in enumerate(self.labels)}
        self.direct_frequency = np.array([direct_frequency[value] for value in self.labels], dtype=np.int64)
        self.propagated_frequency = np.array([propagated_frequency[value] for value in self.labels], dtype=np.int64)
        self.information_content = -np.log((self.propagated_frequency + 1.0) / (66841 + 1.0)).astype(np.float32)
        self.log_prior = np.log((self.direct_frequency + 0.5) / (66841 + 1.0)).astype(np.float32)
        edges = defaultdict(list)
        for row in read_tsv(self.data / "ontology_edges.tsv.gz"):
            if row["ontology"] == self.ontology:
                edges[row["child_go_id"]].append(row["parent_go_id"])
        ancestors = ancestor_closure(edges)
        t0_pairs = defaultdict(set)
        selected = set(self.train_ids) | set(self.val_ids)
        for row in read_tsv(self.data / "t0_annotations_direct_clean.tsv.gz"):
            if row["ontology"] == self.ontology and row["protein_id"] in selected:
                t0_pairs[row["protein_id"]].update(ancestors(row["go_id"]))
        train_rows = [t0_pairs[value] for value in self.train_ids]
        val_rows = [t0_pairs[value] for value in self.val_ids]
        shape = (len(self.train_ids), len(self.labels))
        self.y_train = sparse_matrix(train_rows, self.label_lookup, shape)
        self.y_val = sparse_matrix(val_rows, self.label_lookup, (len(self.val_ids), len(self.labels)))
        self.y_t1 = sparse_matrix([t1_pairs[value] for value in self.t1_ids], self.label_lookup,
                                  (len(self.t1_ids), len(self.labels)))
        self.active = np.asarray(self.y_train.getnnz(axis=0)).ravel() > 0
        if not self.active.any():
            raise ValueError("No trainable labels")
        self.roles = {}
        for row in read_tsv(self.data / "t1_evaluation_membership.tsv.gz"):
            if row["ontology"] == self.ontology:
                self.roles.setdefault(row["role"], []).append((row["protein_id"], int(row["has_official_leaf_truth"])))
        self.role_indices, self.official_counts = {}, {}
        for role in ("NK", "LK"):
            members = self.roles.get(role, [])
            self.official_counts[role] = len(members)
            self.role_indices[role] = np.array([self.t1_lookup[value] for value, has_truth in members if has_truth],
                                                dtype=np.int64)
        homology = {row["protein_id"]: row for row in read_tsv(self.data / "t1_to_t0_homology.tsv.gz")}
        self.homology_bucket = np.array([homology[value]["identity_bucket"] for value in self.t1_ids])
        self.exact_overlap = np.array([homology[value]["exact_sequence_overlap"] == "1" for value in self.t1_ids])
        pd.DataFrame({"go_id": self.labels, "t0_direct_frequency": self.direct_frequency,
                      "t0_propagated_frequency": self.propagated_frequency,
                      "frequency_bucket": [frequency_bucket(x) for x in self.direct_frequency],
                      "train_active": self.active}).to_csv(self.out / "labels.tsv.gz", sep="\t", index=False)

    def predict_logits(self, model, torch, indices, source, path):
        active_indices = np.flatnonzero(self.active)
        result = np.memmap(path, dtype=np.float32, mode="w+", shape=(len(indices), len(self.labels)))
        result[:] = 0
        model.eval()
        with torch.no_grad():
            for start in range(0, len(indices), self.args.batch):
                selected = indices[start:start + self.args.batch]
                array = np.asarray(self.x[source][selected], dtype=np.float32)
                values = model(torch.from_numpy(array).cuda()).float().cpu().numpy()
                if not np.isfinite(values).all():
                    raise ValueError("NaN/Inf model logit")
                result[start:start + len(selected), active_indices] = values
        result.flush()
        return result

    def train_seed(self, torch, seed):
        train_x = torch.from_numpy(np.asarray(self.x["t0"][self.train_index], dtype=np.float32)).cuda()
        active = torch.from_numpy(np.flatnonzero(self.active)).cuda()
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = torch.nn.Linear(960, int(self.active.sum())).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        best_loss, best_epoch, best_state, stale = np.inf, None, None, 0
        history = []
        for epoch in range(1, self.args.epochs + 1):
            model.train()
            order = torch.randperm(len(self.train_index), device="cuda", generator=generator)
            train_loss = 0.0
            for start in range(0, len(order), self.args.batch):
                batch_index = order[start:start + self.args.batch]
                cpu_index = batch_index.cpu().numpy()
                target = self.y_train[cpu_index][:, self.active].toarray().astype(np.float32)
                target = torch.from_numpy(target).cuda()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(model(train_x[batch_index]), target)
                if not torch.isfinite(loss):
                    raise ValueError("NaN/Inf training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if not all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                           for parameter in model.parameters()):
                    raise ValueError("NaN/Inf gradient")
                optimizer.step()
                train_loss += float(loss.detach()) * len(batch_index)
            model.eval()
            val_loss, val_items = 0.0, 0
            with torch.no_grad():
                for start in range(0, len(self.val_index), self.args.batch):
                    selected = self.val_index[start:start + self.args.batch]
                    array = np.asarray(self.x["t0"][selected], dtype=np.float32)
                    logits = model(torch.from_numpy(array).cuda())
                    target = self.y_val[start:start + len(selected)][:, self.active].toarray().astype(np.float32)
                    target = torch.from_numpy(target).cuda()
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="sum")
                    if not torch.isfinite(loss):
                        raise ValueError("NaN/Inf validation loss")
                    val_loss += float(loss)
                    val_items += target.numel()
            val_loss /= val_items
            history.append({"seed": seed, "epoch": epoch, "train_bce": train_loss / len(order),
                            "validation_bce": val_loss, "stale_epochs": stale})
            if val_loss < best_loss - self.args.min_delta:
                best_loss, best_epoch, stale = val_loss, epoch, 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
            print("EPOCH", seed, epoch, val_loss, "best", best_epoch, "stale", stale, flush=True)
            if stale >= self.args.patience:
                break
        if best_state is None:
            raise RuntimeError("No finite validation checkpoint")
        model.load_state_dict(best_state)
        checkpoint = {"seed": seed, "best_epoch": best_epoch, "validation_bce": best_loss,
                      "active_indices": np.flatnonzero(self.active), "state_dict": best_state,
                      "embedding": "ESMC-300M-960-mean-frozen"}
        torch.save(checkpoint, self.out / f"bce_seed{seed}.pt")
        del train_x, active, optimizer, generator
        torch.cuda.empty_cache()
        return model, pd.DataFrame(history), {"seed": seed, "best_epoch": best_epoch, "validation_bce": best_loss}

    def temporary_probabilities(self, logits, tau, name):
        probability = np.memmap(self.out / name, dtype=np.float32, mode="w+", shape=logits.shape)
        probability[:] = 0
        active_indices = np.flatnonzero(self.active)
        for start in range(0, logits.shape[0], self.args.batch):
            raw = np.asarray(logits[start:start + self.args.batch, active_indices])
            values = expit(raw - tau * self.log_prior[active_indices]).astype(np.float32)
            if not np.isfinite(values).all():
                raise ValueError("NaN/Inf adjusted probability")
            probability[start:start + self.args.batch, active_indices] = values
        probability.flush()
        return probability

    def evaluate_selected(self, model, seed, val_probability, t1_probability, fmax_threshold, smin_threshold):
        rows, label_tables = [], []
        fieldnames = ["model", "seed", "split", "ontology", "role", "slice_type", "slice_value",
                      "frequency_bucket", "n_proteins", "official_role_proteins", "macro_auprc", "n_labels",
                      "fmax_threshold", "smin_threshold", "fmax_at_val_threshold", "smin_at_val_threshold",
                      "fmax_coverage", "smin_coverage", "ru", "mi"]
        import io
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        val_auprc, val_labels = macro_auprc(self.y_val, val_probability, self.direct_frequency, return_table=True)
        val_metrics = metric_at_threshold(self.y_val, val_probability, self.information_content, fmax_threshold)
        val_smin = metric_at_threshold(self.y_val, val_probability, self.information_content, smin_threshold)
        val_metrics["smin_at_val_threshold"] = val_smin["smin_at_val_threshold"]
        val_metrics["fmax_coverage"] = val_metrics.pop("coverage")
        val_metrics["smin_coverage"] = val_smin["coverage"]
        val_metrics["ru"], val_metrics["mi"] = val_smin["ru"], val_smin["mi"]
        write_metrics(writer, model, seed, "t0_proxy_test", self.ontology, "ALL", "all", "all",
                      val_metrics, val_auprc, fmax_threshold, smin_threshold, len(self.val_index), len(self.val_index))
        val_labels["split"], val_labels["role"] = "t0_proxy_test", "ALL"
        label_tables.append(val_labels)
        for role in ("NK", "LK"):
            indices = self.role_indices[role]
            if not len(indices):
                raise ValueError(f"No evaluable t1 proteins for {self.ontology} {role}")
            role_y = self.y_t1[indices]
            role_p = t1_probability[indices]
            slices = [("role", role, np.arange(len(indices)))]
            slices.extend(("exact_overlap", str(int(flag)), np.flatnonzero(self.exact_overlap[indices] == flag))
                          for flag in (False, True))
            buckets = ["100", "90-<100", "50-<90", "30-<50", "20-<30", "no_detected_hit"]
            slices.extend(("homology_bucket", bucket, np.flatnonzero(self.homology_bucket[indices] == bucket))
                          for bucket in buckets)
            for slice_type, slice_value, local_index in slices:
                if not len(local_index):
                    continue
                selected_y, selected_p = role_y[local_index], role_p[local_index]
                auprc, label_table = macro_auprc(selected_y, selected_p, self.direct_frequency, return_table=True)
                metrics = metric_at_threshold(selected_y, selected_p, self.information_content, fmax_threshold)
                smin = metric_at_threshold(selected_y, selected_p, self.information_content, smin_threshold)
                metrics["smin_at_val_threshold"] = smin["smin_at_val_threshold"]
                metrics["fmax_coverage"] = metrics.pop("coverage")
                metrics["smin_coverage"] = smin["coverage"]
                metrics["ru"], metrics["mi"] = smin["ru"], smin["mi"]
                write_metrics(writer, model, seed, "t1", self.ontology, role, slice_type, slice_value,
                              metrics, auprc, fmax_threshold, smin_threshold, len(local_index), self.official_counts[role])
                if slice_type == "role":
                    label_table["split"], label_table["role"] = "t1", role
                    label_tables.append(label_table)
        return buffer.getvalue(), label_tables, val_metrics

    def run(self):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("Explicit CUDA required; CPU fallback is forbidden")
        seeds = [int(value) for value in self.args.seeds.split(",")]
        fitted, histories = [], []
        for seed in seeds:
            model, history, metadata = self.train_seed(torch, seed)
            logits = self.predict_logits(model, torch, self.val_index, "t0", self.out / f".val_logits_seed{seed}.mmap")
            fitted.append({"model": model, "metadata": metadata, "logits": logits})
            histories.append(history)
        pd.concat(histories).to_csv(self.out / "validation_history.tsv", sep="\t", index=False)
        tau_rows = []
        for tau in self.args.taus:
            seed_scores = []
            for item in fitted:
                _, table = macro_auprc(self.y_val, item["logits"], self.direct_frequency, active=self.active,
                                       tau=tau, log_prior=self.log_prior, logits=True, return_table=True)
                tail = table[self.direct_frequency < 100].auprc.dropna()
                if tail.empty:
                    raise ValueError("No validation tail labels for tau selection")
                seed_scores.append(float(tail.mean()))
            tau_rows.append({"tau": tau, "validation_tail100_macro_auprc": float(np.mean(seed_scores)),
                             "seed_scores": json.dumps(seed_scores)})
        tau_frame = pd.DataFrame(tau_rows)
        tau_frame.to_csv(self.out / "tau_validation.tsv", sep="\t", index=False)
        selected_tau = float(tau_frame.sort_values("tau").iloc[tau_frame.validation_tail100_macro_auprc.idxmax()].tau)
        all_metrics, all_labels, seed_summaries = [], [], []
        for item in fitted:
            seed = item["metadata"]["seed"]
            val_probability = self.temporary_probabilities(item["logits"], selected_tau, f".val_probability_seed{seed}.mmap")
            fmax_threshold, smin_threshold, grid = validation_thresholds(self.y_val, val_probability, self.information_content)
            grid["seed"] = seed
            grid.to_csv(self.out / f"validation_thresholds_seed{seed}.tsv", sep="\t", index=False)
            t1_logits = self.predict_logits(item["model"], torch, np.arange(len(self.t1_ids)), "t1",
                                             self.out / f".t1_logits_seed{seed}.mmap")
            t1_probability = self.temporary_probabilities(t1_logits, selected_tau, f".t1_probability_seed{seed}.mmap")
            metrics, labels, _ = self.evaluate_selected("frozen_bce_la", seed, val_probability, t1_probability,
                                                         fmax_threshold, smin_threshold)
            all_metrics.append(metrics)
            all_labels.extend(labels)
            seed_summaries.append({**item["metadata"], "tau": selected_tau, "fmax_threshold": fmax_threshold,
                                   "smin_threshold": smin_threshold})
            for temporary in (item["logits"], val_probability, t1_logits, t1_probability):
                del temporary
            for name in (f".val_logits_seed{seed}.mmap", f".val_probability_seed{seed}.mmap",
                         f".t1_logits_seed{seed}.mmap", f".t1_probability_seed{seed}.mmap"):
                os.remove(self.out / name)
            del item["model"]
            torch.cuda.empty_cache()
        knn_metrics, knn_labels = self.run_knn(torch)
        all_metrics.append(knn_metrics)
        all_labels.extend(knn_labels)
        with gzip.open(self.out / "metrics.tsv.gz", "wt", encoding="utf-8", newline="") as handle:
            handle.write(all_metrics[0])
            for text in all_metrics[1:]:
                handle.write(text.split("\n", 1)[1])
        offset = 0
        for seed in seeds:
            for item in all_labels[offset:offset + 3]:
                item["model"], item["seed"] = "frozen_bce_la", seed
            offset += 3
        for item in all_labels[offset:]:
            item["model"], item["seed"] = "deep_knn", "deterministic"
        labels = pd.concat(all_labels, ignore_index=True)
        labels["label_index"] = labels.label_index.astype(int)
        labels["go_id"] = [self.labels[index] for index in labels.label_index]
        labels["t0_direct_frequency"] = self.direct_frequency[labels.label_index]
        labels.drop(columns=["label_index"]).to_csv(self.out / "per_label_metrics.tsv.gz", sep="\t", index=False)
        summary = {"schema_version": "cafa3-esmc-baseline-run-v1", "fold": self.fold, "track": self.track,
                   "ontology": self.ontology, "seeds": seeds, "selected_tau": selected_tau,
                   "bce": seed_summaries, "deep_knn": {"k": self.args.k, "seed": "deterministic"},
                   "n_train": len(self.train_index), "n_validation": len(self.val_index),
                   "n_labels": len(self.labels), "n_active_labels": int(self.active.sum()),
                   "t1_official_role_proteins": self.official_counts,
                   "t1_evaluable": {role: len(indices) for role, indices in self.role_indices.items()},
                   "t0_proxy_test_warning": "same held-out fold used for validation selection; not independent test",
                   "t1_selection": "none; thresholds and tau frozen on t0 validation before t1 prediction",
                   "data_manifest_sha256": sha256_file(self.data / "frozen_manifest.json"),
                   "code_sha256": sha256_file(os.path.abspath(__file__))}
        (self.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print("RUN_DONE", json.dumps(summary), flush=True)

    def run_knn(self, torch):
        active_indices = np.flatnonzero(self.active)
        train_x = torch.nn.functional.normalize(torch.from_numpy(
            np.asarray(self.x["t0"][self.train_index], dtype=np.float32)).cuda(), dim=1)
        train_y = torch.from_numpy(self.y_train[:, self.active].toarray().astype(np.float32)).cuda()

        def predict(indices, source, path):
            result = np.memmap(path, dtype=np.float32, mode="w+", shape=(len(indices), len(self.labels)))
            result[:] = 0
            with torch.no_grad():
                for start in range(0, len(indices), self.args.batch):
                    selected = indices[start:start + self.args.batch]
                    array = np.asarray(self.x[source][selected], dtype=np.float32)
                    batch = torch.nn.functional.normalize(torch.from_numpy(array).cuda(), dim=1)
                    similarity, index = (batch @ train_x.T).topk(self.args.k, dim=1)
                    weight = similarity.clamp(min=0) + 1e-6
                    values = ((train_y[index] * weight.unsqueeze(-1)).sum(1) /
                              weight.sum(1, keepdim=True)).float().cpu().numpy()
                    if not np.isfinite(values).all():
                        raise ValueError("NaN/Inf kNN probability")
                    result[start:start + len(selected), active_indices] = values
            result.flush()
            return result

        val_probability = predict(self.val_index, "t0", self.out / ".val_probability_knn.mmap")
        fmax_threshold, smin_threshold, grid = validation_thresholds(self.y_val, val_probability, self.information_content)
        grid.to_csv(self.out / "validation_thresholds_knn.tsv", sep="\t", index=False)
        t1_probability = predict(np.arange(len(self.t1_ids)), "t1", self.out / ".t1_probability_knn.mmap")
        metrics, labels, _ = self.evaluate_selected("deep_knn", "deterministic", val_probability, t1_probability,
                                                    fmax_threshold, smin_threshold)
        for name in (".val_probability_knn.mmap", ".t1_probability_knn.mmap"):
            os.remove(self.out / name)
        del train_x, train_y
        torch.cuda.empty_cache()
        return metrics, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--emb-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fold", type=int, required=True, choices=[0, 1, 2])
    parser.add_argument("--track", required=True, choices=["main", "id30", "id50", "id90"])
    parser.add_argument("--ontology", required=True, choices=["MFO", "BPO", "CCO"])
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--taus", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    parser.add_argument("--k", type=int, default=20)
    args = parser.parse_args()
    if args.patience != 20 or args.min_delta != 1e-4 or args.epochs != 1000 or args.seeds != "0,1,2":
        raise ValueError("Formal CAFA3 runs require seeds=0,1,2, patience=20, min_delta=1e-4 and 1000-epoch failsafe")
    Cafa3Run(args).run()


if __name__ == "__main__":
    main()
