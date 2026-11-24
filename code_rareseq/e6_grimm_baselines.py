#!/usr/bin/env python3
"""E6 GRIMM EC_v2 split_1 frozen ESM-C 多标签、kNN 与 pseudo-OOD 基准。"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp
from sklearn.metrics import average_precision_score, roc_auc_score

from log_run import RunLogger, sha256_file


def multilabel(frame, labels):
    lookup = {label: i for i, label in enumerate(labels)}
    y = np.zeros((len(frame), len(labels)), dtype=np.uint8)
    for row, value in enumerate(frame["EC number"]):
        for label in str(value).split(";"):
            if label in lookup:
                y[row, lookup[label]] = 1
    return y


def tier(count):
    if count < 10:
        return "tail_lt10"
    if count < 30:
        return "tail_10_29"
    if count < 100:
        return "mid_30_99"
    return "head_ge100"


def per_label(y, probability, frequency, labels):
    rows = []
    for j, label in enumerate(labels):
        positives = int(y[:, j].sum())
        ap = average_precision_score(y[:, j], probability[:, j]) if 0 < positives < len(y) else np.nan
        rows.append({"ec": label, "n_train": int(frequency[j]), "tier": tier(frequency[j]),
                     "n_positive": positives, "auprc": ap})
    return pd.DataFrame(rows)


def macro_ap(table):
    eligible = table.dropna(subset=["auprc"])
    return float(eligible.auprc.mean()) if len(eligible) else np.nan


def sample_fmax(y, probability, threshold=None):
    candidates = np.linspace(0.01, 0.99, 99) if threshold is None else [threshold]
    best = (-1.0, None)
    for value in candidates:
        pred = probability >= value
        tp = (pred & (y == 1)).sum(1)
        n_pred = pred.sum(1)
        n_true = y.sum(1)
        precision = np.divide(tp, n_pred, out=np.zeros_like(tp, dtype=float), where=n_pred > 0)
        recall = np.divide(tp, n_true, out=np.zeros_like(tp, dtype=float), where=n_true > 0)
        score = float(np.mean(2 * precision * recall / np.maximum(precision + recall, 1e-12)))
        if score > best[0]:
            best = (score, float(value))
    return best


def calibration(y, probability, bins=15):
    counts = np.zeros(bins, dtype=np.int64)
    confidence = np.zeros(bins)
    accuracy = np.zeros(bins)
    brier = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for start in range(0, len(y), 128):
        p = probability[start:start + 128].ravel()
        target = y[start:start + 128].ravel()
        index = np.minimum((p * bins).astype(int), bins - 1)
        counts += np.bincount(index, minlength=bins)
        confidence += np.bincount(index, weights=p, minlength=bins)
        accuracy += np.bincount(index, weights=target, minlength=bins)
        brier += float(np.square(p - target).sum())
    used = counts > 0
    ece = float(np.sum(counts[used] / counts.sum() *
                       np.abs(accuracy[used] / counts[used] - confidence[used] / counts[used])))
    return ece, brier / y.size


def function_metrics(y, probability, frequency, labels, threshold):
    table = per_label(y, probability, frequency, labels)
    fmax, _ = sample_fmax(y, probability, threshold)
    result = {"macro_auprc": macro_ap(table),
              "micro_auprc": float(average_precision_score(y.ravel(), probability.ravel())),
              "sample_fmax_at_val_threshold": fmax, "val_fmax_threshold": threshold}
    for name, group in table.groupby("tier"):
        result["macro_auprc_" + name] = macro_ap(group)
        result["labels_with_positive_" + name] = int(group.auprc.notna().sum())
    return result, table


def fit_temperature(logits, y):
    import torch
    x = torch.from_numpy(logits).cuda()
    target = torch.from_numpy(y.astype(np.float32)).cuda()
    log_temperature = torch.zeros(1, device="cuda", requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.2, max_iter=50)

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.clamp(-6.9, 4.6).exp()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(x / temperature, target)
        loss.backward()
        return loss

    optimizer.step(closure)
    value = float(log_temperature.detach().clamp(-6.9, 4.6).exp().cpu())
    del x, target, log_temperature, optimizer
    torch.cuda.empty_cache()
    if not 0.001 <= value <= 100:
        raise ValueError(f"Invalid calibration temperature: {value}")
    return value


def knn_predict(train_x, train_y, eval_x, k, batch=512):
    import torch
    a = torch.nn.functional.normalize(torch.from_numpy(train_x).cuda(), dim=1)
    train_labels = torch.from_numpy(train_y.astype(np.float32)).cuda()
    outputs, maxima = [], []
    with torch.no_grad():
        for start in range(0, len(eval_x), batch):
            b = torch.nn.functional.normalize(torch.from_numpy(eval_x[start:start + batch]).cuda(), dim=1)
            similarity, index = (b @ a.T).topk(k, dim=1)
            weight = similarity.clamp(min=0) + 1e-6
            prediction = (train_labels[index] * weight.unsqueeze(-1)).sum(1) / weight.sum(1, keepdim=True)
            outputs.append(prediction.cpu().numpy())
            maxima.append(similarity[:, 0].cpu().numpy())
    return np.concatenate(outputs), np.concatenate(maxima)


def ood_statistics(y, score):
    order = np.argsort(-score)
    risk = np.cumsum(y[order]) / np.arange(1, len(y) + 1)
    coverage = np.arange(1, len(y) + 1) / len(y)
    threshold = np.quantile(score[y == 0], 0.95)
    return {"auroc_ood": float(roc_auc_score(y, score)),
            "fpr95": float((score[y == 1] >= threshold).mean()),
            "aurc_ood_error": float(np.trapezoid(risk, coverage))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--min-delta", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--taus", default="0,0.5,1")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--log", default="/root/rareseq/logs/experiments.jsonl")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    config = vars(args).copy()
    config.update({"checkpoint_selection": "validation macro AUPRC, train labels with support <100",
                   "tau_selection": "validation macro AUPRC, train labels with support <100; shared across seeds",
                   "fmax_threshold": "validation sample-wise F1, independently per seed/model",
                   "test2": "pseudo-OOD only; no test-driven selection"})
    code = sha256_file(os.path.abspath(__file__))
    lg = RunLogger(args.log, stage="E6", kind="grimm-frozen-baselines")
    lg.start(config=config, data_version="GRIMM-EC_v2/split_1+ESMC-300M-frozen",
             inputs=[args.data_dir, args.emb_dir], code_version=code)
    try:
        import torch
        seeds = [int(x) for x in args.seeds.split(",")]
        taus = [float(x) for x in args.taus.split(",")]
        frames = {name: pd.read_csv(os.path.join(args.data_dir, name + ".tsv"), sep="\t")
                  for name in ("train", "validation", "test1", "test2")}
        labels = sorted({label for value in frames["train"]["EC number"] for label in str(value).split(";")})
        label_counts = frames["train"]["EC number"].str.split(";").explode().value_counts()
        frequency = np.array([label_counts[label] for label in labels])
        y = {name: multilabel(frame, labels) for name, frame in frames.items()}
        x = {name: np.load(os.path.join(args.emb_dir, name + ".npy")).astype(np.float32)
             for name in frames}
        for name in frames:
            ids = pd.read_csv(os.path.join(args.emb_dir, name + "_ids.tsv"), sep="\t")
            assert ids.Entry.tolist() == frames[name].Entry.tolist()
            assert x[name].shape[0] == y[name].shape[0]
        mean = x["train"].mean(0, keepdims=True)
        std = x["train"].std(0, keepdims=True).clip(min=1e-5)
        for name in x:
            x[name] = (x[name] - mean) / std
        train_x = torch.from_numpy(x["train"]).cuda()
        train_y = torch.from_numpy(y["train"].astype(np.float32)).cuda()
        log_prior = np.log((frequency + 0.5) / (len(frames["train"]) + 1.0)).astype(np.float32)
        fitted = []
        history = []
        for seed in seeds:
            torch.manual_seed(seed)
            model = torch.nn.Linear(x["train"].shape[1], len(labels)).cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            generator = torch.Generator(device="cuda").manual_seed(seed)
            best_score, best_state, best_epoch = -np.inf, None, None
            stale = 0
            for epoch in range(1, args.epochs + 1):
                model.train()
                order = torch.randperm(len(train_x), device="cuda", generator=generator)
                total = 0.0
                for start in range(0, len(order), args.batch):
                    index = order[start:start + args.batch]
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(train_x[index]), train_y[index])
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    total += float(loss.detach()) * len(index)
                model.eval()
                with torch.no_grad():
                    val_logits = torch.cat([model(torch.from_numpy(x["validation"][i:i + args.batch]).cuda())
                                            for i in range(0, len(x["validation"]), args.batch)]).cpu().numpy()
                table = per_label(y["validation"], expit(val_logits), frequency, labels)
                score = macro_ap(table[table.n_train < 100])
                history.append({"seed": seed, "epoch": epoch, "train_loss": total / len(order),
                                "val_tail100_macro_auprc": score, "stale_epochs": stale})
                if score > best_score + args.min_delta:
                    best_score, best_epoch, stale = score, epoch, 0
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                else:
                    stale += 1
                print("EPOCH", seed, epoch, score, "stale", stale, flush=True)
                if stale >= args.patience:
                    break
            model.load_state_dict(best_state)
            model.eval()
            logits = {}
            for name in ("validation", "test1", "test2"):
                with torch.no_grad():
                    logits[name] = torch.cat([model(torch.from_numpy(x[name][i:i + args.batch]).cuda())
                                              for i in range(0, len(x[name]), args.batch)]).float().cpu().numpy()
            temperature = fit_temperature(logits["validation"], y["validation"])
            fitted.append({"seed": seed, "epoch": best_epoch, "logits": logits, "temperature": temperature})
            print("FIT_DONE", seed, best_epoch, temperature, flush=True)
        pd.DataFrame(history).to_csv(os.path.join(args.out, "validation_history.tsv"), sep="\t", index=False)
        tau_rows = []
        for tau in taus:
            tables = [per_label(y["validation"], expit(item["logits"]["validation"] - tau * log_prior), frequency, labels)
                      for item in fitted]
            scores = [macro_ap(table[table.n_train < 100]) for table in tables]
            tau_rows.append({"tau": tau, "val_tail100_macro_auprc": np.mean(scores), "seed_sd": np.std(scores)})
        tau_frame = pd.DataFrame(tau_rows)
        tau_frame.to_csv(os.path.join(args.out, "tau_validation.tsv"), sep="\t", index=False)
        selected_tau = float(tau_frame.sort_values("tau").iloc[tau_frame.val_tail100_macro_auprc.idxmax()].tau)
        all_metrics, label_tables, ood_rows = [], [], []
        ood_scores = {"msp": [], "energy": [], "deep_knn": []}
        for item in fitted:
            seed = item["seed"]
            val_probability = expit(item["logits"]["validation"] - selected_tau * log_prior)
            _, threshold = sample_fmax(y["validation"], val_probability)
            test_logits = item["logits"]["test1"] - selected_tau * log_prior
            probability = expit(test_logits)
            metrics, table = function_metrics(y["test1"], probability, frequency, labels, threshold)
            calibrated_probability = expit(test_logits / item["temperature"])
            ece, brier = calibration(y["test1"], calibrated_probability)
            metrics.update({"seed": seed, "model": "frozen_bce_la", "best_epoch": item["epoch"],
                            "temperature": item["temperature"], "tau": selected_tau, "ece": ece, "brier": brier})
            all_metrics.append(metrics)
            table["seed"] = seed
            label_tables.append(table)
            known = expit(item["logits"]["test1"] - selected_tau * log_prior).max(1)
            unknown = expit(item["logits"]["test2"] - selected_tau * log_prior).max(1)
            ood_scores["msp"].append(-np.concatenate([known, unknown]))
            raw = np.concatenate([item["logits"]["test1"] - selected_tau * log_prior,
                                  item["logits"]["test2"] - selected_tau * log_prior], axis=0)
            ood_scores["energy"].append(-logsumexp(raw, axis=1))
        knn_raw = {name: np.load(os.path.join(args.emb_dir, name + ".npy")).astype(np.float32)
                   for name in frames}
        knn_val, _ = knn_predict(knn_raw["train"], y["train"], knn_raw["validation"], args.k)
        _, threshold = sample_fmax(y["validation"], knn_val)
        knn_test, known_similarity = knn_predict(knn_raw["train"], y["train"], knn_raw["test1"], args.k)
        _, unknown_similarity = knn_predict(knn_raw["train"], y["train"], knn_raw["test2"], args.k)
        metrics, table = function_metrics(y["test1"], knn_test, frequency, labels, threshold)
        ece, brier = calibration(y["test1"], knn_test)
        metrics.update({"seed": "deterministic", "model": "deep_knn", "k": args.k, "ece": ece, "brier": brier})
        all_metrics.append(metrics)
        table["seed"] = "deterministic"
        label_tables.append(table)
        ood_scores["deep_knn"].append(-np.concatenate([known_similarity, unknown_similarity]))
        y_ood = np.concatenate([np.zeros(len(knn_test), dtype=np.uint8), np.ones(len(unknown_similarity), dtype=np.uint8)])
        for method, values in ood_scores.items():
            score = np.mean(values, axis=0)
            row = {"method": method, **ood_statistics(y_ood, score)}
            ood_rows.append(row)
            np.save(os.path.join(args.out, f"ood_scores_{method}.npy"), score.astype(np.float32))
            order = np.argsort(-score)
            risk = np.cumsum(y_ood[order]) / np.arange(1, len(score) + 1)
            points = np.linspace(0.05, 1, 20)
            pd.DataFrame({"coverage": points, "ood_error": risk[np.ceil(points * len(score)).astype(int) - 1]}).to_csv(
                os.path.join(args.out, f"risk_coverage_{method}.tsv"), sep="\t", index=False)
        metrics_frame = pd.DataFrame(all_metrics)
        metrics_frame.to_csv(os.path.join(args.out, "function_metrics.tsv"), sep="\t", index=False)
        labels_frame = pd.concat(label_tables, ignore_index=True)
        labels_frame.to_csv(os.path.join(args.out, "per_label_metrics.tsv"), sep="\t", index=False)
        rng = np.random.default_rng(20261124)
        neural_labels = labels_frame[labels_frame.seed != "deterministic"].groupby(
            ["ec", "n_train", "tier"], as_index=False).auprc.mean()
        knn_labels = labels_frame[labels_frame.seed == "deterministic"][["ec", "auprc"]]
        paired_labels = neural_labels.merge(knn_labels, on="ec", suffixes=("_neural", "_knn")).dropna()
        paired_labels["delta_auprc"] = paired_labels.auprc_neural - paired_labels.auprc_knn
        label_bootstrap = []
        groups = [group for _, group in paired_labels.groupby("tier")]
        for bucket in ("all", "tail_lt10", "tail_10_29", "mid_30_99", "head_ge100"):
            selected = paired_labels if bucket == "all" else paired_labels[paired_labels.tier == bucket]
            deltas = np.empty(args.bootstrap)
            for i in range(args.bootstrap):
                if bucket == "all":
                    estimate = sum(len(group) * group.delta_auprc.to_numpy()[rng.integers(0, len(group), len(group))].mean()
                                   for group in groups) / len(paired_labels)
                else:
                    estimate = selected.delta_auprc.to_numpy()[rng.integers(0, len(selected), len(selected))].mean()
                deltas[i] = estimate
            p = min(1.0, 2 * (1 + min((deltas <= 0).sum(), (deltas >= 0).sum())) / (len(deltas) + 1))
            label_bootstrap.append({"bucket": bucket, "n_labels": len(selected),
                                    "delta_macro_auprc_neural_minus_knn": selected.delta_auprc.mean(),
                                    "ci_low": np.quantile(deltas, 0.025), "ci_high": np.quantile(deltas, 0.975),
                                    "p_two_sided": p, "n_bootstrap": args.bootstrap,
                                    "resampling_unit": "EC label within frequency tier"})
        pd.DataFrame(label_bootstrap).to_csv(os.path.join(args.out, "function_label_bootstrap.tsv"), sep="\t", index=False)
        bootstrap_rows = []
        for method in ("msp", "energy"):
            a = np.mean(ood_scores[method], axis=0)
            b = ood_scores["deep_knn"][0]
            observed = ood_statistics(y_ood, a)["auroc_ood"] - ood_statistics(y_ood, b)["auroc_ood"]
            deltas = np.empty(args.bootstrap)
            for i in range(args.bootstrap):
                index = rng.integers(0, len(y_ood), len(y_ood))
                if len(np.unique(y_ood[index])) == 2:
                    deltas[i] = roc_auc_score(y_ood[index], a[index]) - roc_auc_score(y_ood[index], b[index])
                else:
                    deltas[i] = np.nan
            deltas = deltas[np.isfinite(deltas)]
            p = min(1.0, 2 * (1 + min((deltas <= 0).sum(), (deltas >= 0).sum())) / (len(deltas) + 1))
            bootstrap_rows.append({"comparison": method + "_minus_deep_knn", "delta_auroc_ood": observed,
                                   "ci_low": np.quantile(deltas, 0.025), "ci_high": np.quantile(deltas, 0.975),
                                   "p_two_sided": p, "n_bootstrap": len(deltas)})
            np.savez_compressed(os.path.join(args.out, f"bootstrap_{method}.npz"), deltas=deltas)
        pd.DataFrame(ood_rows).to_csv(os.path.join(args.out, "ood_metrics.tsv"), sep="\t", index=False)
        pd.DataFrame(bootstrap_rows).to_csv(os.path.join(args.out, "ood_paired_bootstrap.tsv"), sep="\t", index=False)
        known_test2 = y["test2"].sum(1)
        coverage = {"test2_rows": len(y["test2"]), "test2_rows_with_any_train_label": int((known_test2 > 0).sum()),
                    "test2_rows_without_train_label": int((known_test2 == 0).sum()),
                    "test2_known_label_annotations": int(known_test2.sum()),
                    "test2_total_label_annotations": int(frames["test2"]["EC number"].str.count(";").add(1).sum())}
        summary = {"selected_tau": selected_tau, "seeds": seeds, "test2_coverage": coverage,
                   "function_metrics": all_metrics, "function_label_bootstrap": label_bootstrap,
                   "ood_metrics": ood_rows, "ood_bootstrap": bootstrap_rows}
        with open(os.path.join(args.out, "summary.json"), "w") as handle:
            json.dump(summary, handle, indent=2)
        lg.finish(metrics={"selected_tau": selected_tau, "test2_coverage": coverage,
                           "function_metrics": all_metrics, "function_label_bootstrap": label_bootstrap,
                           "ood_metrics": ood_rows, "ood_bootstrap": bootstrap_rows}, outputs=[args.out])
        print("E6_DONE", json.dumps(summary), flush=True)
    except Exception as exc:
        lg.finish(status="failed", notes=f"{type(exc).__name__}: {exc}", outputs=[args.out])
        raise


if __name__ == "__main__":
    main()
