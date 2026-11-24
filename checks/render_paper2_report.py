import argparse
import json
from pathlib import Path


def _load(path):
    return json.loads(Path(path).read_text())


def _number(value, digits=6):
    return "NA" if value is None else f"{value:.{digits}f}"


def _configuration(summary, method, cap=16):
    return next(row for row in summary["configurations"] if row["method"] == method and row["proposal_cap"] == cap)


def _top(summary, method, k):
    return next(row for row in summary["top_k"] if row["method"] == method and row["k"] == k)


def _comparison(summary, method, reference, metric="regret"):
    return next(row for row in summary["paired_comparisons"] if row["method"] == method and row["reference"] == reference and row["metric"] == metric)


def _context_table(summary):
    lines = ["| Method | Mean regret | Regret@4 | CNN member forwards | Top-4 max measured MRL |", "|---|---:|---:|---:|---:|"]
    for method in ("bon", "lcb", "distance", "com", "diversity_lcb"):
        row = _configuration(summary, method)
        top = _top(summary, method, 4)
        lines.append(f"| {method} | {_number(row['mean_regret'])} | {_number(top['regret_at_k'])} | {row['mean_cnn_member_item_forwards']:.0f} | {_number(top['max_measured_rl'])} |")
    return "\n".join(lines)


def _interval_table(summary):
    lines = ["| Method | Reference | Mean difference | Pool-cluster bootstrap 95% CI |", "|---|---|---:|---|"]
    for method in ("lcb", "distance", "com", "diversity_lcb"):
        for reference in (("bon",) if method == "lcb" else ("bon", "lcb")):
            row = _comparison(summary, method, reference)
            interval = row["pool_cluster_bootstrap_95_percent_interval"]
            lines.append(f"| {method} | {reference} | {_number(row['mean_difference'])} | [{_number(interval[0])}, {_number(interval[1])}] |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--com-summary", required=True)
    parser.add_argument("--dev-frozen", required=True)
    parser.add_argument("--designed-summary", required=True)
    parser.add_argument("--random-summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    com, dev = _load(args.com_summary), _load(args.dev_frozen)
    designed, random = _load(args.designed_summary), _load(args.random_summary)
    lines = [
        "# Paper 2 missing baselines: verified result",
        "",
        "## Evidence boundary",
        "",
        "This is a COM-style conservative-objective adaptation, not the official COM UTR-ResNet or official weights. Outer pools were already viewed: every outer result here is retrospective sensitivity evidence, not a new blind test. No outer value was used for training or hyperparameter selection. Calibration was not used for selection. The earlier FP-REJ/Q falsification result is unchanged.",
        "",
        "## COM convergence",
        "",
        "All reported COM seeds used actual parallel CUDA workers, 50-step discrete adversarial negatives and the dual train/dev patience rule. No maximum-epoch stopping rule was imposed.",
        "",
        "| Seed | Epochs | Best epoch | Best train MSE | Best dev MSE | Best alpha | Seconds | Peak allocated bytes |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in com["seeds"]:
        lines.append(f"| {row['seed']} | {row['epochs']} | {row['best_epoch']} | {_number(row['best_train_eval_loss'])} | {_number(row['best_dev_eval_loss'])} | {row['best_alpha']:.3e} | {_number(row['training_seconds'], 2)} | {row['max_memory_allocated_bytes']} |")
    lines += [
        "",
        f"Training code commit: `{com['seeds'][0]['code_commit']}`. Config SHA256: `{com['config_sha256']}`. Histories, checkpoints, per-epoch hashes and three independent CUDA reload/recompute reports are stored with the artifacts.",
        "",
        "Three aborted implementations are preserved under `invalid_com_training_dual_gradient`, `invalid_com_training_positive_gradient`, and `invalid_com_training_particle_scale`; none contributes a model, selection or scientific conclusion.",
        "",
        "## Dev-only frozen choices",
        "",
        "| Method | Lambda | Gamma | Dev mean regret | Dev regret@4 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, row in dev["selected"].items():
        lines.append(f"| {method} | {_number(row['lambda'])} | {_number(row['gamma'])} | {_number(row['mean_regret'])} | {_number(row['mean_regret_at_4'])} |")
    lines += [
        "",
        "LCB and distance minimize dev top-1 regret; diversity-LCB minimizes dev regret@4. All configurations, including unfavorable ones, remain in the frozen JSON.",
        "",
        "## Designed outer retrospective sensitivity (16 pools)",
        "",
        _context_table(designed),
        "",
        "Negative paired differences favor the method; positive differences favor the reference.",
        "",
        _interval_table(designed),
        "",
        "## Random outer retrospective sensitivity (3 eligible pools; secondary)",
        "",
        _context_table(random),
        "",
        _interval_table(random),
        "",
        "## Negative controls and risk–coverage",
        "",
        "| Context | Method | Observed regret | Permuted regret | Permutation 95% interval |",
        "|---|---|---:|---:|---|",
    ]
    for label, summary in (("designed", designed), ("random", random)):
        for row in summary["permutation_negative_controls"]:
            interval = row["permutation_95_percent_interval"]
            lines.append(f"| {label} | {row['method']} | {_number(row['observed_mean_regret'])} | {_number(row['permutation_mean_regret'])} | [{_number(interval[0])}, {_number(interval[1])}] |")
    lines += [
        "",
        "Thirty-two deterministic within-proposal permutations were retained for every method. Full risk–coverage rows, regret–cost curves, top-k values for k=1/2/4/8/16, per-member costs and all paired intervals are in the summary JSON files; the table above is not a selected subset of favorable configurations.",
        "",
        "## Conclusion",
        "",
        "Designed COM and diversity-LCB top-1 intervals cross zero, so neither demonstrates a regret advantage. Random outer COM is worse, and diversity-LCB is also numerically worse with an interval crossing zero; there is no context-general benefit. Because dev selected zero LCB/distance penalties, those methods equal BoN and their feature permutations are trivial negative controls.",
        "",
        "These are baseline and sensitivity results only. No claim of biological improvement, prospective validation, independent-study replication or official COM reproduction is made. Any unfavorable or interval-crossing result remains evidence against a method advantage; the preserved FP-REJ/Q gate failure is not reversed.",
        "",
        "## Verification",
        "",
        "Three complete software rounds passed (291 tests plus 106 subtests each); the final expanded delivery suite also passed (296 tests plus 106 subtests). COM checkpoints and every outer evaluation were independently checked in three rounds, including CUDA reload/dev-loss recomputation, raw regret/top-k recomputation, record completeness, frozen hyperparameters, permutation counts and source hashes.",
    ]
    Path(args.output).write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": args.output, "designed_pools": designed["pools"], "random_pools": random["pools"]}))


if __name__ == "__main__":
    main()
