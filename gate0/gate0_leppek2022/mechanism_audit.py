import argparse
import collections
import csv
import itertools
import json
import math
import os
from pathlib import Path

from .common import (
    canonical_json,
    checksum_directory,
    distribution,
    ensure_new_directory,
    finite_float,
    load_config,
    read_tsv,
    sha256_file,
    write_json,
    write_tsv,
)
from .xlsx import read_table

LABELS = {
    "ribosome_load": "Ribosome load (MRL)",
    "monosome_to_40s_60s_ratio": "Monosome / 40S+60S",
    "polysome_to_monosome_ratio": "Polysome / monosome",
    "polysome_to_40s_60s_ratio": "Polysome / 40S+60S",
    "in_cell_half_life": "In-cell RNA half-life (h)",
    "in_solution_half_life": "In-solution RNA half-life (h)",
    "cds_in_solution_half_life": "In-solution RNA half-life (h)",
    "nluc_fluc_6h": "Nluc / Fluc at 6 h",
    "nluc_fluc_24h": "Nluc / Fluc at 24 h",
    "measured_protein_expression": "Measured protein expression",
    "predicted_expression": "Predicted expression (not measured)",
}
SOURCE_NAME_OVERRIDES = {
    "START reference": "Nluc start",
    "LinearDesign-beamsize50-USETHISONE": "LinearDesign-beamsize50",
    "12.0": "12",
    "RiboTree_Nullrecurrent_winner_LinearDesign_degscoreall_7": "Ribotree_Nullrecurrent_winner_LinearDesign_degscoreall_7",
}
CORE_OUTPUTS = [
    "analysis_summary.json",
    "build_summary.json",
    "cds_construct_endpoints.tsv",
    "construct_ranks.tsv",
    "delivery_checksums.json",
    "duplicate_audit.json",
    "endpoint_coverage.tsv",
    "figure_source_data.json",
    "input_manifest.json",
    "leppek2022_cds_mechanism_audit.pdf",
    "leppek2022_cds_mechanism_audit.png",
    "leppek2022_cds_mechanism_audit.svg",
    "paired_correlations.tsv",
    "pooled_construct_endpoints.tsv",
    "rank_reversals.tsv",
    "rank_summary.tsv",
    "replicate_audit.tsv",
    "report.md",
    "source_name_mapping.tsv",
]


def pearson(left, right):
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = math.fsum(left) / len(left)
    right_mean = math.fsum(right) / len(right)
    left_delta = [value - left_mean for value in left]
    right_delta = [value - right_mean for value in right]
    numerator = math.fsum(a * b for a, b in zip(left_delta, right_delta))
    left_ss = math.fsum(value * value for value in left_delta)
    right_ss = math.fsum(value * value for value in right_delta)
    if left_ss == 0 or right_ss == 0:
        return None
    return numerator / math.sqrt(left_ss * right_ss)


def average_ranks(values, descending=False):
    ordered = sorted(enumerate(values), key=lambda item: item[1], reverse=descending)
    ranks = [None] * len(values)
    position = 0
    while position < len(ordered):
        end = position + 1
        while end < len(ordered) and ordered[end][1] == ordered[position][1]:
            end += 1
        rank = (position + 1 + end) / 2
        for index in range(position, end):
            ranks[ordered[index][0]] = rank
        position = end
    return ranks


def spearman(left, right):
    return pearson(average_ranks(left), average_ranks(right))


def top_members(rows, endpoint, k):
    ordered = sorted(rows, key=lambda row: (-row[endpoint], row["source_construct_id"]))
    return [row["source_construct_id"] for row in ordered[:k]]


def correlation_and_rank_rows(rows, family, left_endpoint, right_endpoint, top_k):
    paired = [row for row in rows if row.get(left_endpoint) is not None and row.get(right_endpoint) is not None]
    paired.sort(key=lambda row: row["source_construct_id"])
    left = [row[left_endpoint] for row in paired]
    right = [row[right_endpoint] for row in paired]
    correlation = {
        "source_family": family,
        "left_endpoint": left_endpoint,
        "right_endpoint": right_endpoint,
        "n": len(paired),
        "pearson_r": pearson(left, right),
        "spearman_rho": spearman(left, right),
        "complete_pair_rule": "both_measured_no_imputation",
        "inferential_claim": "none_descriptive_only",
    }
    reversal_rows = []
    concordant = discordant = tied = 0
    for first, second in itertools.combinations(paired, 2):
        left_delta = first[left_endpoint] - second[left_endpoint]
        right_delta = first[right_endpoint] - second[right_endpoint]
        product = left_delta * right_delta
        if left_delta == 0 or right_delta == 0:
            relation = "tied"
            tied += 1
        elif product < 0:
            relation = "discordant"
            discordant += 1
        else:
            relation = "concordant"
            concordant += 1
        reversal_rows.append({
            "source_family": family,
            "left_endpoint": left_endpoint,
            "right_endpoint": right_endpoint,
            "left_construct_id": first["source_construct_id"],
            "right_construct_id": second["source_construct_id"],
            "left_endpoint_delta": left_delta,
            "right_endpoint_delta": right_delta,
            "relation": relation,
        })
    denominator = concordant + discordant
    left_top = set(top_members(paired, left_endpoint, top_k))
    right_top = set(top_members(paired, right_endpoint, top_k))
    rank_summary = {
        "source_family": family,
        "left_endpoint": left_endpoint,
        "right_endpoint": right_endpoint,
        "n_constructs": len(paired),
        "n_unordered_pairs": len(reversal_rows),
        "concordant_pairs": concordant,
        "discordant_pairs": discordant,
        "tied_pairs": tied,
        "discordant_fraction_untied": discordant / denominator if denominator else None,
        "top_k": top_k,
        "top_k_overlap_n": len(left_top & right_top),
        "top_k_overlap_fraction_of_k": len(left_top & right_top) / top_k,
        "left_top_members": ";".join(sorted(left_top)),
        "right_top_members": ";".join(sorted(right_top)),
        "left_only_top_members": ";".join(sorted(left_top - right_top)),
        "right_only_top_members": ";".join(sorted(right_top - left_top)),
    }
    rank_rows = []
    left_ranks = average_ranks(left, descending=True)
    right_ranks = average_ranks(right, descending=True)
    for index, row in enumerate(paired):
        rank_rows.append({
            "source_family": family,
            "source_construct_id": row["source_construct_id"],
            "left_endpoint": left_endpoint,
            "right_endpoint": right_endpoint,
            "left_value": row[left_endpoint],
            "right_value": row[right_endpoint],
            "left_rank_descending": left_ranks[index],
            "right_rank_descending": right_ranks[index],
            "rank_difference_left_minus_right": left_ranks[index] - right_ranks[index],
        })
    return correlation, rank_summary, reversal_rows, rank_rows


def verify_inputs(config):
    gate0 = Path(config["gate0_root"])
    checksums_path = gate0 / "delivery_checksums.json"
    observed_checksums_sha = sha256_file(checksums_path)
    if observed_checksums_sha != config["gate0_delivery_checksums_sha256"]:
        raise ValueError("Gate 0 delivery checksum manifest hash mismatch")
    registered = json.loads(checksums_path.read_text())["files"]
    required = [
        "endpoint_measurements.tsv.gz", "endpoint_coverage.tsv.gz", "sequence_index.tsv.gz",
        "exact_duplicate_groups.tsv.gz", "near_duplicate_edges.tsv.gz",
    ]
    files = {}
    for name in required:
        path = gate0 / name
        observed = sha256_file(path)
        if observed != registered[name]["sha256"]:
            raise ValueError(f"Gate 0 input hash mismatch: {name}")
        files[name] = {"path": str(path), "sha256": observed, "bytes": path.stat().st_size}
    source_path = Path(config["source_data_path"])
    source_sha = sha256_file(source_path)
    if source_sha != config["source_data_sha256"]:
        raise ValueError("Source Data hash mismatch")
    files["source_data.xlsx"] = {"path": str(source_path), "sha256": source_sha, "bytes": source_path.stat().st_size}
    return {
        "gate0_delivery_checksums": {"path": str(checksums_path), "sha256": observed_checksums_sha},
        "files": files,
    }


def source_name_for(human_readable_name):
    return SOURCE_NAME_OVERRIDES.get(human_readable_name, human_readable_name)


def load_source_replicates(config):
    _, rows = read_table(config["source_data_path"], config["source_data_sheet"])
    result = {}
    for row in rows:
        name = row["name"]
        if name in result:
            raise ValueError(f"Duplicate Fig_4c source name: {name}")
        result[name] = {
            "source_data_name": name,
            "source_data_row": row["__row_number__"],
            "half_life_n": int(float(row["half_life_n"])),
            "nluc_fluc_6h_n": int(float(row["6h_n"])),
            "nluc_fluc_24h_n": int(float(row["24h_n"])),
            "source_half_life_norm_mean": finite_float(row["half_life_mean"]),
            "source_nluc_fluc_6h_norm_mean": finite_float(row["6h_mean"]),
            "source_nluc_fluc_24h_norm_mean": finite_float(row["24h_mean"]),
        }
    return result


def compute_analysis(config):
    input_manifest = verify_inputs(config)
    gate0 = Path(config["gate0_root"])
    measurements = read_tsv(gate0 / "endpoint_measurements.tsv.gz")
    coverage = read_tsv(gate0 / "endpoint_coverage.tsv.gz")
    sequences = read_tsv(gate0 / "sequence_index.tsv.gz")
    exact_groups = read_tsv(gate0 / "exact_duplicate_groups.tsv.gz")
    near_edges = read_tsv(gate0 / "near_duplicate_edges.tsv.gz")
    source_replicates = load_source_replicates(config)

    sequence_by_id = {row["sequence_id"]: row for row in sequences}
    if len(sequence_by_id) != 257:
        raise ValueError("Expected 257 unique Gate 0 sequence IDs")
    cds_sequences = sorted(
        [row for row in sequences if row["source_family"] == "cds_24"],
        key=lambda row: row["source_construct_id"],
    )
    pooled_sequences = sorted(
        [row for row in sequences if row["source_family"] == "pooled_233"],
        key=lambda row: row["source_construct_id"],
    )
    if len(cds_sequences) != 24 or len(pooled_sequences) != 233:
        raise ValueError("Unexpected Gate 0 family sizes")
    if len({row["source_construct_id"] for row in sequences}) != 257:
        raise ValueError("Duplicate source construct IDs")
    if len({row["normalized_sequence"] for row in sequences}) != 257:
        raise ValueError("Duplicate normalized sequences")

    measurement_keys = collections.Counter(
        (row["source_family"], row["source_construct_id"], row["endpoint"]) for row in measurements
    )
    duplicate_measurement_keys = sorted("|".join(key) for key, count in measurement_keys.items() if count != 1)
    if duplicate_measurement_keys:
        raise ValueError("Duplicate construct-endpoint measurement records")
    measurement_map = {
        (row["source_family"], row["source_construct_id"], row["endpoint"]): row for row in measurements
    }
    coverage_map = {
        (row["source_family"], row["source_construct_id"], row["endpoint"]): row for row in coverage
    }

    cds_rows = []
    replicate_rows = []
    mapping_rows = []
    seen_source_names = set()
    for sequence in cds_sequences:
        construct_id = sequence["source_construct_id"]
        source_name = source_name_for(sequence["human_readable_name"])
        if source_name not in source_replicates:
            raise ValueError(f"Missing Fig_4c mapping for {construct_id}: {source_name}")
        if source_name in seen_source_names:
            raise ValueError(f"Non-unique Fig_4c mapping: {source_name}")
        seen_source_names.add(source_name)
        source = source_replicates[source_name]
        row = {
            "sequence_id": sequence["sequence_id"],
            "source_construct_id": construct_id,
            "human_readable_name": sequence["human_readable_name"],
            "source_data_name": source_name,
            "category": sequence["category"],
            "cluster_id": sequence["cluster_id"],
            "source_row": sequence["source_row"],
        }
        for endpoint in config["cds_primary_endpoints"] + [
            "cds_in_solution_degradation_coefficient", "cds_half_life_norm_to_start",
            "nluc_fluc_6h_norm_to_start", "nluc_fluc_24h_norm_to_start",
        ]:
            measurement = measurement_map.get(("cds_24", construct_id, endpoint))
            row[endpoint] = finite_float(measurement["numeric_value"]) if measurement else None
            row[endpoint + "_error"] = finite_float(measurement["error_numeric_value"]) if measurement else None
            row[endpoint + "_raw"] = measurement["raw_value"] if measurement else ""
        mapping_rows.append({
            "source_construct_id": construct_id,
            "table_s5_name": sequence["human_readable_name"],
            "fig_4c_name": source_name,
            "table_s5_source_row": sequence["source_row"],
            "fig_4c_source_row": source["source_data_row"],
            "mapping_rule": "explicit_override" if sequence["human_readable_name"] in SOURCE_NAME_OVERRIDES else "exact_name",
        })
        for endpoint, n_key in [
            ("cds_in_solution_half_life", "half_life_n"),
            ("nluc_fluc_6h", "nluc_fluc_6h_n"),
            ("nluc_fluc_24h", "nluc_fluc_24h_n"),
        ]:
            measurement = measurement_map[("cds_24", construct_id, endpoint)]
            recorded_n = int(measurement["replicate_n"])
            if recorded_n != source[n_key]:
                raise ValueError(f"Replicate n mismatch for {construct_id} {endpoint}")
            replicate_rows.append({
                "source_construct_id": construct_id,
                "source_data_name": source_name,
                "endpoint": endpoint,
                "replicate_n": recorded_n,
                "replicate_source_file": measurement["replicate_source_file"],
                "source_data_row": source["source_data_row"],
                "independent_construct_n": 1,
                "interpretation": "published_biological_replicates_summarized_within_one_construct",
            })
        cds_rows.append(row)
    if seen_source_names != set(source_replicates):
        raise ValueError("Fig_4c rows are not mapped one-to-one to the 24 CDS constructs")

    pooled_rows = []
    for sequence in pooled_sequences:
        construct_id = sequence["source_construct_id"]
        row = {
            "sequence_id": sequence["sequence_id"],
            "source_construct_id": construct_id,
            "human_readable_name": sequence["human_readable_name"],
            "category": sequence["category"],
        }
        for endpoint in config["pooled_context_endpoints"]:
            measurement = measurement_map.get(("pooled_233", construct_id, endpoint))
            row[endpoint] = finite_float(measurement["numeric_value"]) if measurement else None
            row[endpoint + "_error"] = finite_float(measurement["error_numeric_value"]) if measurement else None
        pooled_rows.append(row)

    coverage_rows = []
    coverage_specs = [("pooled_233", endpoint, 233) for endpoint in config["pooled_context_endpoints"]]
    coverage_specs += [("cds_24", endpoint, 24) for endpoint in config["cds_primary_endpoints"]]
    for family, endpoint, total in coverage_specs:
        family_rows = [row for row in coverage if row["source_family"] == family and row["endpoint"] == endpoint]
        measured = [row for row in family_rows if row["status"] == "measured" and finite_float(row["numeric_value"]) is not None]
        errors = [row for row in measured if finite_float(row["error_numeric_value"]) is not None]
        coverage_rows.append({
            "source_family": family,
            "endpoint": endpoint,
            "endpoint_label": LABELS[endpoint],
            "construct_total": total,
            "measured_n": len(measured),
            "missing_n": total - len(measured),
            "coverage_fraction": len(measured) / total,
            "error_n": len(errors),
            "availability": "measured",
            "note": "complete_case_pairs_no_imputation",
        })
    predicted = [row for row in coverage if row["source_family"] == "pooled_233" and row["endpoint"] == "predicted_expression" and row["status"] == "predicted"]
    coverage_rows.extend([
        {
            "source_family": "pooled_233", "endpoint": "measured_protein_expression",
            "endpoint_label": LABELS["measured_protein_expression"], "construct_total": 233,
            "measured_n": 0, "missing_n": 233, "coverage_fraction": 0.0, "error_n": 0,
            "availability": "not_assayed", "note": "no_measured_protein_expression_column",
        },
        {
            "source_family": "pooled_233", "endpoint": "predicted_expression",
            "endpoint_label": LABELS["predicted_expression"], "construct_total": 233,
            "measured_n": 0, "missing_n": 233 - len(predicted), "coverage_fraction": 0.0,
            "error_n": 0, "availability": "predicted_not_measured", "predicted_n": len(predicted),
            "note": "never_used_as_measured_expression",
        },
        {
            "source_family": "cds_24", "endpoint": "ribosome_load", "endpoint_label": LABELS["ribosome_load"],
            "construct_total": 24, "measured_n": 0, "missing_n": 24, "coverage_fraction": 0.0,
            "error_n": 0, "availability": "not_assayed", "note": "cds_table_has_no_mrl_measurement",
        },
    ])

    correlations = []
    rank_summaries = []
    reversal_rows = []
    rank_rows = []
    for left, right in itertools.combinations(config["cds_primary_endpoints"], 2):
        result = correlation_and_rank_rows(cds_rows, "cds_24", left, right, config["rank_reversal"]["top_k"])
        correlations.append(result[0]); rank_summaries.append(result[1]); reversal_rows.extend(result[2]); rank_rows.extend(result[3])
    for left, right in config["pooled_correlation_pairs"]:
        result = correlation_and_rank_rows(pooled_rows, "pooled_233", left, right, config["rank_reversal"]["top_k"])
        correlations.append(result[0]); rank_summaries.append(result[1]); reversal_rows.extend(result[2]); rank_rows.extend(result[3])

    cds_ids = {row["sequence_id"] for row in cds_sequences}
    cds_near_edges = [row for row in near_edges if row["left_sequence_id"] in cds_ids and row["right_sequence_id"] in cds_ids]
    near_counts = collections.Counter()
    for row in cds_near_edges:
        near_counts[row["left_source_construct_id"]] += 1
        near_counts[row["right_source_construct_id"]] += 1
    for row in cds_rows:
        row["cds_near_duplicate_degree"] = near_counts[row["source_construct_id"]]
    duplicate_audit = {
        "all_257_exact_duplicate_groups": len(exact_groups),
        "all_257_near_duplicate_edges": len(near_edges),
        "all_257_unique_construct_ids": len(sequence_by_id) == len({row["source_construct_id"] for row in sequences}),
        "all_257_unique_normalized_sequences": len({row["normalized_sequence"] for row in sequences}) == 257,
        "cds_24_exact_duplicate_groups": 0,
        "cds_24_near_duplicate_edges": len(cds_near_edges),
        "cds_24_near_duplicate_construct_n": len(near_counts),
        "cds_24_near_duplicate_edges_detail": cds_near_edges,
        "duplicate_construct_endpoint_measurement_records": duplicate_measurement_keys,
        "published_replicate_rows_not_independent_constructs": True,
    }

    cds_distributions = {
        endpoint: distribution([row[endpoint] for row in cds_rows if row[endpoint] is not None])
        for endpoint in config["cds_primary_endpoints"]
    }
    pooled_distributions = {
        endpoint: distribution([row[endpoint] for row in pooled_rows if row[endpoint] is not None])
        for endpoint in config["pooled_context_endpoints"]
    }
    replicate_n_distribution = {
        endpoint: dict(sorted(collections.Counter(row["replicate_n"] for row in replicate_rows if row["endpoint"] == endpoint).items()))
        for endpoint in config["cds_primary_endpoints"]
    }
    summary = {
        "dataset_id": config["dataset_id"],
        "preregistered_at": config["preregistered_at"],
        "preregistration_commit": config.get("preregistration_commit", "cc706aa16c56d5e83a0e9362c873de938f673eca"),
        "supersedes": config.get("supersedes"),
        "scope": "exploratory_descriptive_mechanism_audit_not_generalizable_inference",
        "model_training_or_inference": False,
        "outer_results_accessed": False,
        "predicted_expression_used_as_measured": False,
        "gpu_inference": False,
        "cds_construct_n": len(cds_rows),
        "pooled_construct_n": len(pooled_rows),
        "cds_distributions": cds_distributions,
        "pooled_distributions": pooled_distributions,
        "replicate_n_distribution": replicate_n_distribution,
        "correlations": correlations,
        "rank_summaries": rank_summaries,
        "duplicate_audit": duplicate_audit,
        "coverage": coverage_rows,
        "constraints": [
            "CDS results are restricted to 24 HBB-Nluc constructs, PSU, HEK293T Nluc/Fluc and in-solution stability.",
            "Pooled MRL/stability and CDS expression are separate construct sets and are never joined as paired endpoints.",
            "Predicted expression is not a protein measurement.",
            "No p values, confidence intervals, model fitting, outer tuning or significance claims.",
        ],
    }
    return {
        "input_manifest": input_manifest,
        "cds_rows": cds_rows,
        "pooled_rows": pooled_rows,
        "coverage_rows": coverage_rows,
        "correlations": correlations,
        "rank_summaries": rank_summaries,
        "reversal_rows": reversal_rows,
        "rank_rows": rank_rows,
        "replicate_rows": replicate_rows,
        "mapping_rows": mapping_rows,
        "duplicate_audit": duplicate_audit,
        "summary": summary,
    }


def format_number(value, digits=3):
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def make_report(result):
    summary = result["summary"]
    config_name = "leppek2022_cds_mechanism_audit_" + summary["dataset_id"].rsplit("-", 1)[-1] + ".json"
    lines = [
        "# Leppek 2022：24 条 CDS 小样本跨终点描述性机制审计",
        "",
        f"版本：`{summary['dataset_id']}`；预注册时间：`{summary['preregistered_at']}`；预注册提交：`{summary['preregistration_commit']}`。本报告是**探索性描述**，不构成泛化显著性、因果或模型选择证据。",
        "",
        "v1–v4 草稿完整保留；此前版本依次修正字体与代码指纹、渲染布局、SVG 逐字节复建。v4 报告的重复数表误用字符串键读取内存整数键而显示 0，v5 修正该展示错误并把报告数值纳入逐轮验证。底层测量、统计方法与相关/排序结果不变，历史失败与错误版本未删除。",
        "",
        "## 核心结论",
        "",
        "24 条 CDS 的溶液 RNA 半衰期与 6h、24h 实测 Nluc/Fluc 均可完整配对，但三种终点的相关和排序并不等价；全部数值、负相关与排序反转均列于下表。233 条 pooled 构建体只提供 MRL—稳定性背景，不能与 24 条 CDS 拼成 MRL—表达配对。pooled 的 `Predicted expression` 从未作为实测表达使用。",
        "",
        "样本仅限 HBB-Nluc CDS、PSU、HEK293T 和已发表溶液稳定性实验。未训练模型、未运行推理、未访问 outer 结果、未计算 p 值或置信区间，也未按结果方向筛选构建体。",
        "",
        "## 覆盖与重复测量",
        "",
        "- CDS：24/24 有溶液半衰期、6h Nluc/Fluc 和 24h Nluc/Fluc；MRL 未测（0/24）。",
        "- pooled：233/233 有 ribosome load 和三种 polysome 比率；细胞内半衰期 203/233，溶液半衰期 191/233；实测蛋白表达 0/233。203 个 `Predicted expression` 仅计为预测。",
        "- 24 个 CDS 的 6h 和 24h 表达均为 n=4；溶液半衰期的精确 n 分布见下表。重复测量先在原研究中汇总到构建体，本审计的独立构建体数始终为 24。",
        "- 24 个 CDS 的构建体 ID、全长序列和 construct-endpoint measurement 均不重复；exact duplicate group 为 0。近重复边另行保留，不能声称序列完全独立。",
        "",
        "| 终点 | n=2 | n=3 | n=4 |",
        "|---|---:|---:|---:|",
    ]
    for endpoint in ["cds_in_solution_half_life", "nluc_fluc_6h", "nluc_fluc_24h"]:
        counts = summary["replicate_n_distribution"][endpoint]
        lines.append(f"| {LABELS[endpoint]} | {counts.get(2, counts.get('2', 0))} | {counts.get(3, counts.get('3', 0))} | {counts.get(4, counts.get('4', 0))} |")
    lines += [
        "",
        "逐构建体 n、来源行、显式名称映射及原始/START 归一化字段见 `replicate_audit.tsv`、`source_name_mapping.tsv` 和 `cds_construct_endpoints.tsv`。",
        "",
        "## 预注册配对相关",
        "",
        "Pearson r 描述线性，Spearman rho 描述单调排序；均为完整配对、无插补的描述统计，不附显著性判断。",
        "",
        "| 集合 | 终点 A | 终点 B | n | Pearson r | Spearman rho |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in summary["correlations"]:
        lines.append(
            f"| {row['source_family']} | {LABELS[row['left_endpoint']]} | {LABELS[row['right_endpoint']]} | {row['n']} | {format_number(row['pearson_r'])} | {format_number(row['spearman_rho'])} |"
        )
    lines += [
        "",
        "## 排序反转与 top-6 重叠",
        "",
        "每个 CDS 配对枚举 276 个无序构建体对。反转比例的分母为 concordant + discordant；本数据无 ties。top-6 是预注册的固定 25%，不是按结果调整的阈值。",
        "",
        "| 终点 A | 终点 B | 构建体对 | 反转对 | 反转比例 | top-6 重叠 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary["rank_summaries"]:
        if row["source_family"] != "cds_24":
            continue
        lines.append(
            f"| {LABELS[row['left_endpoint']]} | {LABELS[row['right_endpoint']]} | {row['n_unordered_pairs']} | {row['discordant_pairs']} | {format_number(row['discordant_fraction_untied'])} | {row['top_k_overlap_n']}/6 |"
        )
    lines += [
        "",
        "逐对方向、逐构建体平均秩、top-6 独有成员及 pooled 背景排序均完整保存在 `rank_reversals.tsv`、`construct_ranks.tsv` 和 `rank_summary.tsv`。排序反转只表示终点排序不同，不证明机制冲突。",
        "",
        "## 缺失、重复和解释限制",
        "",
        f"CDS 内近重复边为 {summary['duplicate_audit']['cds_24_near_duplicate_edges']} 条，涉及 {summary['duplicate_audit']['cds_24_near_duplicate_construct_n']} 条构建体；所有 257 条 Gate 0 主序列的近重复边共 {summary['duplicate_audit']['all_257_near_duplicate_edges']} 条。没有 exact duplicate，不代表共享 UTR/CDS 组件或实验条件独立。",
        "",
        " pooled 稳定性拟合缺失按原样保留，相关使用完整配对，不做零值、均值或模型插补。CDS 主图和主表使用 Table S5 的原始半衰期/Nluc-Fluc 均值及 SD；Source Data Fig_4c 用于精确 n 和名称核验，其 START 归一化值不覆盖原始字段。",
        "",
        "## 图形与 source data",
        "",
        "`leppek2022_cds_mechanism_audit.{png,pdf,svg}` 为 2×3 多面板图：覆盖、pooled MRL—两种稳定性、CDS 半衰期—6h/24h 表达、6h—24h 表达排序。每面板标明 n 和 exploratory；CDS 散点误差棒为已发表 SD，不显示拟合趋势线。",
        "",
        "聚合证据包括 24 行 CDS、233 行 pooled 的作图数据，以及覆盖、相关、反转、秩、重复测量和名称映射表。`figure_source_data.json` 固定面板—文件—字段关系；`input_manifest.json` 和 `delivery_checksums.json` 固定输入与输出 SHA256。",
        "",
        "## 可复现与验证",
        "",
        "```bash",
        "cd /home/proxy-function-gap",
        "export PYTHONPATH=/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/leppek2022_cds_mechanism_audit/python_deps:gate0",
        f".venv/bin/python -m gate0_leppek2022.mechanism_audit --config gate0/configs/{config_name}",
        f".venv/bin/python -m gate0_leppek2022.verify_mechanism_audit --config gate0/configs/{config_name} --round 1",
        f".venv/bin/python -m gate0_leppek2022.verify_mechanism_audit --config gate0/configs/{config_name} --round 2",
        f".venv/bin/python -m gate0_leppek2022.verify_mechanism_audit --config gate0/configs/{config_name} --round 3",
        "```",
        "",
        "三轮均独立重算数值、检查 PNG/PDF/SVG 和核验交付哈希；第三轮在新临时目录完整重建并逐字节比对核心产物。构建器拒绝覆盖既有输出。验证结果见 `verification_round*.json` 与 `verification_summary.json`。",
        "",
    ]
    return "\n".join(lines)


def write_analysis(result, output_dir, config):
    output_dir = Path(output_dir)
    write_json(output_dir / "input_manifest.json", result["input_manifest"])
    write_json(output_dir / "analysis_summary.json", result["summary"])
    write_json(output_dir / "duplicate_audit.json", result["duplicate_audit"])
    write_tsv(output_dir / "cds_construct_endpoints.tsv", list(result["cds_rows"][0]), result["cds_rows"])
    write_tsv(output_dir / "pooled_construct_endpoints.tsv", list(result["pooled_rows"][0]), result["pooled_rows"])
    write_tsv(output_dir / "endpoint_coverage.tsv", list(result["coverage_rows"][0]), result["coverage_rows"])
    write_tsv(output_dir / "paired_correlations.tsv", list(result["correlations"][0]), result["correlations"])
    write_tsv(output_dir / "rank_summary.tsv", list(result["rank_summaries"][0]), result["rank_summaries"])
    write_tsv(output_dir / "rank_reversals.tsv", list(result["reversal_rows"][0]), result["reversal_rows"])
    write_tsv(output_dir / "construct_ranks.tsv", list(result["rank_rows"][0]), result["rank_rows"])
    write_tsv(output_dir / "replicate_audit.tsv", list(result["replicate_rows"][0]), result["replicate_rows"])
    write_tsv(output_dir / "source_name_mapping.tsv", list(result["mapping_rows"][0]), result["mapping_rows"])
    figure_source = {
        "figure": "leppek2022_cds_mechanism_audit",
        "exploratory": True,
        "panels": {
            "A": {"source": "endpoint_coverage.tsv", "content": "measured coverage and structural absence"},
            "B": {"source": "pooled_construct_endpoints.tsv", "x": "ribosome_load", "y": "in_cell_half_life"},
            "C": {"source": "pooled_construct_endpoints.tsv", "x": "ribosome_load", "y": "in_solution_half_life"},
            "D": {"source": "cds_construct_endpoints.tsv", "x": "cds_in_solution_half_life", "y": "nluc_fluc_6h", "errors": "published_SD"},
            "E": {"source": "cds_construct_endpoints.tsv", "x": "cds_in_solution_half_life", "y": "nluc_fluc_24h", "errors": "published_SD"},
            "F": {"source": "construct_ranks.tsv", "content": "nluc_fluc_6h versus nluc_fluc_24h descending average ranks"},
        },
    }
    write_json(output_dir / "figure_source_data.json", figure_source)
    (output_dir / "report.md").write_text(make_report(result))


def render_figure(result, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir = Path(output_dir)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.9,
        "axes.titleweight": "bold",
        "legend.frameon": False,
        "svg.fonttype": "none",
        "svg.hashsalt": "leppek2022-cds-cross-endpoint-audit",
        "pdf.fonttype": 42,
    })
    blue = "#0F4D92"
    teal = "#42949E"
    orange = "#C56B32"
    neutral = "#767676"
    light = "#E8EDF2"
    fig = plt.figure(figsize=(15.2, 9.2), facecolor="white")
    grid = fig.add_gridspec(2, 3, left=0.065, right=0.985, bottom=0.105, top=0.80, wspace=0.32, hspace=0.52)
    fig.text(0.065, 0.955, "Leppek 2022 | Cross-endpoint measurements do not define one ranking", fontsize=18, weight="bold", color="#272727")
    fig.text(0.065, 0.918, "Exploratory descriptive audit • real measured endpoints • no imputation, model fitting or significance testing", fontsize=10.5, color=neutral)
    fig.text(0.065, 0.885, "Pooled MRL/stability (n ≤ 233) and CDS stability/expression (n = 24) are separate construct sets; predicted expression is excluded.", fontsize=9.5, color=neutral)

    def panel_label(ax, label, title):
        ax.text(-0.14, 1.09, label, transform=ax.transAxes, fontsize=14, weight="bold", color="#272727")
        ax.set_title(title, loc="left", pad=12, fontsize=10.5)

    def get_correlation(family, left, right):
        return next(row for row in result["correlations"] if row["source_family"] == family and row["left_endpoint"] == left and row["right_endpoint"] == right)

    ax = fig.add_subplot(grid[0, 0])
    panel_label(ax, "A", "Measured coverage and structural gaps")
    coverage_specs = [
        ("Pooled | MRL", 233, 233, blue),
        ("Pooled | in-cell t½", 203, 233, teal),
        ("Pooled | in-solution t½", 191, 233, teal),
        ("Pooled | measured expression", 0, 233, orange),
        ("CDS | in-solution t½", 24, 24, teal),
        ("CDS | Nluc/Fluc 6 h", 24, 24, blue),
        ("CDS | Nluc/Fluc 24 h", 24, 24, blue),
        ("CDS | MRL", 0, 24, orange),
    ]
    y_positions = np.arange(len(coverage_specs))[::-1]
    for y, (label, n, total, color) in zip(y_positions, coverage_specs):
        ax.barh(y, 100, color=light, height=0.62)
        ax.barh(y, 100 * n / total, color=color, height=0.62)
        ax.text(102, y, f"{n}/{total}", va="center", fontsize=8.5, color="#272727")
    ax.set_yticks(y_positions, [item[0] for item in coverage_specs])
    ax.set_xlim(0, 119)
    ax.set_xticks([0, 50, 100], ["0", "50", "100%"])
    ax.set_xlabel("Constructs with measured endpoint")
    ax.tick_params(axis="y", labelsize=8.2)
    ax.text(0, -0.23, "Pooled predicted expression (n = 203) is not shown as measured.", transform=ax.transAxes, fontsize=8, color=orange)

    def scatter_pooled(ax, label, endpoint, title):
        rows = [row for row in result["pooled_rows"] if row["ribosome_load"] is not None and row[endpoint] is not None]
        x = [row["ribosome_load"] for row in rows]
        y = [row[endpoint] for row in rows]
        ax.scatter(x, y, s=17, color=teal, alpha=0.62, edgecolors="white", linewidths=0.25)
        corr = get_correlation("pooled_233", "ribosome_load", endpoint)
        ax.text(0.03, 0.97, f"n = {len(rows)}\nPearson r = {corr['pearson_r']:.3f}\nSpearman ρ = {corr['spearman_rho']:.3f}", transform=ax.transAxes, va="top", fontsize=8.5, color="#272727", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 3})
        ax.set_xlabel("Ribosome load (MRL)")
        ax.set_ylabel(LABELS[endpoint])
        panel_label(ax, label, title)
        ax.text(0.98, 0.02, "exploratory", transform=ax.transAxes, ha="right", fontsize=7.5, color=neutral)

    scatter_pooled(fig.add_subplot(grid[0, 1]), "B", "in_cell_half_life", "Pooled MRL vs in-cell stability")
    scatter_pooled(fig.add_subplot(grid[0, 2]), "C", "in_solution_half_life", "Pooled MRL vs in-solution stability")

    def scatter_cds(ax, label, endpoint, title):
        ax = ax
        rows = result["cds_rows"]
        x = [row["cds_in_solution_half_life"] for row in rows]
        y = [row[endpoint] for row in rows]
        xerr = [row["cds_in_solution_half_life_error"] for row in rows]
        yerr = [row[endpoint + "_error"] for row in rows]
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="none", ecolor="#B9C5CC", elinewidth=0.6, capsize=0, zorder=1)
        ordinary = [row for row in rows if row["source_data_name"] != "Nluc start"]
        start = [row for row in rows if row["source_data_name"] == "Nluc start"]
        ax.scatter([row["cds_in_solution_half_life"] for row in ordinary], [row[endpoint] for row in ordinary], s=38, color=teal, alpha=0.88, edgecolors="white", linewidths=0.6, zorder=2)
        ax.scatter([row["cds_in_solution_half_life"] for row in start], [row[endpoint] for row in start], s=68, color=orange, marker="D", edgecolors="white", linewidths=0.7, zorder=3, label="START reference")
        corr = get_correlation("cds_24", "cds_in_solution_half_life", endpoint)
        ax.text(0.03, 0.97, f"n = {len(rows)} CDS\nPearson r = {corr['pearson_r']:.3f}\nSpearman ρ = {corr['spearman_rho']:.3f}", transform=ax.transAxes, va="top", fontsize=8.5, color="#272727", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 3})
        ax.set_xlabel("In-solution RNA half-life (h)")
        ax.set_ylabel(LABELS[endpoint])
        ax.set_yscale("log")
        panel_label(ax, label, title)
        ax.legend(loc="lower left", fontsize=7.5)
        ax.text(0.98, 0.02, "mean ± SD • exploratory", transform=ax.transAxes, ha="right", fontsize=7.5, color=neutral)

    scatter_cds(fig.add_subplot(grid[1, 0]), "D", "nluc_fluc_6h", "CDS stability vs expression at 6 h")
    scatter_cds(fig.add_subplot(grid[1, 1]), "E", "nluc_fluc_24h", "CDS stability vs expression at 24 h")

    ax = fig.add_subplot(grid[1, 2])
    panel_label(ax, "F", "Expression ranks reverse across time")
    rows = [row for row in result["rank_rows"] if row["source_family"] == "cds_24" and row["left_endpoint"] == "nluc_fluc_6h" and row["right_endpoint"] == "nluc_fluc_24h"]
    summary = next(row for row in result["rank_summaries"] if row["source_family"] == "cds_24" and row["left_endpoint"] == "nluc_fluc_6h" and row["right_endpoint"] == "nluc_fluc_24h")
    for row in rows:
        ax.plot([0, 1], [row["left_rank_descending"], row["right_rank_descending"]], color="#C7D0D6", linewidth=0.75, alpha=0.8, zorder=1)
    ax.scatter([0] * len(rows), [row["left_rank_descending"] for row in rows], s=24, color=blue, zorder=2)
    ax.scatter([1] * len(rows), [row["right_rank_descending"] for row in rows], s=24, color=teal, zorder=2)
    ax.set_xlim(-0.22, 1.22)
    ax.set_ylim(24.8, 0.2)
    ax.set_xticks([0, 1], ["6 h", "24 h"])
    ax.set_ylabel("Expression rank (1 = highest)")
    ax.set_yticks([1, 6, 12, 18, 24])
    ax.text(0.5, 0.97, f"n = 24 CDS • {summary['discordant_pairs']}/{summary['n_unordered_pairs']} pairs reversed\nTop-6 overlap = {summary['top_k_overlap_n']}/6 • Spearman ρ = {get_correlation('cds_24', 'nluc_fluc_6h', 'nluc_fluc_24h')['spearman_rho']:.3f}", transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color="#272727", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 3})
    ax.text(0.98, 0.02, "one line per CDS • exploratory", transform=ax.transAxes, ha="right", fontsize=7.5, color=neutral)

    fig.text(0.065, 0.035, "Source: Leppek et al. (2022), GSE173083 Table S1/S5 and Source Data Fig_4c. CDS points use raw Table S5 means; error bars are published SD.\nNo p values or confidence intervals. Rank reversals are descriptive ordering differences, not causal claims.", fontsize=8.5, color=neutral, linespacing=1.5)
    metadata = {"Title": "Leppek 2022 exploratory cross-endpoint mechanism audit", "Author": "", "Subject": "Descriptive measured-endpoint audit", "Keywords": "Leppek, CDS, MRL, stability, expression, exploratory", "CreationDate": None, "ModDate": None}
    fig.savefig(output_dir / "leppek2022_cds_mechanism_audit.png", dpi=300, facecolor="white", metadata={"Software": "matplotlib"})
    fig.savefig(output_dir / "leppek2022_cds_mechanism_audit.pdf", facecolor="white", metadata=metadata)
    fig.savefig(output_dir / "leppek2022_cds_mechanism_audit.svg", facecolor="white", metadata={"Date": None, "Creator": "matplotlib"})
    plt.close(fig)


def build(config, output_override=None):
    output_dir = Path(output_override or config["output_dir"])
    ensure_new_directory(output_dir)
    result = compute_analysis(config)
    write_analysis(result, output_dir, config)
    render_figure(result, output_dir)
    build_summary = {
        "dataset_id": config["dataset_id"],
        "preregistered_at": config["preregistered_at"],
        "preregistration_commit": config.get("preregistration_commit", "cc706aa16c56d5e83a0e9362c873de938f673eca"),
        "analysis_code_sha256": sha256_file(Path(__file__)),
        "verification_code_sha256": sha256_file(Path(__file__).with_name("verify_mechanism_audit.py")),
        "cds_constructs": len(result["cds_rows"]),
        "pooled_constructs": len(result["pooled_rows"]),
        "correlation_pairs": len(result["correlations"]),
        "rank_reversal_rows": len(result["reversal_rows"]),
        "model_runs": 0,
        "outer_results_accessed": False,
        "predicted_expression_as_measured": False,
        "compute": "CPU_only",
        "verification_required_rounds": config["verification_rounds"],
    }
    write_json(output_dir / "build_summary.json", build_summary)
    core_files = [name for name in CORE_OUTPUTS if name != "delivery_checksums.json"]
    checksums = checksum_directory(output_dir, exclude={"delivery_checksums.json"})
    if set(checksums) != set(core_files):
        raise ValueError(f"Unexpected build outputs: {set(checksums) ^ set(core_files)}")
    write_json(output_dir / "delivery_checksums.json", {"algorithm": "sha256", "files": checksums})
    return result


def main():
    parser = argparse.ArgumentParser(description="Build the preregistered read-only Leppek CDS mechanism audit")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    result = build(config)
    print(json.dumps({
        "status": "PASS",
        "dataset_id": config["dataset_id"],
        "output_dir": config["output_dir"],
        "cds_constructs": len(result["cds_rows"]),
        "pooled_constructs": len(result["pooled_rows"]),
        "model_runs": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
