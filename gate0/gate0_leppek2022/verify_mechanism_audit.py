import argparse
import json
import math
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from .common import load_config, read_tsv, sha256_file, write_json
from .mechanism_audit import CORE_OUTPUTS, build, compute_analysis


def close_enough(left, right, tolerance=1e-12):
    if left in (None, "") or right in (None, ""):
        return left in (None, "") and right in (None, "")
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def verify_numeric(output_dir, result):
    expected_correlations = read_tsv(output_dir / "paired_correlations.tsv")
    expected_ranks = read_tsv(output_dir / "rank_summary.tsv")
    if len(result["correlations"]) != 6 or len(expected_correlations) != 6:
        raise ValueError("Expected all six preregistered correlation pairs")
    for actual, stored in zip(result["correlations"], expected_correlations):
        for key in ["source_family", "left_endpoint", "right_endpoint", "complete_pair_rule", "inferential_claim"]:
            if actual[key] != stored[key]:
                raise ValueError(f"Correlation mismatch in {key}")
        if int(stored["n"]) != actual["n"]:
            raise ValueError("Correlation n mismatch")
        for key in ["pearson_r", "spearman_rho"]:
            if not close_enough(actual[key], stored[key]):
                raise ValueError(f"Correlation numeric mismatch in {key}")
    if len(result["rank_summaries"]) != 6 or len(expected_ranks) != 6:
        raise ValueError("Expected all six preregistered rank summaries")
    for actual, stored in zip(result["rank_summaries"], expected_ranks):
        for key in ["source_family", "left_endpoint", "right_endpoint", "left_top_members", "right_top_members", "left_only_top_members", "right_only_top_members"]:
            if actual[key] != stored[key]:
                raise ValueError(f"Rank summary mismatch in {key}")
        for key in ["n_constructs", "n_unordered_pairs", "concordant_pairs", "discordant_pairs", "tied_pairs", "top_k", "top_k_overlap_n"]:
            if int(stored[key]) != actual[key]:
                raise ValueError(f"Rank count mismatch in {key}")
        if not close_enough(actual["discordant_fraction_untied"], stored["discordant_fraction_untied"]):
            raise ValueError("Discordant fraction mismatch")
    cds = [row for row in result["correlations"] if row["source_family"] == "cds_24"]
    if any(row["n"] != 24 for row in cds):
        raise ValueError("CDS primary endpoints must have 24 complete pairs")
    for row in result["rank_summaries"]:
        if row["n_unordered_pairs"] != row["n_constructs"] * (row["n_constructs"] - 1) // 2:
            raise ValueError("Unordered pair count mismatch")
        if row["n_unordered_pairs"] != row["concordant_pairs"] + row["discordant_pairs"] + row["tied_pairs"]:
            raise ValueError("Pair relation counts do not add up")
    reversal_counts = {}
    for row in result["reversal_rows"]:
        key = (row["source_family"], row["left_endpoint"], row["right_endpoint"], row["relation"])
        reversal_counts[key] = reversal_counts.get(key, 0) + 1
    for row in result["rank_summaries"]:
        for relation, count_key in [("concordant", "concordant_pairs"), ("discordant", "discordant_pairs"), ("tied", "tied_pairs")]:
            key = (row["source_family"], row["left_endpoint"], row["right_endpoint"], relation)
            if reversal_counts.get(key, 0) != row[count_key]:
                raise ValueError("Detailed reversal rows disagree with summary")
    replicate_counts = {}
    for row in result["replicate_rows"]:
        replicate_counts.setdefault(row["endpoint"], []).append(row["replicate_n"])
    if len(replicate_counts.get("cds_in_solution_half_life", [])) != 24:
        raise ValueError("Missing half-life replicate audit")
    if replicate_counts["cds_in_solution_half_life"].count(2) != 1:
        raise ValueError("Expected exactly one half-life n=2 construct")
    if any(n != 4 for endpoint in ["nluc_fluc_6h", "nluc_fluc_24h"] for n in replicate_counts[endpoint]):
        raise ValueError("Expression replicate n must be four for every CDS")
    if result["duplicate_audit"]["all_257_exact_duplicate_groups"] != 0:
        raise ValueError("Unexpected exact duplicate groups")
    if result["duplicate_audit"]["duplicate_construct_endpoint_measurement_records"]:
        raise ValueError("Duplicate measurement records detected")
    if result["summary"]["predicted_expression_used_as_measured"]:
        raise ValueError("Predicted expression entered measured analysis")
    if any(row["source_family"] == "pooled_233" and row["endpoint"] == "measured_protein_expression" and row["measured_n"] != 0 for row in result["coverage_rows"]):
        raise ValueError("Pooled measured expression must be zero")
    report_text = (output_dir / "report.md").read_text()
    required_report_rows = [
        "| In-solution RNA half-life (h) | 1 | 20 | 3 |",
        "| Nluc / Fluc at 6 h | 0 | 0 | 24 |",
        "| Nluc / Fluc at 24 h | 0 | 0 | 24 |",
    ]
    for row in required_report_rows:
        if row not in report_text:
            raise ValueError(f"Report replicate table mismatch: {row}")
    return {
        "correlation_pairs": len(result["correlations"]),
        "cds_unordered_pairs_per_endpoint_pair": 276,
        "rank_reversal_rows": len(result["reversal_rows"]),
        "replicate_rows": len(result["replicate_rows"]),
        "source_name_mappings": len(result["mapping_rows"]),
        "coverage_rows": len(result["coverage_rows"]),
        "report_replicate_rows_verified": 3,
        "predicted_expression_as_measured": False,
    }


def verify_figure(output_dir):
    from PIL import Image

    png_path = output_dir / "leppek2022_cds_mechanism_audit.png"
    pdf_path = output_dir / "leppek2022_cds_mechanism_audit.pdf"
    svg_path = output_dir / "leppek2022_cds_mechanism_audit.svg"
    with Image.open(png_path) as image:
        image.load()
        if image.size != (4560, 2760):
            raise ValueError(f"Unexpected PNG dimensions: {image.size}")
        colors = image.convert("RGB").resize((456, 276)).getcolors(maxcolors=456 * 276)
        if not colors or len(colors) < 100:
            raise ValueError("PNG appears blank or lacks plotted content")
        png_details = {"width": image.width, "height": image.height, "mode": image.mode, "sampled_colors": len(colors)}
    pdf_bytes = pdf_path.read_bytes()
    if not pdf_bytes.startswith(b"%PDF-") or b"%%EOF" not in pdf_bytes[-2048:] or len(pdf_bytes) < 20000:
        raise ValueError("PDF is missing, empty or malformed")
    svg_root = ET.parse(svg_path).getroot()
    svg_text = " ".join(svg_root.itertext())
    required_text = ["Exploratory descriptive audit", "n = 24 CDS", "Pooled MRL vs in-cell stability", "Expression ranks reverse across time", "predicted expression is excluded"]
    for text in required_text:
        if text not in svg_text:
            raise ValueError(f"SVG is missing required text: {text}")
    return {
        "png": png_details,
        "pdf_bytes": len(pdf_bytes),
        "svg_root": svg_root.tag,
        "required_text_checks": len(required_text),
        "formats": 3,
    }


def verify_hashes(output_dir):
    manifest = json.loads((output_dir / "delivery_checksums.json").read_text())
    files = manifest["files"]
    if set(files) != set(CORE_OUTPUTS) - {"delivery_checksums.json"}:
        raise ValueError("Delivery checksum file set mismatch")
    for name, record in files.items():
        path = output_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise ValueError(f"Delivery hash mismatch: {name}")
    return {"algorithm": "sha256", "verified_files": len(files), "delivery_checksums_sha256": sha256_file(output_dir / "delivery_checksums.json")}


def main():
    parser = argparse.ArgumentParser(description="Verify numeric, figure and hash integrity for the Leppek CDS audit")
    parser.add_argument("--config", required=True)
    parser.add_argument("--round", type=int, required=True, choices=[1, 2, 3])
    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = Path(config["output_dir"])
    result = compute_analysis(config)
    numeric = verify_numeric(output_dir, result)
    figure = verify_figure(output_dir)
    hashes = verify_hashes(output_dir)
    reproduction = None
    if args.round == 3:
        with tempfile.TemporaryDirectory(prefix="leppek_audit_rebuild_") as temporary:
            rebuilt_path = Path(temporary) / "rebuilt"
            build(config, output_override=rebuilt_path)
            rebuilt_manifest = json.loads((rebuilt_path / "delivery_checksums.json").read_text())["files"]
            original_manifest = json.loads((output_dir / "delivery_checksums.json").read_text())["files"]
            if rebuilt_manifest != original_manifest:
                differing = sorted(name for name in original_manifest if rebuilt_manifest.get(name) != original_manifest[name])
                raise ValueError(f"Byte-identical rebuild failed: {differing}")
            reproduction = {"byte_identical_core_files": len(original_manifest), "status": "PASS"}
    report = {
        "dataset_id": config["dataset_id"],
        "round": args.round,
        "status": "PASS",
        "numeric_status": "PASS",
        "figure_status": "PASS",
        "hash_status": "PASS",
        "numeric": numeric,
        "figure": figure,
        "hashes": hashes,
        "reproduction": reproduction,
    }
    write_json(output_dir / f"verification_round{args.round}.json", report)
    rounds = []
    for round_number in [1, 2, 3]:
        path = output_dir / f"verification_round{round_number}.json"
        if path.is_file():
            rounds.append(json.loads(path.read_text()))
    write_json(output_dir / "verification_summary.json", {
        "dataset_id": config["dataset_id"],
        "required_rounds": 3,
        "completed_rounds": len(rounds),
        "all_pass": len(rounds) == 3 and all(item["status"] == "PASS" for item in rounds),
        "rounds": [{"round": item["round"], "numeric": item["numeric_status"], "figure": item["figure_status"], "hash": item["hash_status"], "status": item["status"]} for item in rounds],
    })
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
