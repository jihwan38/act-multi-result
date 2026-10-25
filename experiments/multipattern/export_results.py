#!/usr/bin/env python3
"""Archive completed three-way measurements without dropping failed/warm-up rows.

Uses only recorded JSON evidence; it never builds or runs a compiler. The raw
archive retains complete sample records, structured checks, manifests, and logs
for failed samples. CSVs contain the three final measurement sets only.
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path


MODES = ("multi-result", "multipattern", "multipattern-rewrite")
FINAL_SETS = (
    ("probe", "final-default"),
    ("probe", "final-tight"),
    ("micro", "final-default"),
)
DEFAULT_LIMITS = {
    "ACT_JOINT_LIMIT": 64, "ACT_JOINT_RAW_LIMIT": 4096,
    "ACT_JOINT_SEARCH_LIMIT": 4096, "ACT_JOINT_ATTEMPT_LIMIT": 4096,
}
TIGHT_LIMITS = {key: 2 if key == "ACT_JOINT_LIMIT" else 16 for key in DEFAULT_LIMITS}
TIGHT_CASES = {"shared_pair_concat", "shared_triple_concat", "shared_reshape"}
MICRO_SCENARIOS = {
    "shared": [(name, size) for name in ("shared", "disjoint") for size in (8, 32, 128)]
              + [("dependent", 8), ("wrong-shape", 32), ("metadata", 8)],
    "independent": [("independent", size) for size in (8, 32, 128)],
}
WORK_FIELDS = ("search_calls", "raw_matches", "replayed_matches", "attempts", "accepted", "registered")
BASE_FIELDS = (
    "set", "toolchain", "repetition", "warmup", "order", "passed", "exit_status", "timed_out",
    "wall_seconds", "gnu_time_max_rss_kib", "binary_sha256", "source_fingerprint",
    "log",
)
WORK_COLUMNS = WORK_FIELDS + (
    "search_counter_scope", "last_search_complete", "last_search_stop_reason",
)
COMPILER_FIELDS = BASE_FIELDS + (
    "case", "correctness_pass", "selection_expectations_pass", "has_selection_expectation",
    "quality", "oracle_passed", "allocation_passed", "model_cost", "api_call_count",
    "candidate_pii_count", "api_calls", "hlo_sha256", "assembly_sha256", "joint_limit_scope",
    "joint_limit", "raw_limit", "search_limit", "attempt_limit", "direct_applications",
) + WORK_COLUMNS
MICRO_FIELDS = BASE_FIELDS + (
    "profile", "scenario", "size", "semantic_assertions_passed",
    "joint_candidate_pass_ms", "discovery_ms", "rules_init_ms", "rule_search_ms",
    "rule_apply_rebuild_ms", "rule_search_substitutions", "registered_tuple_count",
    "implemented_tuple_count", "joint_isa_nodes", "original_nodes", "final_nodes",
    "theoretical_valid_pairs", "theoretical_valid_triples",
) + WORK_COLUMNS


def read_json(path):
    if not path.is_file():
        raise ValueError(f"Missing evidence: {path}")
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def completed_set(directory, family, final=False):
    config = read_json(directory / "configuration.json")
    if not config.get("finished_utc"):
        raise ValueError(f"Measurement/check set is unfinished: {directory}")
    samples = read_json(directory / "samples.json")
    summary = read_json(directory / "summary.json")
    repeats, warmups = config["repeats"], config["warmups"]
    if repeats < 1 or warmups < 0 or (final and (repeats < 5 or warmups < 1)):
        raise ValueError(f"Invalid repetition configuration: {directory}")
    rounds = range(-warmups, repeats)
    if family == "micro":
        groups = {(row["profile"], row["scenario"], row["size"]): set(row["variants"]) for row in summary}
        if final:
            expected_groups = {(profile, scenario, size) for profile, scenarios in MICRO_SCENARIOS.items()
                               for scenario, size in scenarios}
            if set(groups) != expected_groups or any(modes != set(MODES) for modes in groups.values()):
                raise ValueError(f"Final microbenchmark coverage differs from the planned matrix: {directory}")
        expected = {(*group, mode, repetition) for group, modes in groups.items()
                    for mode in modes for repetition in rounds}
        identity = lambda row: (row["profile"], row["scenario"], row["size"], row["toolchain"], row["repetition"])
    else:
        groups = {row["case"]: set(row["variants"]) for row in summary}
        if set(groups) != set(config["cases"]) or any(modes != set(config["toolchains"]) for modes in groups.values()):
            raise ValueError(f"Compiler summary/configuration coverage differs: {directory}")
        if final and set(config["toolchains"]) != set(MODES):
            raise ValueError(f"Final compiler set must include all three variants: {directory}")
        expected = {(case, mode, repetition) for case in config["cases"]
                    for mode in config["toolchains"] for repetition in rounds}
        identity = lambda row: (row["case"], row["toolchain"], row["repetition"])
    if len(groups) != len(summary):
        raise ValueError(f"Duplicate summary groups: {directory}")
    actual = set()
    for row in samples:
        key = identity(row)
        if key in actual:
            raise ValueError(f"Duplicate sample {key} in {directory}")
        actual.add(key)
        if not isinstance(row.get("passed"), bool) or row.get("warmup") != (row["repetition"] < 0):
            raise ValueError(f"Incomplete sample status {key} in {directory}")
    if actual != expected:
        raise ValueError(f"Incomplete/unexpected sample matrix in {directory}: "
                         f"missing={len(expected - actual)}, extra={len(actual - expected)}")
    for group in summary:
        if any(value["samples"] != repeats for value in group["variants"].values()):
            raise ValueError(f"Summary repetition counts differ: {directory}")
    return {"configuration": config, "summary": summary, "samples": samples}


def work_columns(totals, latest, scope):
    totals, latest = totals or {}, latest or {}
    return {**{name: totals.get(name) for name in WORK_FIELDS},
            "search_counter_scope": scope,
            "last_search_complete": latest.get("complete"),
            "last_search_stop_reason": latest.get("stop_reason")}


def compiler_row(label, row):
    result = {key: row.get(key) for key in COMPILER_FIELDS}
    result["set"] = label
    verification = row.get("verification") or {}
    result.update(oracle_passed=verification.get("passed"),
                  allocation_passed=(verification.get("allocation") or {}).get("passed"))
    limits = row.get("discovery_limits") or {}
    result.update({column: limits.get(key) for column, key in (
        ("joint_limit", "ACT_JOINT_LIMIT"), ("raw_limit", "ACT_JOINT_RAW_LIMIT"),
        ("search_limit", "ACT_JOINT_SEARCH_LIMIT"), ("attempt_limit", "ACT_JOINT_ATTEMPT_LIMIT"))})
    direct = row.get("direct_final_report")
    if direct is not None:
        result.update(work_columns(row.get("direct_work_totals"), direct.get("search"), "lifetime"))
        result["direct_applications"] = direct.get("applications")
    else:
        reports = row.get("discovery_reports") or []
        latest = reports[-1]["search"] if reports else None
        result.update(work_columns(row.get("discovery_work_totals"), latest,
                                   "sum_of_refresh_snapshots" if reports else "unavailable"))
    return result


def micro_row(label, row):
    result = {key: row.get(key) for key in MICRO_FIELDS}
    result["set"] = label
    metrics = row.get("metrics") or {}
    for key in MICRO_FIELDS:
        if key in metrics:
            result[key] = metrics[key]
        elif key.endswith("_ms"):
            value = metrics.get(key[:-3] + "_ns")
            if key == "discovery_ms" and isinstance(value, dict):
                value = sum(value.values())
            result[key] = value / 1_000_000 if value is not None else None
    direct = metrics.get("direct_report")
    work = direct.get("search") if direct is not None else metrics.get("discovery_report")
    result.update(work_columns(work, work, "lifetime" if direct is not None
                               else "single_refresh_snapshot" if work is not None else "unavailable"))
    return result


def csv_bytes(rows, fields):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: (json.dumps(value, sort_keys=True, ensure_ascii=False) if isinstance(value, (dict, list))
                               else str(value).lower() if isinstance(value, bool) else value)
                         for key, value in row.items()})
    return output.getvalue().encode()


def export(root, output):
    measurements, checks, documents, failed_logs = {}, {}, {}, {}
    for family in ("probe", "independent", "micro"):
        relative = f"{family}/builds.json"
        documents[relative] = read_json(root / relative)
    compiler_rows, micro_rows = [], []
    for family, label in FINAL_SETS:
        relative = f"{family}/measurements/{label}"
        data = completed_set(root / relative, family, final=True)
        config = data["configuration"]
        limits = config["limits" if family == "micro" else "discovery_limits"]
        if limits != (TIGHT_LIMITS if label == "final-tight" else DEFAULT_LIMITS):
            raise ValueError(f"Unexpected limits in {relative}: {limits}")
        if label == "final-tight" and set(config["cases"]) != TIGHT_CASES:
            raise ValueError(f"final-tight must contain exactly {sorted(TIGHT_CASES)}")
        if family == "probe" and label == "final-default":
            all_cases = {case["name"] for case in documents["probe/builds.json"]["workloads"]}
            if len(all_cases) != 14 or set(config["cases"]) != all_cases:
                raise ValueError("probe final-default must include all 14 manifest workloads")
        if config["build_manifest_sha256"] != digest(root / family / "builds.json"):
            raise ValueError(f"Build manifest changed since {relative}")
        measurements[relative] = data
        target, convert = (micro_rows, micro_row) if family == "micro" else (compiler_rows, compiler_row)
        target.extend(convert(relative, row) for row in data["samples"])
    for family in ("probe", "independent", "micro"):
        directory = root / family / "checks"
        if directory.is_dir():
            for path in sorted(directory.iterdir()):
                if path.is_dir():
                    checks[str(path.relative_to(root))] = completed_set(path, family)
    regression_paths = (
        "reference/test-summary.json", "stage1/full-regression/summary.json", "stage2/full-regression/summary.json",
    )
    for relative in regression_paths:
        documents[relative] = read_json(root / relative)
        if not documents[relative] or any(not isinstance(row.get("passed"), bool) for row in documents[relative]):
            raise ValueError(f"Incomplete regression summary: {relative}")
        mode = "multi-result" if relative.startswith("reference/") else (
            "multipattern" if relative.startswith("stage1/") else "multipattern-rewrite")
        required = {"python", "baseline-qkv", f"{mode}-qkv", f"{mode}-double_gemm",
                    "verification-cli", "allocation-mutations", "dependent_tuple",
                    "dependent_tuple_transitive", "rust-QKV", "rust-QKVDoubleGemm"}
        recorded = [row["test"] for row in documents[relative]]
        if len(recorded) != len(set(recorded)) or not required.issubset(recorded):
            raise ValueError(f"Incomplete/duplicated regression groups: {relative}")
    for directory in (root / "reference", root / "stage1/full-regression", root / "stage2/full-regression"):
        for path in sorted(directory.glob("*.json")):
            documents[str(path.relative_to(root))] = read_json(path)
    for data in [*measurements.values(), *checks.values()]:
        for row in data["samples"]:
            if row["passed"]:
                continue
            records = [row, row.get("verification_process") or {}]
            for record in records:
                if record.get("log"):
                    path = Path(record["log"])
                    if not path.is_absolute():
                        path = root / path
                    relative = str(path.resolve().relative_to(root))
                    failed_logs[relative] = path.read_text()
    counts = {name: {"samples": len(data["samples"]),
                     "warmups": sum(row["warmup"] for row in data["samples"]),
                     "failed": sum(not row["passed"] for row in data["samples"])}
              for name, data in {**measurements, **checks}.items()}
    raw = {"schema_version": 1, "measurement_sets": measurements, "check_sets": checks,
           "supporting_documents": documents, "failed_sample_logs": failed_logs, "sample_counts": counts}
    aggregate = {"schema_version": 1, "sample_counts": counts,
                 "measurement_sets": {name: {key: value for key, value in data.items() if key != "samples"}
                                      for name, data in measurements.items()},
                 "check_sets": {name: {key: value for key, value in data.items() if key != "samples"}
                                for name, data in checks.items()},
                 "supporting_documents": documents}
    compressed = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=compressed, compresslevel=9, mtime=0) as stream:
        stream.write(json_bytes(raw))
    artifacts = {"raw-results.json.gz": compressed.getvalue(), "aggregate.json": json_bytes(aggregate),
                 "compiler-samples.csv": csv_bytes(compiler_rows, COMPILER_FIELDS),
                 "micro-samples.csv": csv_bytes(micro_rows, MICRO_FIELDS)}
    output.mkdir(parents=True, exist_ok=True)
    for name, content in artifacts.items():
        temporary = output / (name + ".tmp")
        temporary.write_bytes(content)
        temporary.replace(output / name)
    print(json.dumps({"output": str(output), "compiler_rows": len(compiler_rows),
                      "micro_rows": len(micro_rows), "sample_counts": counts}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("build/goal-multipattern"))
    parser.add_argument("--output", type=Path, default=Path("experiments/multipattern/results"))
    args = parser.parse_args()
    try:
        export(args.root.resolve(), args.output.resolve())
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Cannot export incomplete/inconsistent evidence: {error}\n")


if __name__ == "__main__":
    main()
