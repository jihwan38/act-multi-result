#!/usr/bin/env python3
"""Measure one generated joint search/apply pass on controlled ACT e-graphs."""
import argparse
import json
import os
from pathlib import Path
import shutil
import statistics
import sys

from run import HERE, command, environment, sha, utc, verify_build, write_json


def sources(args):
    locations = {"shared": args.shared_build}
    if args.independent_build:
        locations["independent"] = args.independent_build
    records = {}
    for profile, location in locations.items():
        manifest = json.loads((location / "builds.json").read_text())
        modes = args.toolchains or list(manifest["builds"])
        for mode in modes:
            record = manifest["builds"][mode]
            verify_build(record)
            records[f"{profile}/{mode}"] = {**record, "profile": profile,
                                            "build_manifest": str(location / "builds.json")}
    for mode in args.toolchains or sorted({record["toolchain"] for record in records.values()}):
        fingerprints = {record["source_fingerprint"] for record in records.values() if record["toolchain"] == mode}
        if len(fingerprints) != 1:
            raise ValueError(f"Profile source plans differ for {mode}")
    return records


def build(args):
    records = sources(args)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "builds.json"
    manifest = json.loads(path.read_text()) if path.exists() else {"created_utc": utc(), "builds": {}}
    harness = HERE / "discovery_probe.rs"
    for key, record in records.items():
        if key in manifest["builds"]:
            prior = manifest["builds"][key]
            if prior["harness_sha256"] != sha(harness) or prior["source_fingerprint"] != record["source_fingerprint"]:
                raise ValueError(f"Changed microbenchmark {key}; choose a fresh --output")
            if sha(prior["binary"]) != prior["binary_sha256"]:
                raise ValueError(f"Changed microbenchmark executable: {key}")
            print(f"Reusing microbenchmark {key}", flush=True)
            continue
        base = Path(record["base"])
        target_name = record.get("target", "MultiPatternProbe")
        generated = base / "targets" / target_name / "backend"
        crate = args.output / "crates" / key
        shutil.copytree(generated, crate, ignore=shutil.ignore_patterns("target", "build", "__pycache__", "*.pyc"))
        examples = crate / "examples"
        examples.mkdir(exist_ok=True)
        shutil.copy2(harness, examples / "discovery_probe.rs")
        env = environment(record)
        # Reuse dependencies while compiling this isolated crate's library/example.
        env["CARGO_TARGET_DIR"] = str(generated / "target")
        direct = (crate / "src/isel/rewrites/multi.rs").exists()
        reports = "pub fn report(" in (crate / "src/isel/discovery.rs").read_text()
        flags = ["--cfg", "act_direct_rules"] if direct else (["--cfg", "act_discovery_report"] if reports else [])
        if flags:
            env["RUSTFLAGS"] = " ".join(flags)
        result = command(["cargo", "build", "--offline", "--release", "--locked", "--manifest-path",
                          crate / "Cargo.toml", "--example", "discovery_probe"], env,
                         args.output / "generation" / key, args.timeout)
        if result["exit_status"] != 0 or result["timed_out"]:
            raise RuntimeError(f"Microbenchmark build failed for {key}: {result['log']}")
        binary = args.output / "bin" / key
        binary.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(generated / "target/release/examples/discovery_probe", binary)
        manifest["builds"][key] = {**record, "crate": str(crate), "binary": str(binary),
                                   "binary_sha256": sha(binary), "harness_sha256": sha(harness),
                                   "cfg": flags, "direct_rules": direct, "has_search_report": reports,
                                   "generation": result}
        write_json(path, manifest)
        print(f"Built microbenchmark {key}", flush=True)


def scenarios(profile):
    if profile == "independent":
        return [("independent", size) for size in (8, 32, 128)]
    return ([(scenario, size) for scenario in ("shared", "disjoint") for size in (8, 32, 128)]
            + [("dependent", 8), ("wrong-shape", 32), ("metadata", 8)])


def run(args, measured):
    manifest = json.loads((args.output / "builds.json").read_text())
    records = {key: record for key, record in manifest["builds"].items()
               if not args.toolchains or record["toolchain"] in args.toolchains}
    for key, record in records.items():
        verify_build(record)
        if sha(record["binary"]) != record["binary_sha256"]:
            raise ValueError(f"Changed executable: {key}")
    output = args.output / ("measurements" if measured else "checks") / args.label
    output.mkdir(parents=True, exist_ok=False)
    limits = {"ACT_JOINT_LIMIT": args.joint_limit, "ACT_JOINT_RAW_LIMIT": args.raw_limit,
              "ACT_JOINT_SEARCH_LIMIT": args.search_limit, "ACT_JOINT_ATTEMPT_LIMIT": args.attempt_limit}
    config = {"started_utc": utc(), "repeats": args.repeats if measured else 1,
              "warmups": args.warmups if measured else 0, "limits": limits,
              "timeout_seconds": args.timeout, "build_manifest_sha256": sha(args.output / "builds.json"),
              "runner_sha256": sha(Path(__file__)), "ordering": "profile/scenario-major; rotate modes per repetition",
              "scope": "Graph construction excluded from internal timings; full process wall/RSS includes construction/runtime",
              "comparison": "Main/stage1 discovery plus generated joint rule search/apply; direct mode generated joint rules only",
              "quota_note": "Main uses its existing tuple-cap-derived attempt budget; bounded variants also honor explicit raw/call/attempt limits",
              "report_note": "Raw matches include replay; replayed_matches recorded separately; completeness concerns this search snapshot"}
    write_json(output / "configuration.json", config)
    samples = []
    for profile in sorted({record["profile"] for record in records.values()}):
        for scenario, size in scenarios(profile):
            rounds = range(-args.warmups, args.repeats) if measured else range(1)
            for repetition in rounds:
                keys = [key for key, record in records.items() if record["profile"] == profile]
                offset = repetition % len(keys)
                keys = keys[offset:] + keys[:offset]
                for order, key in enumerate(keys):
                    record = records[key]
                    env = environment(record)
                    env.update({name: str(value) for name, value in limits.items()})
                    directory = output / profile / f"{scenario}-{size}" / f"{record['toolchain']}-{repetition}"
                    result = command([record["binary"], scenario, str(size)], env, directory, args.timeout, measure=True)
                    result.update(profile=profile, scenario=scenario, size=size, toolchain=record["toolchain"],
                                  repetition=repetition, warmup=repetition < 0, order=order,
                                  binary_sha256=record["binary_sha256"], source_fingerprint=record["source_fingerprint"])
                    parsed = None
                    for line in reversed((directory / "stdout.log").read_text().splitlines()):
                        try:
                            data = json.loads(line)
                        except ValueError:
                            continue
                        if isinstance(data, dict) and "semantic_assertions_passed" in data:
                            parsed = data
                            break
                    result["metrics"] = parsed
                    result["passed"] = (result["exit_status"] == 0 and not result["timed_out"]
                                        and parsed is not None and parsed["semantic_assertions_passed"])
                    write_json(directory / "result.json", result)
                    samples.append(result)
                    write_json(output / "samples.json", samples)
                    print(f"{key} {scenario}-{size} repetition={repetition}: passed={result['passed']}", flush=True)
    summaries = []
    for profile in sorted({record["profile"] for record in records.values()}):
        for scenario, size in scenarios(profile):
            summary = {"profile": profile, "scenario": scenario, "size": size, "variants": {}}
            for mode in sorted({record["toolchain"] for record in records.values() if record["profile"] == profile}):
                rows = [row for row in samples if row["profile"] == profile and row["scenario"] == scenario
                        and row["size"] == size and row["toolchain"] == mode and not row["warmup"]]
                fields = {}
                for field in ("joint_candidate_pass_ns", "rule_search_ns", "rule_apply_rebuild_ns",
                              "registered_tuple_count", "implemented_tuple_count"):
                    values = [row["metrics"][field] for row in rows if row["metrics"] is not None]
                    fields[field] = {"count": len(values), "median": statistics.median(values),
                                     "min": min(values), "max": max(values)} if values else None
                for field in ("wall_seconds", "gnu_time_max_rss_kib"):
                    values = [row[field] for row in rows if field in row]
                    fields[field] = {"count": len(values), "median": statistics.median(values),
                                     "min": min(values), "max": max(values)} if values else None
                summary["variants"][mode] = {"samples": len(rows), "passed": sum(row["passed"] for row in rows), "metrics": fields}
            summaries.append(summary)
    write_json(output / "summary.json", summaries)
    config["finished_utc"] = utc()
    write_json(output / "configuration.json", config)
    if not all(row["passed"] for row in samples):
        raise RuntimeError(f"Some microbenchmarks failed; evidence retained at {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("build", "check", "measure"):
        subparser = subparsers.add_parser(action)
        subparser.add_argument("--output", type=Path, required=True)
        subparser.add_argument("--toolchains", nargs="+")
        subparser.add_argument("--timeout", type=float, default=240 if action == "build" else 15)
        if action == "build":
            subparser.add_argument("--shared-build", type=Path, required=True)
            subparser.add_argument("--independent-build", type=Path)
        else:
            subparser.add_argument("--label", required=True)
            subparser.add_argument("--joint-limit", type=int, default=64)
            subparser.add_argument("--raw-limit", type=int, default=4096)
            subparser.add_argument("--search-limit", type=int, default=4096)
            subparser.add_argument("--attempt-limit", type=int, default=4096)
        if action == "measure":
            subparser.add_argument("--repeats", type=int, default=5)
            subparser.add_argument("--warmups", type=int, default=1)
    args = parser.parse_args()
    for name in ("output", "shared_build", "independent_build"):
        if getattr(args, name, None):
            setattr(args, name, getattr(args, name).resolve())
    if args.action == "measure" and (args.repeats < 5 or args.warmups < 1):
        parser.error("Measurements require at least five repetitions and one warmup")
    if args.action == "build":
        build(args)
    else:
        run(args, measured=args.action == "measure")


if __name__ == "__main__":
    main()
