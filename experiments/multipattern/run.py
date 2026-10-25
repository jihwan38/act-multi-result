#!/usr/bin/env python3
"""Build, check, and measure the generated MultiPatternProbe ISA separately.

Run inside the ACT Conda environment. Every compiler run has a fresh directory
and an external timeout; Oracle validation is excluded from compiler timings.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from _internal.common import (COMPONENTS, TOOLCHAINS, fingerprint, process_env,
                              source_hashes, source_plan, toolchain_lock, write_json)
from prepare_sources import prepare
from workloads import cases

TARGET = "MultiPatternProbe"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def git(*arguments):
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def command(argv, env, directory, timeout, measure=False):
    """Capture failure evidence and terminate the whole process group on timeout."""
    directory.mkdir(parents=True, exist_ok=False)
    original = [str(value) for value in argv]
    argv = original
    if measure:
        argv = ["/usr/bin/time", "-f", "wall_seconds=%e\nuser_seconds=%U\nsystem_seconds=%S\nmax_rss_kib=%M\nexit_status=%x",
                "-o", str(directory / "time.txt"), *argv]
    start = time.perf_counter_ns()
    timed_out = False
    child = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             start_new_session=True)
    try:
        captured, _ = child.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        captured, _ = child.communicate()
    elapsed = (time.perf_counter_ns() - start) / 1e9
    status = child.returncode
    (directory / "stdout.log").write_bytes(captured)
    result = {"command": original, "exit_status": status, "timed_out": timed_out,
              "wall_seconds": elapsed,
              "timeout_seconds": timeout, "log": str(directory / "stdout.log")}
    if (directory / "time.txt").is_file():
        for line in (directory / "time.txt").read_text().splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            try:
                result["gnu_time_" + key] = float(value) if "." in value else int(value)
            except ValueError:
                result["gnu_time_" + key] = value
    write_json(directory / "process.json", result)
    return result


def runtime_files(base, target_name):
    target = base / "targets" / target_name
    paths = [base / "backends" / target_name,
             target / "backend/python/cost/model.py",
             target / "backend/src/isel/rewrites/ir2ir_rewrites.txt",
             target / "backend/src/isel/rewrites/ir2isa_rewrites.txt",
             target / "oracle/api.py", target / "oracle/decorator.py"]
    paths.extend(sorted((target / "oracle/build").glob("*.so")))
    return {str(path): sha(path) for path in paths}


def verify_build(record):
    changed = [path for path, digest in record["runtime_sha256"].items()
               if not Path(path).is_file() or sha(path) != digest]
    if changed:
        raise ValueError(f"Build runtime changed since generation: {changed}")
    if source_hashes(Path(record["base"]) / "src") != record["prepared_source_sha256"]:
        raise ValueError(f"Frozen source changed: {record['base']}")
    generated = Path(record["base"]) / "targets" / record.get("target", TARGET)
    if source_hashes(generated) != record["generated_source_sha256"]:
        raise ValueError(f"Generated source changed: {generated}")


def environment(record):
    env = process_env(record["toolchain"])
    base = Path(record["base"])
    env.update(PYTHONPATH=os.pathsep.join(str(base / "src" / name) for name in COMPONENTS),
               ACT_BUILD_DIR=str(base), ACT_TOOLCHAIN=record["toolchain"])
    for key in ("ACT_JOINT_LIMIT", "ACT_JOINT_RAW_LIMIT", "ACT_JOINT_SEARCH_LIMIT", "ACT_JOINT_ATTEMPT_LIMIT",
                "ACT_PII_TIME_MS", "ACT_PII_VISITS", "RUSTFLAGS"):
        env.pop(key, None)
    return env


def build(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "builds.json"
    fixture_file = "IndependentISA.py" if args.fixture == "independent" else "ProbeISA.py"
    target_name = "IndependentPatternProbe" if args.fixture == "independent" else TARGET
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "created_utc": utc(), "repository_commit": git("rev-parse", "HEAD"),
        "main_commit": git("rev-parse", "main"), "builds": {},
        "fixture_file": fixture_file, "fixture_sha256": sha(HERE / fixture_file),
        "support_fixture_sha256": {"ProbeISA.py": sha(HERE / "ProbeISA.py")},
        "workloads_sha256": sha(HERE / "workloads.py"),
    }
    if (manifest.get("fixture_file", "ProbeISA.py") != fixture_file
            or manifest["fixture_sha256"] != sha(HERE / fixture_file)
            or manifest["workloads_sha256"] != sha(HERE / "workloads.py")
            or any(sha(HERE / name) != digest for name, digest in manifest.get("support_fixture_sha256", {}).items())):
        raise ValueError("Fixtures changed; choose a fresh --output directory")
    inputs = output / "inputs"
    inputs.mkdir(exist_ok=True)
    for name in {"ProbeISA.py", "workloads.py", fixture_file}:
        target = inputs / name
        if not target.exists():
            shutil.copy2(HERE / name, target)
    workload_records = []
    for case in cases():
        path = inputs / f"{case.name}.hlo"
        if path.exists() and path.read_text() != case.hlo:
            raise ValueError(f"Refuse changed workload: {path}")
        if not path.exists():
            path.write_text(case.hlo)
        required = list(case.require_calls)
        if args.fixture == "independent":
            required = [name.replace("shared_", "independent_") for name in required if name != "reshape_pair"]
        workload_records.append({"name": case.name, "hlo": str(path), "hlo_sha256": sha(path),
                                 "require_calls": required, "purpose": case.purpose})
    manifest["workloads"] = workload_records
    diff_path = output / "repository.diff"
    if not diff_path.exists():
        diff_path.write_text(git("diff", "--binary", "HEAD") + "\n")
    for mode in args.toolchains:
        with toolchain_lock(mode):
            provenance = prepare(mode)
            if mode in manifest["builds"]:
                record = manifest["builds"][mode]
                if record["source_fingerprint"] != provenance["fingerprint"]:
                    raise ValueError(f"{mode} source plan changed; choose a fresh --output directory")
                verify_build(record)
                print(f"Reusing verified build: {mode}", flush=True)
                continue
            base = output / "builds" / mode
            base.mkdir(parents=True, exist_ok=False)
            source = ROOT / "build" / mode / "src"
            shutil.copytree(source, base / "src", ignore=shutil.ignore_patterns("build", "target", "__pycache__", "*.pyc"))
            patch_root = base / "patches"
            for patch in provenance["patches"]:
                source_patch = ROOT / patch["path"]
                destination = patch_root / Path(patch["path"]).relative_to("patches")
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_patch, destination)
            record = {"toolchain": mode, "base": str(base), "source_fingerprint": provenance["fingerprint"],
                      "target": target_name, "fixture_file": fixture_file,
                      "source_plan": source_plan(mode), "prepared_source_sha256": source_hashes(base / "src"),
                      "repository_commit": git("rev-parse", "HEAD"), "started_utc": utc()}
        env = environment(record)
        result = command([sys.executable, "-B", inputs / fixture_file], env, base / "generation", args.timeout)
        record["generation"] = result
        write_json(base / "generation-result.json", record)
        if result["exit_status"] != 0 or result["timed_out"]:
            raise RuntimeError(f"{mode} generation failed; see {result['log']}")
        # Oracle generation creates an ignored native build under src; source_hashes excludes it.
        record["runtime_sha256"] = runtime_files(base, target_name)
        record["generated_source_sha256"] = source_hashes(base / "targets" / target_name)
        record["finished_utc"] = utc()
        manifest["builds"][mode] = record
        write_json(manifest_path, manifest)
        print(f"Built {mode}: {base / 'backends' / target_name}", flush=True)


def load(args):
    manifest = json.loads((args.output / "builds.json").read_text())
    for name, key in ((manifest.get("fixture_file", "ProbeISA.py"), "fixture_sha256"), ("workloads.py", "workloads_sha256")):
        if sha(args.output / "inputs" / name) != manifest[key]:
            raise ValueError(f"Frozen fixture changed: {name}")
    for name, digest in manifest.get("support_fixture_sha256", {}).items():
        if sha(args.output / "inputs" / name) != digest:
            raise ValueError(f"Frozen supporting fixture changed: {name}")
    selected = args.toolchains or list(manifest["builds"])
    records = {mode: manifest["builds"][mode] for mode in selected}
    for record in records.values():
        verify_build(record)
    workloads = [case for case in manifest["workloads"] if not args.cases or case["name"] in args.cases]
    if args.cases and set(args.cases) != {case["name"] for case in workloads}:
        raise ValueError("Unknown workload name")
    for case in workloads:
        if sha(case["hlo"]) != case["hlo_sha256"]:
            raise ValueError(f"Workload changed: {case['hlo']}")
    return manifest, records, workloads


def compile_case(record, case, directory, timeout, limits):
    verify_build(record)
    base = Path(record["base"])
    target_name = record.get("target", TARGET)
    env = environment(record)
    env.update({key: str(value) for key, value in limits.items()})
    directory.mkdir(parents=True, exist_ok=False)
    assembly = directory / "assembly.py"
    result = command([base / "backends" / target_name, "--input", case["hlo"], "--output", assembly,
                      "--log", directory / "compiler"], env, directory / "compile", timeout, measure=True)
    result.update(toolchain=record["toolchain"], case=case["name"], discovery_limits=limits,
                  assembly=str(assembly), binary_sha256=sha(base / "backends" / target_name),
                  source_fingerprint=record["source_fingerprint"], hlo_sha256=case["hlo_sha256"])
    result["effective_act_environment"] = {key: value for key, value in env.items() if key.startswith("ACT_")}
    reports = []
    for line in (directory / "compile/stdout.log").read_text().splitlines():
        if "ACT_DISCOVERY " in line:
            reports.append(json.loads(line.split("ACT_DISCOVERY ", 1)[1]))
    result["discovery_reports"] = reports
    result["discovery_work_totals"] = {
        key: sum(report["search"].get(key, 0) for report in reports)
        for key in ("search_calls", "raw_matches", "replayed_matches", "attempts", "accepted", "registered")
    } if reports else None
    result["candidate_pii_count"] = len(list((directory / "compiler").glob("*.pii")))
    result["candidate_assembly_count"] = len(list((directory / "compiler").glob("*.py")))
    if assembly.is_file():
        text = assembly.read_text()
        calls = Counter(re.findall(r"^\s+api\.([A-Za-z0-9_]+)\(", text, re.M))
        result.update(assembly_sha256=sha(assembly), api_calls=dict(calls), api_call_count=sum(calls.values()),
                      model_cost=len(text.splitlines()), model_cost_definition="Current ACT model: assembly line count; not hardware latency")
        match = re.search(r"^# PII number: (\d+)$", text, re.M)
        result["selected_pii"] = int(match[1]) if match else None
        result["required_calls_present"] = all(calls[name] > 0 for name in case["require_calls"])
    else:
        result["required_calls_present"] = False
    result["runtime_unchanged"] = all(sha(path) == digest for path, digest in record["runtime_sha256"].items())
    write_json(directory / "result.json", result)
    return result


def validate(record, case, result, directory, timeout):
    base = Path(record["base"])
    target_name = record.get("target", TARGET)
    report = directory / "verification.json"
    process = command([sys.executable, "-B", ROOT / "scripts/verify.py", "--hlo", case["hlo"],
                       "--assembly", result["assembly"], "--oracle", base / "targets" / target_name / "oracle",
                       "--isa", Path(case["hlo"]).parent / record.get("fixture_file", "ProbeISA.py"), "--report", report],
                      environment(record), directory / "verification", timeout)
    result["verification_process"] = process
    result["correct"] = process["exit_status"] == 0 and report.is_file()
    if report.is_file():
        data = json.loads(report.read_text())
        result["correct"] = result["correct"] and data.get("passed", False)
        result["verification"] = data
    write_json(directory / "result.json", result)


def run(args, measured):
    manifest, records, workloads = load(args)
    destination = args.output / ("measurements" if measured else "checks") / args.label
    destination.mkdir(parents=True, exist_ok=False)
    configuration = {"started_utc": utc(), "toolchains": list(records), "cases": [case["name"] for case in workloads],
                     "repeats": args.repeats if measured else 1, "warmups": args.warmups if measured else 0,
                     "timeout_seconds": args.timeout,
                     "allow_fallback": args.allow_fallback,
                     "host": platform.platform(), "python": sys.executable,
                     "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                     "cpu_model": next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                                        if line.startswith('model name')), "unknown"),
                     "ordering": "case-major; rotate toolchain order each repetition",
                     "wall_definition": "perf_counter_ns around GNU time/compiler process, excludes Oracle and build",
                     "rss_definition": "GNU time %M compiler process maximum RSS in KiB",
                     "input_datasets": "verify.py seed0, seed1, seed2, zero; identical across variants",
                     "build_manifest_sha256": sha(args.output / "builds.json"),
                     "runner_sha256": sha(Path(__file__))}
    limits = {"ACT_JOINT_LIMIT": args.joint_limit, "ACT_JOINT_RAW_LIMIT": args.raw_limit,
              "ACT_JOINT_SEARCH_LIMIT": args.search_limit, "ACT_JOINT_ATTEMPT_LIMIT": args.attempt_limit}
    configuration["discovery_limits"] = limits
    write_json(destination / "configuration.json", configuration)
    samples = []
    rounds = range(-args.warmups, args.repeats) if measured else range(1)
    for case in workloads:
        for repetition in rounds:
            modes = list(records)
            offset = repetition % len(modes)
            modes = modes[offset:] + modes[:offset]
            for order, mode in enumerate(modes):
                directory = destination / case["name"] / f"{mode}-{repetition}"
                result = compile_case(records[mode], case, directory, args.timeout, limits)
                result.update(repetition=repetition, warmup=repetition < 0, order=order)
                if result["exit_status"] == 0 and not result["timed_out"] and Path(result["assembly"]).is_file():
                    validate(records[mode], case, result, directory, args.verify_timeout)
                else:
                    result["correct"] = False
                result["correctness_pass"] = bool(result["correct"] and result["runtime_unchanged"]
                                                  and result["exit_status"] == 0 and not result["timed_out"])
                result["selection_expectations_pass"] = result["required_calls_present"]
                result["quality"] = ("joint_selection_satisfied" if result["required_calls_present"]
                                     else "fallback_for_expected_joint")
                result["passed"] = bool(result["correctness_pass"] and (args.allow_fallback or result["required_calls_present"]))
                write_json(directory / "result.json", result)
                samples.append(result)
                write_json(destination / "samples.json", samples)
                print(f"{case['name']} {mode} repetition={repetition}: passed={result['passed']} wall={result['wall_seconds']:.3f}s calls={result.get('api_calls', {})}", flush=True)
    summaries = []
    for case in workloads:
        summary = {"case": case["name"], "variants": {}}
        for mode in records:
            rows = [sample for sample in samples if sample["case"] == case["name"] and sample["toolchain"] == mode and not sample["warmup"]]
            metrics = {}
            for name in ("wall_seconds", "gnu_time_max_rss_kib", "model_cost", "api_call_count", "candidate_pii_count"):
                values = [row[name] for row in rows if name in row]
                metrics[name] = {"count": len(values), "median": statistics.median(values), "min": min(values), "max": max(values)} if values else None
            summary["variants"][mode] = {"samples": len(rows), "passed": sum(row["passed"] for row in rows),
                                           "correctness_passed": sum(row["correctness_pass"] for row in rows),
                                           "selection_expectations_passed": sum(row["selection_expectations_pass"] for row in rows),
                                           "failures": [index for index, row in enumerate(rows) if not row["passed"]], "metrics": metrics}
        summaries.append(summary)
    write_json(destination / "summary.json", summaries)
    configuration["finished_utc"] = utc()
    write_json(destination / "configuration.json", configuration)
    if not all(sample["passed"] for sample in samples):
        raise RuntimeError(f"Some runs failed; complete evidence retained in {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("build", "check", "measure"):
        subparser = subparsers.add_parser(action)
        subparser.add_argument("--output", type=Path, required=True)
        subparser.add_argument("--toolchains", nargs="+", choices=TOOLCHAINS,
                               default=["multi-result", "multipattern"] if action == "build" else None)
        subparser.add_argument("--timeout", type=float, default=360 if action == "build" else 35)
        if action == "build":
            subparser.add_argument("--fixture", choices=("shared", "independent"), default="shared")
        if action != "build":
            subparser.add_argument("--label", required=True, help="Fresh label; existing run directories are refused")
            subparser.add_argument("--cases", nargs="+")
            subparser.add_argument("--joint-limit", type=int, default=64)
            subparser.add_argument("--raw-limit", type=int, default=4096)
            subparser.add_argument("--search-limit", type=int, default=4096)
            subparser.add_argument("--attempt-limit", type=int, default=4096)
            subparser.add_argument("--verify-timeout", type=float, default=60)
            subparser.add_argument("--allow-fallback", action="store_true", default=action == "measure",
                                   help="Record missed expected joint calls as a quality result; still require numerical/allocation correctness")
        if action == "measure":
            subparser.add_argument("--repeats", type=int, default=5)
            subparser.add_argument("--warmups", type=int, default=1)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.action == "measure" and (args.repeats < 5 or args.warmups < 1):
        parser.error("Measurements require at least five repetitions and one warmup")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.action == "build":
        build(args)
    else:
        run(args, measured=args.action == "measure")


if __name__ == "__main__":
    main()
