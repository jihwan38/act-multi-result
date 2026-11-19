"""Reproduce functional and dual-engine RTL cases on one fixed hardware binary.

Run with multi-result-legacy selected, after hardware/build.sh and generate.py.
Each native mode/dataset gets a fresh process/reset and two checked invocations.
Failures are recorded and the command fails; they are never averaged into passes.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


EXPERIMENT = Path(__file__).resolve().parent
BUILD = EXPERIMENT.parents[1] / "build/multi-result-legacy"
sys.path.insert(0, str(EXPERIMENT.parents[1] / "scripts"))
from _internal.common import fingerprint, source_plan, toolchain_lock

CASES = ("two_gemm", "shared_input", "dependent_gemm", "four_gemm")
MODES = ("sequential", "manual_parallel", "act")
DATASETS = ("901", "902", "903", "904", "identity", "zero")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(args, log):
    with log.open("w") as stream:
        subprocess.run([str(arg) for arg in args], check=True, stdout=stream,
                       stderr=subprocess.STDOUT, cwd=EXPERIMENT)


def native_job(case, dataset, mode, assembly, directory):
    name = f"{case}_{dataset}_{mode}"
    binary = directory / name
    execute([sys.executable, EXPERIMENT / "emit_native.py", "--assembly", assembly,
             "--mode", mode, "--dataset", dataset, "--output", binary.with_suffix(".c")],
            directory / f"{name}.emit.log")
    execute(["bash", EXPERIMENT / "native/build.sh", binary.with_suffix(".c"), binary],
            directory / f"{name}.build.log")
    log = directory / f"{name}.log"
    execute(["bash", EXPERIMENT / "hardware/run_native.sh", binary, log],
            directory / f"{name}.process.log")
    report = directory / f"{name}.rtl.json"
    execute([sys.executable, EXPERIMENT / "verify_rtl.py", "--binary", binary,
             "--log", log, "--report", report], directory / f"{name}.audit.log")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=BUILD)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3, choices=range(1, 5))
    args = parser.parse_args()
    target, output = args.target.resolve(), args.output.resolve()
    if target != BUILD or os.environ.get("ACT_TOOLCHAIN") != "multi-result-legacy":
        raise ValueError("Select multi-result-legacy with source scripts/env.sh multi-result-legacy")
    if not output.is_relative_to(BUILD) or output.exists():
        raise ValueError("Use a fresh directory under build/multi-result-legacy")
    generation = json.loads((target / "generation.json").read_text())
    if generation["source_fingerprint"] != fingerprint(source_plan("multi-result-legacy")):
        raise ValueError("ACT sources changed: prepare sources and regenerate the target")
    if generation["spec_sha256"] != sha(EXPERIMENT / "GemminiDual.py"):
        raise ValueError("Regenerate the target: accelerator specification changed")
    for filename, digest in generation["workloads"].items():
        if sha(EXPERIMENT / "workloads" / filename) != digest:
            raise ValueError(f"Workload differs from generation manifest: {filename}")
    backend = target / "backends/GemminiDual"
    if sha(backend) != generation["backend_binary_sha256"]:
        raise ValueError("Backend binary provenance differs")
    output.mkdir(parents=True)
    native = output / "native"
    native.mkdir()
    failures, compiled, reports = [], {}, []
    for case in CASES:
        directory = output / case
        directory.mkdir()
        (directory / "compiler").mkdir()
        assembly = directory / "assembly.py"
        start = time.monotonic()
        try:
            execute([backend, "--input", EXPERIMENT / "workloads" / f"{case}.hlo",
                     "--output", assembly, "--log", directory / "compiler"], directory / "compiler.log")
            if not assembly.is_file():
                raise RuntimeError("ACT did not produce assembly")
            elapsed = time.monotonic() - start
            execute([sys.executable, EXPERIMENT / "verify.py", "--target", target,
                     "--hlo", EXPERIMENT / "workloads" / f"{case}.hlo", "--assembly", assembly,
                     "--report", directory / "functional.json"], directory / "functional.log")
            compiled[case] = {"assembly": str(assembly), "compiler_wall_seconds": elapsed}
            print(f"FUNCTIONAL PASS {case}", flush=True)
        except (subprocess.CalledProcessError, RuntimeError) as error:
            failures.append({"stage": "ACT", "case": case, "error": str(error)})
            print(f"FUNCTIONAL FAIL {case}: {error}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(native_job, case, dataset, mode, Path(item["assembly"]), native):
                   (case, dataset, mode)
                   for case, item in compiled.items() for dataset in DATASETS for mode in MODES}
        for future in as_completed(pending):
            case, dataset, mode = pending[future]
            try:
                report = future.result()
                reports.append(str(report))
                data = json.loads(report.read_text())
                print(f"RTL PASS {case}/{dataset}/{mode}: {data['cycles']} cycles, "
                      f"MAC witness overlap={data['pe00_nonzero_overlap_cycles']}", flush=True)
            except Exception as error:
                failures.append({"stage": "RTL", "case": case, "dataset": dataset,
                                 "mode": mode, "error": str(error)})
                print(f"RTL FAIL {case}/{dataset}/{mode}: {error}", flush=True)
    summary = {"generation": generation, "compiled": compiled, "reports": sorted(reports),
               "failures": failures, "rtl_passes": len(reports),
               "expected_rtl_runs": len(CASES) * len(DATASETS) * len(MODES),
               "repetitions_per_process": 2,
               "environment_overrides": {key: value for key, value in os.environ.items()
                                         if key.startswith("ACT_")},
               "note": "Compiler wall time is not accelerator performance. Each RTL report contains measured cycles."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if failures or len(reports) != summary["expected_rtl_runs"]:
        raise SystemExit(1)


if __name__ == "__main__":
    with toolchain_lock("multi-result-legacy"):
        main()
