#!/usr/bin/env python3
"""Run generator regressions, supported cases, dependency checks and Rust tests."""
import argparse
import json
import os
import shutil
import sys

from .common import ROOT, command, main_guard, process_env, toolchain_lock, write_json
from prepare_sources import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit-only", action="store_true", help="Run Python generator tests only")
    args = parser.parse_args()
    result_dir = ROOT / "build/checks/tests"
    result_dir.mkdir(parents=True, exist_ok=True)
    records = []

    def check(label, argv, env):
        print(f"Testing {label}; log: {result_dir / (label + '.log')}", flush=True)
        try:
            command(argv, env=env, log=result_dir / f"{label}.log")
            records.append({"test": label, "passed": True})
        except RuntimeError as error:
            records.append({"test": label, "passed": False, "error": str(error)})
            raise
        finally:
            write_json(result_dir / "summary.json", records)

    with toolchain_lock("multi-result"):
        prepare("multi-result")
        env = process_env("multi-result")
        env["PYTHONPATH"] = str(ROOT / "scripts") + os.pathsep + env["PYTHONPATH"]
        check("python", [sys.executable, "-m", "unittest", "discover", "-s", ROOT / "tests",
                         "-p", "test_*.py", "-v"], env)
    if args.unit_only:
        return

    for case, toolchain in (("qkv", "baseline"), ("qkv", "multi-result"), ("double_gemm", "multi-result")):
        check(f"{toolchain}-{case}", [ROOT / "scripts/run.sh", case, "--toolchain", toolchain],
              process_env(toolchain))
    check("verification-cli", [sys.executable, ROOT / "tests/check_verification.py"], env)
    check("allocation-mutations", [sys.executable, ROOT / "tests/check_allocation.py"], env)
    for name in ("dependent_tuple", "dependent_tuple_transitive"):
        check(name, [ROOT / "scripts/run.sh", "double_gemm", "--skip-build", "--workload", ROOT / "tests/fixtures" / f"{name}.hlo"], env)
        report = json.loads((ROOT / "build/multi-result/runs/double_gemm" / name / "verification.json").read_text())
        if report["api_calls"].get("double_gemm", 0):
            records[-1].update(passed=False, error="Dependent GEMMs were combined")
            write_json(result_dir / "summary.json", records)
            raise ValueError(f"Dependent GEMMs were combined in {name}")

    # Compile regression harnesses in separate copies. Public generated targets
    # remain unchanged after running the Rust tests.
    suites = {"QKV": ("gte_frontend", "program_outputs", "recursive_tuples", "scalar_buffers"),
              "QKVDoubleGemm": ("iterative_discovery",)}
    with toolchain_lock("multi-result"):
        for target, suites_for_target in suites.items():
            source = ROOT / "build/multi-result/targets" / target / "backend"
            workspace = result_dir / "rust" / target
            if workspace.exists():
                shutil.rmtree(workspace)
            shutil.copytree(source, workspace, ignore=shutil.ignore_patterns("target", "build", "__pycache__"))
            (workspace / "tests").mkdir(exist_ok=True)
            shutil.copytree(ROOT / "tests/rust/common", workspace / "tests/common", dirs_exist_ok=True)
            for suite in suites_for_target:
                shutil.copy2(ROOT / "tests/rust" / f"{suite}.rs", workspace / "tests" / f"{suite}.rs")
            test_env = {**env, "ACT_TEST_REPO_ROOT": str(ROOT),
                        "ACT_TEST_BUILD": str(workspace / "artifacts"),
                        # Both generated crates are named backend, but their
                        # TensorOp enums and linked allocator code differ.
                        "CARGO_TARGET_DIR": str(result_dir / "cargo" / target)}
            # The fresh copy has no cpp/malloc/build. Cargo does not track
            # that external build output, so invalidate this crate's cached
            # build.rs result while retaining its Rust dependencies.
            command(["cargo", "clean", "--release", "--package", "backend", "--manifest-path", workspace / "Cargo.toml"],
                    env=test_env, log=result_dir / f"rust-{target}-clean.log")
            argv = ["cargo", "test", "--release", "--manifest-path", workspace / "Cargo.toml", "--lib", "--bin", "backend"]
            for suite in suites_for_target:
                argv.extend(("--test", suite))
            check(f"rust-{target}", [*argv, "--", "--test-threads=1"], test_env)
    print(f"PASS: {len(records)} test groups. Summary: {result_dir / 'summary.json'}")


if __name__ == "__main__":
    main_guard(main)
