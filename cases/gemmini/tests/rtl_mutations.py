"""Real RTL negative tests: corrupt emitted native code, never the golden data."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="A passing two_gemm ACT native source")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    experiment = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if not output.is_relative_to(experiment.parents[1] / "build/multi-result-legacy") or output.exists():
        raise ValueError("Use a fresh mutation directory under build/multi-result-legacy")
    output.mkdir(parents=True)
    source = args.source.read_text()
    header = (experiment / "native/gemmini_dual.h").read_text()
    if source.count("act_mvout(1,") != 1 or source.count("act_double_gemm(") != 1:
        raise ValueError("Expected the checked two-GEMM joint assembly")
    mutations = {
        "duplicate_lane": (source.replace("act_mvout(1,", "act_mvout(0,"), header),
        "missing_completion": (
            source, header.replace('asm volatile("fence" ::: "memory");',
                                   'asm volatile("" ::: "memory");')),
        "unwanted_accumulation": (source, header.replace('act_tile(output | (1U << 31))',
                                                         'act_tile(output | (1U << 31) | (1U << 30))')),
    }
    results = []
    for name, (mutated_source, mutated_header) in mutations.items():
        assert mutated_source != source or mutated_header != header
        directory = output / name
        directory.mkdir()
        header_path = directory / "mutated.h"
        header_path.write_text(mutated_header)
        # Quoted includes resolve this private header; the official ISA headers
        # and all simulator/source worktrees remain unchanged.
        mutated_source = mutated_source.replace('#include "gemmini_dual.h"', '#include "mutated.h"')
        source_path = directory / "test.c"
        source_path.write_text(mutated_source)
        binary = directory / "test"
        with (directory / "build.log").open("w") as log:
            subprocess.run(["bash", str(experiment / "native/build.sh"), str(source_path), str(binary)],
                           check=True, stdout=log, stderr=subprocess.STDOUT)
        log_path = directory / "rtl.log"
        with (directory / "process.log").open("w") as log:
            process = subprocess.run(["bash", str(experiment / "hardware/run_native.sh"), str(binary), str(log_path)],
                                     stdout=log, stderr=subprocess.STDOUT)
        text = log_path.read_text()
        detected = process.returncode != 0 and "ACT_FAIL" in text and "ACT_PASS" not in text
        result = {"mutation": name, "detected": detected, "exit_status": process.returncode,
                  "source_sha256": hashlib.sha256(mutated_source.encode()).hexdigest(),
                  "mutated_header_sha256": hashlib.sha256(mutated_header.encode()).hexdigest(),
                  "log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
                  "failure_lines": [line for line in text.splitlines() if "ACT_FAIL" in line]}
        results.append(result)
        print(result, flush=True)
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    if not all(result["detected"] for result in results):
        raise SystemExit("A mutation escaped numerical detection; inspect the logs")


if __name__ == "__main__":
    main()
