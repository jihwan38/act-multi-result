"""Reject incomparable runs and summarize the measured RTL evidence."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics

from assembly import read_program
from hardware.run_record import record_path, sha, verify_run_record
from verify_rtl import OPCODES, inspect_trace, read_measurements


def artifact_path(recorded, path_map=None):
    """Locate moved artifacts without editing their recorded paths or hashes."""
    path = Path(recorded)
    if path_map is None:
        return path
    old, new = map(Path, path_map)
    if not old.is_absolute() or not new.is_absolute():
        raise ValueError("Path-map roots must be absolute")
    if any(".." in item.parts for item in (path, old, new)):
        raise ValueError("Path maps must not contain parent traversal")
    if path.is_relative_to(old):
        return new / path.relative_to(old)
    return path


def report_evidence(report, log):
    """Recheck the run record and raw cycles instead of trusting report fields."""
    run = verify_run_record(log)
    if report.get("run_record_sha256") != sha(record_path(log)):
        raise AssertionError("Report is not bound to this execution-time run record")
    for key in ("rtl_sha256", "binary_sha256", "stdout_sha256", "trace_sha256"):
        if report[key] != run[key]:
            raise AssertionError(f"Report {key} differs from the execution-time run record")
    provenance = report["provenance"]
    if any(str(report[key]) != str(provenance[key]) for key in ("workload", "mode", "dataset")):
        raise AssertionError("Report labels differ from the native provenance")
    measurements, _ = read_measurements(log.read_text(), provenance)
    if report["cycles"] != measurements:
        raise AssertionError("Report cycles differ from the original firmware output")
    return measurements, Path(str(log) + ".trace").read_text()


def service_windows(report, trace):
    commands, busy, _ = inspect_trace(trace)
    lanes = {lane: [command for command in commands if command[1] == opcode]
             for lane, opcode in OPCODES.items()}
    # Flush and three configuration commands per engine, once before the kernel.
    cursors = {0: 4, 1: 4}
    invocations = []
    for _ in range(report["provenance"]["repetitions"]):
        spans = []
        for name, _ in report["provenance"]["calls"]:
            if name == "double_gemm":
                engines = (0, 1)
            else:
                operation, suffix = name.rsplit("_", 1)
                lane = int(suffix)
                if operation != "gemm":
                    cursors[lane] += 3 if operation == "requantize_store" else 1
                    continue
                engines = (lane,)
            starts, ends = [], []
            for lane in engines:
                preload, compute = lanes[lane][cursors[lane]:cursors[lane] + 2]
                cursors[lane] += 2
                assert preload[2] == 6 and compute[2] == 4
                starts.append(preload[0])
                intervals = [(start, end) for start, end in busy[OPCODES[lane]]
                             if start <= compute[0] + 1 < end]
                if len(intervals) != 1:
                    raise AssertionError("Cannot bind compute acceptance to its real RTL busy interval")
                ends.append(intervals[0][1])
            # Same simulator clock, not a max of separately simulated engines.
            spans.append({"instruction": name, "begin": min(starts), "end": max(ends),
                          "cycles": max(ends) - min(starts)})
        invocations.append(spans)
    for lane in lanes:
        if cursors[lane] != len(lanes[lane]):
            raise AssertionError("Unaccounted native commands")
    return invocations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--path-map", nargs=2, type=Path, metavar=("OLD_ROOT", "NEW_ROOT"),
                        help="Explicitly relocate recorded artifact paths; evidence is not rewritten")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a fresh analysis output")
    suite = json.loads((args.suite / "summary.json").read_text())
    if suite["failures"] or suite["rtl_passes"] != suite["expected_rtl_runs"]:
        raise AssertionError("Incomplete suite: do not summarize it as success")
    report_paths = [artifact_path(path, args.path_map) for path in suite["reports"]]
    reports = [json.loads(path.read_text()) for path in report_paths]
    if len({report["rtl_sha256"] for report in reports}) != 1:
        raise AssertionError("Modes used different hardware binaries")
    seen, inputs, metadata = set(), {}, {}
    rows = []
    for path, report in zip(report_paths, reports):
        if not report["rtl_passed"]:
            raise AssertionError("A report failed its RTL audit")
        key = (report["workload"], report["dataset"], report["mode"])
        if key in seen:
            raise AssertionError("Duplicate trial")
        seen.add(key)
        shared = key[:2]
        provenance = report["provenance"]
        assembly = artifact_path(suite["compiled"][key[0]]["assembly"], args.path_map)
        if hashlib.sha256(assembly.read_bytes()).hexdigest() != provenance["assembly_sha256"]:
            raise AssertionError("Native run is not bound to the functionally checked ACT assembly")
        if key[2] == "act":
            program = read_program(assembly)
            if [list(call) for call in program.calls] != provenance["calls"]:
                raise AssertionError("ACT instructions were regrouped or replaced during native lowering")
            if program.metadata != provenance["metadata"]:
                raise AssertionError("ACT-assigned addresses or ABI changed during lowering")
        if inputs.setdefault(shared, provenance["input_sha256"]) != provenance["input_sha256"]:
            raise AssertionError("Modes used different input bytes")
        if metadata.setdefault(shared, provenance["metadata"]) != provenance["metadata"]:
            raise AssertionError("Modes used different logical memory layouts")
        log_path = Path(path).with_suffix("").with_suffix(".log")
        measurements, trace = report_evidence(report, log_path)
        windows = service_windows(report, trace)
        rows.append({"workload": key[0], "dataset": key[1], "mode": key[2],
                     "cycles": measurements, "double_gemm_count": report["double_gemm_count"],
                     "compute_service_cycles": [sum(span["cycles"] for span in spans) for spans in windows],
                     "compute_service_windows": windows,
                     "pe00_nonzero_overlap_cycles": report["pe00_nonzero_overlap_cycles"]})
    aggregates = []
    for case in suite["compiled"]:
        for mode in ("sequential", "manual_parallel", "act"):
            selected = [row for row in rows if row["workload"] == case and row["mode"] == mode]
            assert len(selected) == 6
            aggregates.append({"workload": case, "mode": mode,
                               "median_cycles": [statistics.median(row["cycles"][index] for row in selected)
                                                 for index in (0, 1)],
                               "range_cycles": [[min(row["cycles"][index] for row in selected),
                                                 max(row["cycles"][index] for row in selected)] for index in (0, 1)],
                               "double_gemm_count": selected[0]["double_gemm_count"]})
    result = {"rtl_sha256": reports[0]["rtl_sha256"], "verified_processes": len(reports),
              "verified_kernel_invocations": len(reports) * 2, "aggregates": aggregates, "trials": rows,
              "measurement": "First invocation and repeated-after-verification invocation, separate cycle values.",
              "compute_measurement": "Sum of per-GEMM command-acceptance-to-completion windows in the same RTL trace; excludes data transfers and gaps between macros. Not end-to-end latency."}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for row in aggregates:
        print(f"{row['workload']:16} {row['mode']:16} median={row['median_cycles']} range={row['range_cycles']} DoubleGEMM={row['double_gemm_count']}")


if __name__ == "__main__":
    main()
