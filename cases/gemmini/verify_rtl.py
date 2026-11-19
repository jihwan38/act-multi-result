"""Audit RTL command acceptance, numerical completion and real PE overlap."""

import argparse
import hashlib
import json
from pathlib import Path
import re

from hardware.run_record import record_path, verify_run_record


ACTIVITY = re.compile(r"ACT_GEMMINI opcode=\s*(\d+) cycle=\s*(\d+) busy=([01]) mac00_nonzero=([01])")
COMMAND = re.compile(r"ACT_COMMAND opcode=\s*(\d+) cycle=\s*(\d+) funct=\s*(\d+) rs1=\s*(\d+) rs2=\s*(\d+)")
PASS = re.compile(r"ACT_PASS name=(\w+) mode=(\w+) dataset=(\w+) iteration=(\d+) cycles=(\d+) begin=(\d+) end=(\d+) base=(\d+)")
OPCODES = {0: 123, 1: 91}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tile(addr):
    return (16 << 48) | (16 << 32) | addr


def expected_commands(calls, base, repetitions=1):
    """Independent encoding audit against the pinned ISA, not a C-code parser."""
    result = []

    def command(lane, funct, rs1, rs2):
        result.append((OPCODES[lane], funct, rs1, rs2))

    def preload(lane, right, output):
        command(lane, 6, tile(right), tile(output | (1 << 31)))

    def compute(lane, left):
        command(lane, 4, tile(left), tile(0xFFFFFFFF))

    # IEEE754 1.0; same no-activation, WS, row-stride configuration for both.
    scale = 0x3F800000 << 32
    for lane in (0, 1):
        command(lane, 7, 0, 0)
        command(lane, 0, scale | (1 << 16) | 4, 1 << 48)
        command(lane, 0, scale | (16 << 16) | (1 << 8) | 1, 16)
        command(lane, 0, 2, scale | 64)
    for name, args in calls * repetitions:
        if name == "double_gemm":
            preload(0, args["i1"], args["o0"])
            preload(1, args["i3"], args["o1"])
            compute(0, args["i0"])
            compute(1, args["i2"])
            continue
        operation, suffix = name.rsplit("_", 1)
        lane = int(suffix)
        if operation == "mvin":
            command(lane, 2, base + args["i0"], tile(args["o0"]))
        elif operation == "mvout":
            command(lane, 3, base + args["o0"], tile(args["i0"] | (1 << 31) | (1 << 29)))
        elif operation == "requantize_store":
            command(lane, 0, 2, scale | 16)
            command(lane, 3, base + args["o0"], tile(args["i0"] | (1 << 31)))
            command(lane, 0, 2, scale | 64)
        elif operation == "gemm":
            preload(lane, args["i1"], args["o0"])
            compute(lane, args["i0"])
        else:
            raise ValueError(f"No ISA audit for {name}")
    return result


def intervals(events, field):
    start = None
    result = []
    for cycle, busy, mac in events:
        active = (busy, mac)[field]
        if active and start is None:
            start = cycle
        elif not active and start is not None:
            result.append((start, cycle))
            start = None
    if start is not None:
        raise AssertionError("Simulator stopped with an active engine")
    return result


def intersection(left, right):
    return sum(max(0, min(b, d) - max(a, c)) for a, b in left for c, d in right)


def inspect_trace(trace):
    events = {123: [], 91: []}
    commands = []
    for line in trace.splitlines():
        match = ACTIVITY.fullmatch(line.strip())
        if match:
            opcode, cycle, busy, mac = map(int, match.groups())
            if opcode not in events:
                raise AssertionError("Unknown observed engine")
            if events[opcode] and cycle <= events[opcode][-1][0]:
                raise AssertionError("Non-monotonic engine clock")
            if mac and not busy:
                raise AssertionError("MAC activity outside engine busy interval")
            events[opcode].append((cycle, busy, mac))
        match = COMMAND.fullmatch(line.strip())
        if match:
            opcode, cycle, funct, rs1, rs2 = map(int, match.groups())
            commands.append((cycle, opcode, funct, rs1, rs2))
    if any(not values for values in events.values()) or not commands:
        raise AssertionError("Missing RTL observations; build with ACT_GEMMINI_TRACE=1 and run +verbose")
    # Two commands may be accepted on the same edge; compare per engine below.
    busy = {op: intervals(values, 0) for op, values in events.items()}
    mac = {op: intervals(values, 1) for op, values in events.items()}
    return commands, busy, mac


def read_measurements(stdout, provenance):
    """Read the original firmware output, including its cycle arithmetic."""
    if "ACT_FAIL" in stdout:
        raise AssertionError("Firmware numerical/address checks failed")
    lines = [line.strip() for line in stdout.splitlines() if "ACT_PASS" in line]
    if not lines or len(lines) != provenance["repetitions"]:
        raise AssertionError("Each invocation must pass numerical checks and report cycles")
    measurements = []
    for index, line in enumerate(lines):
        match = PASS.fullmatch(line)
        if match is None:
            raise AssertionError("Malformed firmware cycle record")
        name, mode, dataset, iteration, elapsed, begin, end, base = match.groups()
        iteration, elapsed, begin, end, base = map(int, (iteration, elapsed, begin, end, base))
        if iteration != index or (name, mode, dataset) != (provenance["workload"], provenance["mode"], str(provenance["dataset"])):
            raise AssertionError("Wrong workload/mode/input/iteration log")
        if elapsed != end - begin or elapsed <= 0:
            raise AssertionError("Invalid simulated cycle measurement")
        if base != 0x81000040:
            raise AssertionError("Modes must use the same fixed physical DRAM placement")
        measurements.append(elapsed)
    return measurements, base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise ValueError("Use a fresh report path")
    run = verify_run_record(args.log, args.binary)
    provenance_path = args.binary.with_suffix(".json")
    provenance = json.loads(provenance_path.read_text())
    if sha(args.binary.with_suffix(".c")) != provenance["source_sha256"]:
        raise AssertionError("Generated test source changed")
    build = json.loads(Path(str(args.binary) + ".build.json").read_text())
    if build["binary_sha256"] != sha(args.binary) or build["source_sha256"] != provenance["source_sha256"]:
        raise AssertionError("Test source/binary provenance differs")
    measurements, base = read_measurements(args.log.read_text(), provenance)
    trace_path = Path(str(args.log) + ".trace")
    commands, busy, mac = inspect_trace(trace_path.read_text())
    expected = expected_commands(provenance["calls"], base, provenance["repetitions"])
    for opcode in OPCODES.values():
        actual_lane = [(op, func, rs1, rs2) for _, op, func, rs1, rs2 in commands if op == opcode]
        expected_lane = [entry for entry in expected if entry[0] == opcode]
        if actual_lane != expected_lane:
            raise AssertionError(f"Native command/row address mismatch on opcode {opcode}: {actual_lane} != {expected_lane}")
    overlap = intersection(mac[123], mac[91])
    joint = sum(name == "double_gemm" for name, _ in provenance["calls"])
    if joint == 0 and overlap != 0:
        raise AssertionError("Sequential instruction completion unexpectedly overlapped computation")
    # A zero or identity matrix need not activate this conservative nonzero probe.
    if joint and isinstance(provenance["dataset"], int) and overlap == 0:
        raise AssertionError("Joint command produced no witnessed PE multiplication overlap")
    name, mode, dataset = provenance["workload"], provenance["mode"], str(provenance["dataset"])
    report = {"rtl_passed": True, "workload": name, "mode": mode, "dataset": dataset,
              "cycles": measurements, "pe00_nonzero_overlap_cycles": overlap,
              "busy_overlap_cycles": intersection(busy[123], busy[91]),
              "pe00_nonzero_intervals": mac, "double_gemm_count": joint,
              "accepted_command_count": len(commands), "provenance": provenance,
              "build": build,
              "binary_sha256": run["binary_sha256"], "rtl_sha256": run["rtl_sha256"],
              "run_record_sha256": sha(record_path(args.log)),
              "stdout_sha256": sha(args.log), "trace_sha256": sha(trace_path)}
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{name}/{mode}/{dataset}: {measurements} cycles (first/repeated), PE00 overlap {overlap}, exact outputs/commands")


if __name__ == "__main__":
    main()
