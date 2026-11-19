"""Lower static ACT instructions to native Gemmini commands without regrouping.

The two hand-written controls use canonical independent GEMM schedules. Golden
arrays are calculated independently on the host, not by the lowering routines.
The firmware checks every output and input byte outside the timed kernel.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from assembly import read_program, run_address_model
from verify import make_inputs, reference


def lower_calls(calls):
    lines = []
    for name, args in calls:
        if name == "double_gemm":
            fields = [args[key] for key in ("i0", "i1", "i2", "i3", "o0", "o1")]
            lines.append(f"act_double_gemm({', '.join(map(str, fields))});")
        else:
            operation, lane = name.rsplit("_", 1)
            if operation == "mvin":
                lines.append(f"act_mvin({lane}, memory + {args['i0']}, {args['o0']});")
            elif operation == "mvout":
                lines.append(f"act_mvout({lane}, {args['i0']}, memory + {args['o0']});")
            elif operation == "requantize_store":
                lines.append(f"act_requantize_store({lane}, {args['i0']}, memory + {args['o0']});")
            elif operation == "gemm":
                lines.append(f"act_gemm({lane}, {args['i0']}, {args['i1']}, {args['o0']});")
            else:
                raise ValueError(f"No native implementation of {name}")
    return lines


def independent_pairs(name):
    if name == "two_gemm":
        return [(0, 1), (2, 3)]
    if name == "shared_input":
        return [(0, 1), (0, 2)]
    if name == "four_gemm":
        return [(0, 1), (2, 3), (4, 5), (6, 7)]
    raise ValueError(f"No independent native control for {name}")


def manual_calls(program, parallel):
    inputs, outputs = program.metadata["input"], program.metadata["output"]
    if program.name == "dependent_gemm":
        # No parallel reference exists across this true dependency. Reuse the
        # final output region as a temporary BEFORE producing the final INT32s.
        temporary = outputs[0]["addr"]
        return [("mvin_0", {"i0": inputs[0]["addr"], "o0": 0}),
                ("mvin_0", {"i0": inputs[1]["addr"], "o0": 16}),
                ("gemm_0", {"i0": 0, "i1": 16, "o0": 0}),
                ("requantize_store_0", {"i0": 0, "o0": temporary}),
                ("mvin_1", {"i0": temporary, "o0": 0}),
                ("mvin_1", {"i0": inputs[2]["addr"], "o0": 16}),
                ("gemm_1", {"i0": 0, "i1": 16, "o0": 0}),
                ("mvout_1", {"i0": 0, "o0": outputs[0]["addr"]})]
    pairs = independent_pairs(program.name)
    calls = []
    for start in range(0, len(pairs), 2):
        pair = pairs[start:start + 2]
        if len(pair) != 2:
            raise ValueError("This control requires pairs of independent GEMMs")
        for lane, (left, right) in enumerate(pair):
            calls += [(f"mvin_{lane}", {"i0": inputs[left]["addr"], "o0": 0}),
                      (f"mvin_{lane}", {"i0": inputs[right]["addr"], "o0": 16})]
        if parallel:
            calls += [("double_gemm", {"i0": 0, "i1": 16, "i2": 0, "i3": 16, "o0": 0, "o1": 0})]
        else:
            calls += [(f"gemm_{lane}", {"i0": 0, "i1": 16, "o0": 0}) for lane in (0, 1)]
        calls += [(f"mvout_{lane}", {"i0": 0, "o0": outputs[start + lane]["addr"]})
                  for lane in (0, 1)]
    return calls


def c_array(name, ctype, values):
    flat = np.asarray(values).reshape(-1)
    rows = [", ".join(str(int(item)) for item in flat[start:start + 16])
            for start in range(0, flat.size, 16)]
    return f"static const {ctype} {name}[{flat.size}] = {{\n    " + ",\n    ".join(rows) + "\n};\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly", type=Path, required=True)
    parser.add_argument("--mode", choices=("act", "sequential", "manual_parallel"), required=True)
    parser.add_argument("--dataset", default="901")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise ValueError("Use fresh output paths")
    program = read_program(args.assembly)
    dataset = int(args.dataset) if args.dataset.isdigit() else args.dataset
    inputs = make_inputs(len(program.metadata["input"]), dataset)
    expected = reference(program.name, inputs)
    if args.mode != "act":
        program.calls = manual_calls(program, args.mode == "manual_parallel")
    actual = run_address_model(program, inputs)
    for value, golden in zip(actual, expected):
        np.testing.assert_array_equal(value, golden)
    memory_size = program.metadata["hbm"]
    initial = np.full(memory_size, 0xA5, dtype=np.uint8)
    for descriptor, value in zip(program.metadata["input"], inputs):
        addr = descriptor["addr"]
        initial[addr:addr + value.nbytes] = value.view(np.uint8).reshape(-1)
    prefix = '#include <stdint.h>\n#include <stdio.h>\n#include "gemmini_dual.h"\n'
    prefix += c_array("initial", "uint8_t", initial)
    for index, value in enumerate(expected):
        prefix += c_array(f"expected_{index}", "int32_t", value)
    body = "\n    ".join(lower_calls(program.calls))
    checks = []
    for index, descriptor in enumerate(program.metadata["output"]):
        checks.append(f'''for (unsigned i = 0; i < 256; ++i) {{
        int32_t value = ((volatile int32_t *)(memory + {descriptor['addr']}))[i];
        if (value != expected_{index}[i]) {{
            printf("ACT_FAIL output={index} index=%u actual=%d expected=%d\\n",
                   i, value, expected_{index}[i]);
            return 1;
        }}
    }}''')
    for descriptor in program.metadata["input"]:
        checks.append(f'''for (unsigned i = {descriptor['addr']}; i < {descriptor['addr'] + 256}; ++i)
        if (memory[i] != initial[i]) {{ printf("ACT_FAIL input overwritten\\n"); return 2; }}''')
    checks_text = "\n    ".join(checks)
    source = prefix + f'''
static uint8_t guarded_memory[{memory_size + 128}] __attribute__((aligned(64), section(".act_memory")));
static uint8_t *const memory = guarded_memory + 64;

__attribute__((noinline, aligned(64))) static void run_kernel(void) {{
    {body}
}}

int main(void) {{
    for (unsigned i = 0; i < sizeof(guarded_memory); ++i) guarded_memory[i] = 0xD3;
    for (unsigned i = 0; i < {memory_size}; ++i) memory[i] = initial[i];
    act_wait();
    act_configure(0);
    act_configure(1);
    act_wait();
    uint64_t begin[2], end[2];
    for (unsigned iteration = 0; iteration < 2; ++iteration) {{
        begin[iteration] = act_cycles();
        run_kernel();
        act_wait();
        end[iteration] = act_cycles();
        {checks_text}
        for (unsigned i = 0; i < 64; ++i) {{
            if (guarded_memory[i] != 0xD3 || guarded_memory[{memory_size + 64} + i] != 0xD3) {{
                printf("ACT_FAIL output guard overwritten\\n"); return 3;
            }}
        }}
    }}
    for (unsigned iteration = 0; iteration < 2; ++iteration)
        printf("ACT_PASS name={program.name} mode={args.mode} dataset={dataset} iteration=%u cycles=%lu begin=%lu end=%lu base=%lu\\n",
               iteration, end[iteration] - begin[iteration], begin[iteration], end[iteration], (uint64_t)memory);
    return 0;
}}
'''
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(source)
    provenance = {"workload": program.name, "mode": args.mode, "dataset": dataset, "repetitions": 2,
                  "assembly_sha256": hashlib.sha256(args.assembly.read_bytes()).hexdigest(),
                  "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                  "calls": program.calls, "metadata": program.metadata,
                  "input_sha256": hashlib.sha256(b"".join(value.tobytes() for value in inputs)).hexdigest(),
                  "expected_sha256": [hashlib.sha256(value.tobytes()).hexdigest() for value in expected],
                  "hardware_execution_verified": False}
    args.output.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
