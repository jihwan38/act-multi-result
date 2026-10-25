"""Small generated ISA for shared-input, three-result, and metadata experiments.

Run with an ACT source environment and ACT_BUILD_DIR to generate its compiler
and Oracle. All operations also have single-result implementations, so a
missed joint candidate remains observable as a larger correct assembly.
"""
import os
from pathlib import Path

from act.taidl import Accelerator
from act.generators import generate_backend, generate_oracle


build_dir = Path(os.environ.get("ACT_BUILD_DIR", Path(__file__).resolve().parents[2] / "build/multipattern"))
probe = Accelerator("MultiPatternProbe", base_dir=str(build_dir))
for width in (8, 16, 32):
    probe.add_data_model(f"d{width}", [4096], [width], "bf16")


def declare(name, inputs, outputs, body, attributes=()):
    """Declare one-dimensional row-addressed ports and explicit HLO semantics."""
    parameters = [f"i{index}" for index in range(len(inputs))]
    parameters += [f"o{index}" for index in range(len(outputs))]
    instruction = probe.add_instruction(name, list(attributes), parameters)
    instruction.set_inputs([(buffer, [f"@a.i{index}"], [length])
                            for index, (buffer, length) in enumerate(inputs)])
    instruction.set_outputs([(buffer, [f"@a.o{index}"], [length])
                             for index, (buffer, length) in enumerate(outputs)])
    instruction.add_semantics(f"ENTRY {name} {{\n{body}\n}}")


declare("load8", [("d0", "@c.n * 16")], [("d8", "@c.n")], """
    %input = u8[`@c.n * 16`] parameter(0);
    %bytes = u8[`@c.n`,8,2] reshape(%input);
    ROOT %result = bf16[`@c.n`,8] bitcast_convert(%bytes);
""", ["n"])

for width in (8, 16, 32):
    declare(f"store{width}", [(f"d{width}", "@c.n")], [("d0", f"@c.n * {width * 2}")], f"""
        %input = bf16[`@c.n`,{width}] parameter(0);
        %bytes = u8[`@c.n`,{width},2] bitcast_convert(%input);
        ROOT %result = u8[`@c.n * {width * 2}`] reshape(%bytes);
    """, ["n"])

for operation in ("add", "multiply", "dot"):
    suffix = ", lhs_contracting_dims={1}, rhs_contracting_dims={0}" if operation == "dot" else ""
    declare(f"{operation}8", [("d8", "8"), ("d8", "8")], [("d8", "8")], f"""
        %x = bf16[8,8] parameter(0);
        %y = bf16[8,8] parameter(1);
        ROOT %result = bf16[8,8] {operation}(%x, %y){suffix};
    """)

for width, rows in ((16, 2), (32, 1)):
    declare(f"reshape{width}", [("d8", "4")], [(f"d{width}", str(rows))], f"""
        %x = bf16[4,8] parameter(0);
        ROOT %result = bf16[{rows},{width}] reshape(%x);
    """)

declare("shared_pair", [("d8", "8")] * 3, [("d8", "8")] * 2, """
    %x = bf16[8,8] parameter(0);
    %a = bf16[8,8] parameter(1);
    %b = bf16[8,8] parameter(2);
    %sum = bf16[8,8] add(%x, %a);
    %product = bf16[8,8] multiply(%x, %b);
    ROOT %result = (bf16[8,8], bf16[8,8]) tuple(%sum, %product);
""")

declare("shared_triple", [("d8", "8")] * 4, [("d8", "8")] * 3, """
    %x = bf16[8,8] parameter(0);
    %a = bf16[8,8] parameter(1);
    %b = bf16[8,8] parameter(2);
    %c = bf16[8,8] parameter(3);
    %sum = bf16[8,8] add(%x, %a);
    %product = bf16[8,8] multiply(%x, %b);
    %matrix = bf16[8,8] dot(%x, %c), lhs_contracting_dims={1}, rhs_contracting_dims={0};
    ROOT %result = (bf16[8,8], bf16[8,8], bf16[8,8]) tuple(%sum, %product, %matrix);
""")

declare("reshape_pair", [("d8", "4")], [("d16", "2"), ("d32", "1")], """
    %x = bf16[4,8] parameter(0);
    %first = bf16[2,16] reshape(%x);
    %second = bf16[1,32] reshape(%x);
    ROOT %result = (bf16[2,16], bf16[1,32]) tuple(%first, %second);
""")

# These stores make the discovery checks independent of a workload ROOT tuple.
declare("join_pair_store", [("d8", "8")] * 2, [("d0", "256")], """
    %first = bf16[8,8] parameter(0);
    %second = bf16[8,8] parameter(1);
    %joined = bf16[16,8] concatenate(%first, %second), dimensions={0};
    %bytes = u8[16,8,2] bitcast_convert(%joined);
    ROOT %result = u8[256] reshape(%bytes);
""")
declare("join_triple_store", [("d8", "8")] * 3, [("d0", "384")], """
    %first = bf16[8,8] parameter(0);
    %second = bf16[8,8] parameter(1);
    %third = bf16[8,8] parameter(2);
    %pair = bf16[16,8] concatenate(%first, %second), dimensions={0};
    %joined = bf16[24,8] concatenate(%pair, %third), dimensions={0};
    %bytes = u8[24,8,2] bitcast_convert(%joined);
    ROOT %result = u8[384] reshape(%bytes);
""")


if __name__ == "__main__":
    generate_oracle(probe)
    generate_backend(probe)
