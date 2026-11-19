"""Fixed 16x16, WS, INT8 -> INT32 Gemmini command macros.

Each GEMM lowers to native preload + compute_preloaded, not a fictitious opcode.
DoubleGEMM issues both native pairs before its completion fence. Explicit mvin
and full-width accumulator mvout instructions retain ACT-assigned row addresses.
An explicit requantize_store uses native identity-scale INT8 saturation.
No activation, nonidentity scaling, bias, accumulation, transpose or auto-tiling.
"""

from act.taidl import Accelerator

DIM = 16
SP_ROWS = 256 * 1024 // DIM
ACC_ROWS = 64 * 1024 // (DIM * 4)


def make_target(base_dir):
    target = Accelerator("GemminiDual", base_dir=str(base_dir))
    for lane in range(2):
        target.add_data_model(f"sp_{lane}", [SP_ROWS], [DIM], "s8")
        target.add_data_model(f"acc_{lane}", [ACC_ROWS], [DIM], "s32")

    def instruction(name, inputs, outputs, semantics):
        addresses = [f"i{i}" for i in range(len(inputs))]
        addresses += [f"o{i}" for i in range(len(outputs))]
        inst = target.add_instruction(name, [], addresses)
        inst.set_inputs([
            (buffer, [f"@a.i{i}"], [str(rows)])
            for i, (buffer, rows) in enumerate(inputs)
        ])
        inst.set_outputs([
            (buffer, [f"@a.o{i}"], [str(rows)])
            for i, (buffer, rows) in enumerate(outputs)
        ])
        inst.add_semantics(f"ENTRY {name} {{\n{semantics}\n}}")

    def parameters(count):
        return "\n".join(
            f"%In{i + 1} = s8[16,16] parameter({i});" for i in range(count)
        )

    for lane in range(2):
        sp, acc = f"sp_{lane}", f"acc_{lane}"
        instruction(f"mvin_{lane}", [("d0", 256)], [(sp, DIM)], """
%In1 = u8[256] parameter(0);
%bytes = u8[16,16] reshape(%In1);
ROOT %Out0 = s8[16,16] bitcast_convert(%bytes);
""")
        instruction(f"gemm_{lane}", [(sp, DIM)] * 2, [(acc, DIM)],
                    parameters(2) + """
ROOT %Out0 = s32[16,16] dot(%In1, %In2), lhs_contracting_dims={1}, rhs_contracting_dims={0};
""")
        instruction(f"mvout_{lane}", [(acc, DIM)], [("d0", 1024)], """
%In1 = s32[16,16] parameter(0);
%bytes = u8[16,16,4] bitcast_convert(%In1);
ROOT %Out0 = u8[1024] reshape(%bytes);
""")

        def constant_matrix(value):
            row = "{" + ",".join([str(value)] * DIM) + "}"
            return "{" + ",".join([row] * DIM) + "}"

        instruction(f"requantize_store_{lane}", [(acc, DIM)], [("d0", 256)], f"""
%In1 = s32[16,16] parameter(0);
%lo = s32[16,16] constant(`{constant_matrix(-128)}`);
%hi = s32[16,16] constant(`{constant_matrix(127)}`);
%lower = s32[16,16] maximum(%In1, %lo);
%clamped = s32[16,16] minimum(%lower, %hi);
%quantized = s8[16,16] convert(%clamped);
%bytes = u8[16,16] bitcast_convert(%quantized);
ROOT %Out0 = u8[256] reshape(%bytes);
""")
    instruction("double_gemm", [("sp_0", DIM)] * 2 + [("sp_1", DIM)] * 2,
                [("acc_0", DIM), ("acc_1", DIM)], parameters(4) + """
ROOT %Out0 = s32[16,16] dot(%In1, %In2), lhs_contracting_dims={1}, rhs_contracting_dims={0};
ROOT %Out1 = s32[16,16] dot(%In3, %In4), lhs_contracting_dims={1}, rhs_contracting_dims={0};
""")
    return target
