"""Tutorial QKV on two lanes, plus DoubleGEMM and join_store.

Select multi-result with scripts/env.sh, then run this file to generate the
Oracle and compiler. workloads/two_gemm_concat.hlo contains ordinary HLO dots
and concatenate; ACT discovers their implementation using this ISA.

Each lane keeps all seven tutorial instructions so separate GEMMs remain valid
alternatives. The two additions are at the end: double_gemm produces two
64x64 results in separate D2 buffers, and join_store writes one 128x64 output.
ROOT tuple order determines which result is written to each output port.
"""
import os
from pathlib import Path

from act.taidl import Accelerator
from act.generators import generate_oracle, generate_backend

# Keep generated files under build/; ACT_BUILD_DIR can select another directory.
build_dir = Path(os.environ.get("ACT_BUILD_DIR", Path(__file__).resolve().parents[2] / "build/multi-result"))
qkv = Accelerator("QKVDoubleGemm", base_dir=str(build_dir))

# Each lane has the tutorial's D1 (128 rows) and D2 (64 rows).
# Both lanes use the same byte-addressable HBM (implicit d0).
qkv.add_data_model("d1_0", [128], [64], "bf16")
qkv.add_data_model("d2_0", [64], [64], "bf16")
qkv.add_data_model("d1_1", [128], [64], "bf16")
qkv.add_data_model("d2_1", [64], [64], "bf16")

# Lane 0: the same seven tutorial instructions.

# (1) load_rm_0: Loads data from HBM (d0) in row-major format to d1_0
instr = qkv.add_instruction("load_rm_0", ["n"], ["addr_in", "addr_out"])
instr.set_inputs([["d0", ["@a.addr_in"], ["@c.n * 128"]]])  # u8[@c.n * 128]
instr.set_outputs([["d1_0", ["@a.addr_out"], ["@c.n"]]])  # bf16[@c.n, 64]
instr.add_semantics("""
ENTRY load_rm_0 {
    %In1 = u8[`@c.n * 128`] parameter(0);
    %a = u8[`@c.n`,64,2] reshape(%In1);
    ROOT %Out0 = bf16[`@c.n`,64] bitcast_convert(%a);
}
""")

# (2) load_cm_0: Loads data from HBM (d0) in column-major format to d1_0 (with transpose)
instr = qkv.add_instruction("load_cm_0", ["n"], ["addr_in", "addr_out"])
instr.set_inputs([["d0", ["@a.addr_in"], ["@c.n * 128"]]])  # u8[@c.n * 128]
instr.set_outputs([["d1_0", ["@a.addr_out"], ["@c.n"]]])  # bf16[@c.n, 64]
instr.add_semantics("""
ENTRY load_cm_0 {
    %In1 = u8[`@c.n * 128`] parameter(0);
    %a = u8[`@c.n`,64,2] reshape(%In1);
    %b = bf16[`@c.n`,64] bitcast_convert(%a);
    ROOT %Out0 = bf16[64,`@c.n`] transpose(%b), dimensions={1,0};
}
""")

# (3) store_rm_0: Stores data from d1_0 to HBM (d0) in row-major format
instr = qkv.add_instruction("store_rm_0", ["n"], ["addr_in", "addr_out"])
instr.set_inputs([["d1_0", ["@a.addr_in"], ["@c.n"]]])  # bf16[@c.n, 64]
instr.set_outputs([["d0", ["@a.addr_out"], ["@c.n * 128"]]])  # u8[@c.n * 128]
instr.add_semantics("""
ENTRY store_rm_0 {
    %In1 = bf16[`@c.n`,64] parameter(0);
    %a = u8[`@c.n`,64,2] bitcast_convert(%In1);
    ROOT %Out0 = u8[`@c.n*128`] reshape(%a);
}
""")

# (4) store_cm_0: Stores data from d1_0 to HBM (d0) in column-major format (with transpose)
instr = qkv.add_instruction("store_cm_0", ["n"], ["addr_in", "addr_out"])
instr.set_inputs([["d1_0", ["@a.addr_in"], ["@c.n"]]])  # bf16[@c.n, 64]
instr.set_outputs([["d0", ["@a.addr_out"], ["@c.n * 128"]]])  # u8[@c.n * 128]
instr.add_semantics("""
ENTRY store_cm_0 {
    %In1 = bf16[`@c.n`,64] parameter(0);
    %a = bf16[64,`@c.n`] transpose(%In1), dimensions={1,0};
    %b = u8[64,`@c.n`,2] bitcast_convert(%a);
    ROOT %Out0 = u8[`@c.n*128`] reshape(%b);
}
""")

# (5) mov_0: Copies data from d2_0 to d1_0
instr = qkv.add_instruction("mov_0", ["n"], ["addr_in", "addr_out"])
instr.set_inputs([["d2_0", ["@a.addr_in"], ["@c.n"]]])  # bf16[@c.n, 64]
instr.set_outputs([["d1_0", ["@a.addr_out"], ["@c.n"]]])  # bf16[@c.n, 64]
instr.add_semantics("""
ENTRY mov_0 {
    %In1 = bf16[`@c.n`,64] parameter(0);
    ROOT %Out0 = bf16[`@c.n`,64] copy(%In1);
}
""")

# (6) gemm_0: Matrix multiplication between two d1_0 tensors, output to d2_0
instr = qkv.add_instruction("gemm_0", [], ["addr_1", "addr_2", "addr_out"])
instr.set_inputs([["d1_0", ["@a.addr_1"], ["64"]], ["d1_0", ["@a.addr_2"], ["64"]]])
instr.set_outputs([["d2_0", ["@a.addr_out"], ["64"]]])  # bf16[64, 64]
instr.add_semantics("""
ENTRY gemm_0 {
    %In1 = bf16[64,64] parameter(0);
    %In2 = bf16[64,64] parameter(1);
    ROOT %Out0 = bf16[64,64] dot(%In1, %In2), lhs_contracting_dims={1}, rhs_contracting_dims={0};
}
""")

# (7) softmax_0: Applies softmax_0 along dimension 1 (rows) on d2_0
instr = qkv.add_instruction("softmax_0", ["n"], ["addr"])
instr.set_inputs([["d2_0", ["@a.addr"], ["@c.n"]]])  # bf16[@c.n, 64]
instr.set_outputs([["d2_0", ["@a.addr"], ["@c.n"]]])  # bf16[@c.n, 64]
instr.add_semantics("""
ENTRY softmax_0 {
    %In1 = bf16[`@c.n`,64] parameter(0);
    %a = bf16[`@c.n`,64] exponential(%In1);
    %reduced = bf16[`@c.n`] reduce_add(%a), dimensions={1};
    %b = bf16[`@c.n`,64] broadcast(%reduced), dimensions={0};
    ROOT %Out0 = bf16[`@c.n`,64] divide(%a, %b);
}
""")

# Lane 1: the same seven tutorial instructions.

# (8) load_rm_1: Loads data from HBM (d0) in row-major format to d1_1
instr = qkv.add_instruction("load_rm_1", ["n"], ["addr_in", "addr_out"])
instr.set_inputs([["d0", ["@a.addr_in"], ["@c.n * 128"]]])  # u8[@c.n * 128]
instr.set_outputs([["d1_1", ["@a.addr_out"], ["@c.n"]]])  # bf16[@c.n, 64]
instr.add_semantics("""
ENTRY load_rm_1 {
    %In1 = u8[`@c.n * 128`] parameter(0);
    %a = u8[`@c.n`,64,2] reshape(%In1);
    ROOT %Out0 = bf16[`@c.n`,64] bitcast_convert(%a);
}
""")

# (9) load_cm_1: Loads data from HBM (d0) in column-major format to d1_1 (with transpose)
instr = qkv.add_instruction("load_cm_1", ["n"], ["addr_in", "addr_out"])
instr.set_inputs([["d0", ["@a.addr_in"], ["@c.n * 128"]]])  # u8[@c.n * 128]
instr.set_outputs([["d1_1", ["@a.addr_out"], ["@c.n"]]])  # bf16[@c.n, 64]
instr.add_semantics("""
ENTRY load_cm_1 {
    %In1 = u8[`@c.n * 128`] parameter(0);
    %a = u8[`@c.n`,64,2] reshape(%In1);
    %b = bf16[`@c.n`,64] bitcast_convert(%a);
    ROOT %Out0 = bf16[64,`@c.n`] transpose(%b), dimensions={1,0};
}
""")

# (10) store_rm_1: Stores data from d1_1 to HBM (d0) in row-major format
instr = qkv.add_instruction("store_rm_1", ["n"], ["addr_in", "addr_out"])
instr.set_inputs([["d1_1", ["@a.addr_in"], ["@c.n"]]])  # bf16[@c.n, 64]
instr.set_outputs([["d0", ["@a.addr_out"], ["@c.n * 128"]]])  # u8[@c.n * 128]
instr.add_semantics("""
ENTRY store_rm_1 {
    %In1 = bf16[`@c.n`,64] parameter(0);
    %a = u8[`@c.n`,64,2] bitcast_convert(%In1);
    ROOT %Out0 = u8[`@c.n*128`] reshape(%a);
}
""")

# (11) store_cm_1: Stores data from d1_1 to HBM (d0) in column-major format (with transpose)
instr = qkv.add_instruction("store_cm_1", ["n"], ["addr_in", "addr_out"])
instr.set_inputs([["d1_1", ["@a.addr_in"], ["@c.n"]]])  # bf16[@c.n, 64]
instr.set_outputs([["d0", ["@a.addr_out"], ["@c.n * 128"]]])  # u8[@c.n * 128]
instr.add_semantics("""
ENTRY store_cm_1 {
    %In1 = bf16[`@c.n`,64] parameter(0);
    %a = bf16[64,`@c.n`] transpose(%In1), dimensions={1,0};
    %b = u8[64,`@c.n`,2] bitcast_convert(%a);
    ROOT %Out0 = u8[`@c.n*128`] reshape(%b);
}
""")

# (12) mov_1: Copies data from d2_1 to d1_1
instr = qkv.add_instruction("mov_1", ["n"], ["addr_in", "addr_out"])
instr.set_inputs([["d2_1", ["@a.addr_in"], ["@c.n"]]])  # bf16[@c.n, 64]
instr.set_outputs([["d1_1", ["@a.addr_out"], ["@c.n"]]])  # bf16[@c.n, 64]
instr.add_semantics("""
ENTRY mov_1 {
    %In1 = bf16[`@c.n`,64] parameter(0);
    ROOT %Out0 = bf16[`@c.n`,64] copy(%In1);
}
""")

# (13) gemm_1: Matrix multiplication between two d1_1 tensors, output to d2_1
instr = qkv.add_instruction("gemm_1", [], ["addr_1", "addr_2", "addr_out"])
instr.set_inputs([["d1_1", ["@a.addr_1"], ["64"]], ["d1_1", ["@a.addr_2"], ["64"]]])
instr.set_outputs([["d2_1", ["@a.addr_out"], ["64"]]])  # bf16[64, 64]
instr.add_semantics("""
ENTRY gemm_1 {
    %In1 = bf16[64,64] parameter(0);
    %In2 = bf16[64,64] parameter(1);
    ROOT %Out0 = bf16[64,64] dot(%In1, %In2), lhs_contracting_dims={1}, rhs_contracting_dims={0};
}
""")

# (14) softmax_1: Applies softmax along dimension 1 (rows) on d2_1
instr = qkv.add_instruction("softmax_1", ["n"], ["addr"])
instr.set_inputs([["d2_1", ["@a.addr"], ["@c.n"]]])  # bf16[@c.n, 64]
instr.set_outputs([["d2_1", ["@a.addr"], ["@c.n"]]])  # bf16[@c.n, 64]
instr.add_semantics("""
ENTRY softmax_1 {
    %In1 = bf16[`@c.n`,64] parameter(0);
    %a = bf16[`@c.n`,64] exponential(%In1);
    %reduced = bf16[`@c.n`] reduce_add(%a), dimensions={1};
    %b = bf16[`@c.n`,64] broadcast(%reduced), dimensions={0};
    ROOT %Out0 = bf16[`@c.n`,64] divide(%a, %b);
}
""")

# (15) Four inputs -> two independent dot results, from one ISA instruction.
# ROOT tuple element 0 -> output port 0 (d2_0, addr_out0).
# ROOT tuple element 1 -> output port 1 (d2_1, addr_out1).
instr = qkv.add_instruction("double_gemm", [], ["addr_1", "addr_2", "addr_3", "addr_4", "addr_out0", "addr_out1"])
instr.set_inputs([
    ("d1_0", ["@a.addr_1"], ["64"]),
    ("d1_0", ["@a.addr_2"], ["64"]),
    ("d1_1", ["@a.addr_3"], ["64"]),
    ("d1_1", ["@a.addr_4"], ["64"]),
])
instr.set_outputs([
    ("d2_0", ["@a.addr_out0"], ["64"]),
    ("d2_1", ["@a.addr_out1"], ["64"]),
])
instr.add_semantics("""
ENTRY double_gemm {
    %In1 = bf16[64,64] parameter(0);
    %In2 = bf16[64,64] parameter(1);
    %In3 = bf16[64,64] parameter(2);
    %In4 = bf16[64,64] parameter(3);
    %left = bf16[64,64] dot(%In1, %In2), lhs_contracting_dims={1}, rhs_contracting_dims={0};
    %right = bf16[64,64] dot(%In3, %In4), lhs_contracting_dims={1}, rhs_contracting_dims={0};
    ROOT %result = (bf16[64,64], bf16[64,64]) tuple(%left, %right);
}
""")

# (16) Concatenate the lane results vertically and store 128*64*2 bytes in HBM.
instr = qkv.add_instruction("join_store", [], ["addr_in0", "addr_in1", "addr_out"])
instr.set_inputs([
    ("d2_0", ["@a.addr_in0"], ["64"]),
    ("d2_1", ["@a.addr_in1"], ["64"]),
])
instr.set_outputs([("d0", ["@a.addr_out"], ["16384"])])
instr.add_semantics("""
ENTRY join_store {
    %In1 = bf16[64,64] parameter(0);
    %In2 = bf16[64,64] parameter(1);
    %joined = bf16[128,64] concatenate(%In1, %In2), dimensions={0};
    %bytes = u8[128,64,2] bitcast_convert(%joined);
    ROOT %Out0 = u8[16384] reshape(%bytes);
}
""")

if __name__ == "__main__":
    # Generate the programming API / functional simulator, then the compiler.
    generate_oracle(qkv)
    generate_backend(qkv)
