"""Unconstrained pair/triple counterpart for discovery microbenchmarks.

The storage and scalar instructions are identical to ProbeISA. The joint
semantics deliberately have no shared input variables, so their legal match
space is the Cartesian product even under MultiPattern.
"""
from act.generators import generate_backend, generate_oracle
from ProbeISA import declare, probe


probe.name = "IndependentPatternProbe"
probe.instructions = [instruction for instruction in probe.instructions
                      if instruction.instruction not in {"shared_pair", "shared_triple", "reshape_pair"}]

declare("independent_pair", [("d8", "8")] * 4, [("d8", "8")] * 2, """
    %x = bf16[8,8] parameter(0);
    %a = bf16[8,8] parameter(1);
    %y = bf16[8,8] parameter(2);
    %b = bf16[8,8] parameter(3);
    %sum = bf16[8,8] add(%x, %a);
    %product = bf16[8,8] multiply(%y, %b);
    ROOT %result = (bf16[8,8], bf16[8,8]) tuple(%sum, %product);
""")

declare("independent_triple", [("d8", "8")] * 6, [("d8", "8")] * 3, """
    %x = bf16[8,8] parameter(0);
    %a = bf16[8,8] parameter(1);
    %y = bf16[8,8] parameter(2);
    %b = bf16[8,8] parameter(3);
    %z = bf16[8,8] parameter(4);
    %c = bf16[8,8] parameter(5);
    %sum = bf16[8,8] add(%x, %a);
    %product = bf16[8,8] multiply(%y, %b);
    %matrix = bf16[8,8] dot(%z, %c), lhs_contracting_dims={1}, rhs_contracting_dims={0};
    ROOT %result = (bf16[8,8], bf16[8,8], bf16[8,8]) tuple(%sum, %product, %matrix);
""")


if __name__ == "__main__":
    generate_oracle(probe)
    generate_backend(probe)
