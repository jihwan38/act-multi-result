"""Deterministic HLO inputs for the generated MultiPatternProbe ISA."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Workload:
    name: str
    hlo: str
    require_calls: tuple = ()
    purpose: str = ""


class Hlo:
    def __init__(self, name):
        self.name, self.parameters, self.body = name, [], []

    def parameter(self, name, shape="bf16[8,8]"):
        self.body.append(f"  %{name} = {shape} parameter({len(self.parameters)})")
        self.parameters.append(shape)
        return name

    def operation(self, name, shape, operation):
        self.body.append(f"  %{name} = {shape} {operation}")
        return name

    def finish(self, outputs, concatenate=False):
        if concatenate:
            assert all(shape == "bf16[8,8]" for _, shape in outputs)
            previous = outputs[0][0]
            for index, (name, _) in enumerate(outputs[1:], 2):
                previous = self.operation(f"join{index}", f"bf16[{index * 8},8]",
                                          f"concatenate(%{previous}, %{name}), dimensions={{0}}")
            shape = f"bf16[{len(outputs) * 8},8]"
            # Use the final concatenate itself as ROOT: no explicit tuple or copy.
            self.body[-1] = self.body[-1].replace("  %", "  ROOT %", 1)
        else:
            shape = "(" + ", ".join(shape for _, shape in outputs) + ")"
            self.body.append(f"  ROOT %output = {shape} tuple({', '.join('%' + name for name, _ in outputs)})")
        header = f"HloModule {self.name}, entry_computation_layout={{({', '.join(self.parameters)}) -> {shape}}}"
        return header + "\n\nENTRY main {\n" + "\n".join(self.body) + "\n}\n"


def arithmetic_group(hlo, group, triple=False, shared=True, repeated=False, dependent=False):
    x = hlo.parameter(f"x{group}")
    a = x if repeated else hlo.parameter(f"a{group}")
    y = x if shared else hlo.parameter(f"y{group}")
    b = hlo.parameter(f"b{group}") if not dependent else None
    c = hlo.parameter(f"c{group}") if triple else None
    first = hlo.operation(f"sum{group}", "bf16[8,8]", f"add(%{x}, %{a})")
    second = hlo.operation(f"product{group}", "bf16[8,8]", f"multiply(%{y}, %{first if dependent else b})")
    outputs = [(first, "bf16[8,8]"), (second, "bf16[8,8]")]
    if triple:
        third = hlo.operation(f"matrix{group}", "bf16[8,8]",
                              f"dot(%{x}, %{c}), lhs_contracting_dims={{1}}, rhs_contracting_dims={{0}}")
        outputs.append((third, "bf16[8,8]"))
    return outputs


def arithmetic(name, groups=1, triple=False, shared=True, concatenate=False, repeated=False, dependent=False):
    hlo = Hlo(name)
    outputs = [output for group in range(groups)
               for output in arithmetic_group(hlo, group, triple, shared, repeated, dependent)]
    required = (("shared_triple" if triple else "shared_pair"),) if shared and groups == 1 and not dependent else ()
    return Workload(name, hlo.finish(outputs, concatenate), required,
                    f"{groups} {'shared' if shared else 'different'} input groups; {'three' if triple else 'two'} results")


def mixed():
    hlo = Hlo("mixed_pair_triple")
    outputs = arithmetic_group(hlo, 0, triple=True) + arithmetic_group(hlo, 1)
    return Workload(hlo.name, hlo.finish(outputs), ("shared_pair", "shared_triple"),
                    "Two joint rule families must both reach the selected assembly")


def reshape(name, shared=True):
    hlo = Hlo(name)
    x = hlo.parameter("x", "bf16[4,8]")
    y = x if shared else hlo.parameter("y", "bf16[4,8]")
    first = hlo.operation("first", "bf16[2,16]", f"reshape(%{x})")
    second = hlo.operation("second", "bf16[1,32]", f"reshape(%{y})")
    return Workload(name, hlo.finish([(first, "bf16[2,16]"), (second, "bf16[1,32]")]),
                    ("reshape_pair",) if shared else (),
                    "Metadata-bearing results with shared input" if shared else "Different reshape inputs must stay distinct")


def cases():
    return [
        arithmetic("shared_pair_tuple"),
        arithmetic("shared_pair_concat", concatenate=True),
        arithmetic("shared_triple_tuple", triple=True),
        arithmetic("shared_triple_concat", triple=True, concatenate=True),
        arithmetic("different_pair_inputs", shared=False),
        arithmetic("dependent_pair", dependent=True),
        arithmetic("repeated_pair_input", repeated=True),
        mixed(),
        reshape("shared_reshape"),
        reshape("different_reshape_inputs", shared=False),
        *[arithmetic(f"shared_pairs_{groups}", groups=groups) for groups in (4, 16)],
        *[arithmetic(f"shared_triples_{groups}", groups=groups, triple=True) for groups in (4, 16)],
    ]
