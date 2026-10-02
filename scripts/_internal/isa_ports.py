"""Read the example ISA ports for allocation checks, independently of Backend.

Supports the tutorial's one-dimensional addresses and integer length expressions
such as @c.n * 128. This is a test helper, not an ACT generation interface.
"""
import ast
import operator
import re


def length_value(expression, attributes):
    expression = re.sub(r"@c\.([A-Za-z_]\w*)", r"\1", str(expression))
    operations = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                  ast.FloorDiv: operator.floordiv}

    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.Name) and node.id in attributes:
            return attributes[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in operations:
            return operations[type(node.op)](evaluate(node.left), evaluate(node.right))
        raise ValueError(f"Unsupported port length: {expression}")

    value = evaluate(ast.parse(expression, mode="eval").body)
    if type(value) is not int or value <= 0:
        raise ValueError(f"Nonpositive port length: {expression}")
    return value


def describe(target):
    buffers = {}
    for model in target.data_model:
        name = "HBM" if model.var_name == "d0" else model.var_name.upper()
        if name != "HBM" and len(model.access_dim) != 1:
            raise ValueError("The example audit requires one-dimensional buffers")
        buffers[name] = {"capacity": model.access_dim[0] if model.access_dim else None,
                         "unit_shape": model.unit_dim, "dtype": model.var_type}
    instructions = {}
    for inst in target.instructions:
        if inst.constraints or inst.update:
            raise ValueError("Use the dedicated constraint tests for this ISA")

        def ports(items):
            result = []
            for buffer, addresses, lengths in items:
                if (len(addresses) != 1 or len(lengths) != 1
                        or not re.fullmatch(r"@a\.[A-Za-z_]\w*", addresses[0])):
                    raise ValueError("Unsupported port descriptor")
                name = "HBM" if buffer == "d0" else buffer.upper()
                param = addresses[0][3:]
                if param not in inst.parameters:
                    raise ValueError("Undeclared address parameter")
                result.append({"buffer": name, "param": param, "length_expr": str(lengths[0]),
                               "unit_shape": buffers[name]["unit_shape"], "dtype": buffers[name]["dtype"]})
            return result

        inputs, outputs = ports(inst.instr_inputs), ports(inst.instr_outputs)
        if {item["param"] for item in inputs + outputs} != set(inst.parameters):
            raise ValueError("Every address attribute must be used by a port")
        instructions[inst.instruction] = {"parameters": inst.parameters, "comp_attr": inst.comp_attr,
                                          "inputs": inputs, "outputs": outputs}
    return {"buffers": buffers, "instructions": instructions}


def resolve_ports(instruction, params):
    if set(params) != set(instruction["parameters"]) | set(instruction["comp_attr"]):
        raise ValueError("Assembly attributes differ from the ISA declaration")
    attributes = {name: params[name] for name in instruction["comp_attr"]}
    result = {}
    for role in ("inputs", "outputs"):
        result[role] = []
        for port in instruction[role]:
            length = length_value(port["length_expr"], attributes)
            result[role].append({"buffer": port["buffer"], "param": port["param"], "dtype": port["dtype"],
                                 "length": length, "shape": [length] + port["unit_shape"]})
    return result
