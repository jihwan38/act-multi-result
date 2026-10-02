"""Read generated straight-line ACT assembly and its I/O metadata."""
import ast
from dataclasses import dataclass


@dataclass(eq=True)
class Call:
    op: str
    params: dict


def literal(node):
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "jnp":
        return "jnp." + node.attr
    if isinstance(node, (ast.List, ast.Tuple)):
        return [literal(x) for x in node.elts]
    if isinstance(node, ast.Dict):
        return {literal(k): literal(v) for k, v in zip(node.keys, node.values)}
    return ast.literal_eval(node)


def parse(text):
    tree = ast.parse(text)
    functions = [x for x in tree.body if isinstance(x, ast.FunctionDef)]
    if len(functions) != 1:
        raise ValueError("Expected one assembly factory")
    factory = functions[0]
    kernels = [x for x in factory.body if isinstance(x, ast.FunctionDef)]
    if len(kernels) != 1 or len(kernels[0].decorator_list) != 1:
        raise ValueError("Expected one decorated kernel")
    body = kernels[0]
    decorator = body.decorator_list[0]
    if not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Name) or decorator.func.id != "kernel" or decorator.args:
        raise ValueError("Invalid kernel declaration")
    metadata = {x.arg: ast.literal_eval(x.value) if x.arg in ("input_tree", "output_tree")
                else literal(x.value) for x in decorator.keywords}
    required = {"hbm", "input", "output", "constant"}
    if (not required <= set(metadata) or set(metadata) - required - {"input_tree", "output_tree"}
            or len(metadata) != len(decorator.keywords) or metadata["constant"]):
        raise ValueError("Expected explicit input/output ABI without embedded constants")
    for key, descriptors in (("input_tree", "input"), ("output_tree", "output")):
        if key not in metadata:
            continue
        def leaf_indices(node):
            if type(node) is int:
                return [node]
            if isinstance(node, list):
                return [leaf for child in node for leaf in leaf_indices(child)]
            raise ValueError("Invalid logical value tree")
        if key == "input_tree" and not isinstance(metadata[key], list):
            raise ValueError("input_tree must describe a parameter list")
        if leaf_indices(metadata[key]) != list(range(len(metadata[descriptors]))):
            raise ValueError("Logical tree/ABI leaf count or order mismatch")
    metadata["module_name"] = factory.name
    calls = []
    for statement in body.body:
        if isinstance(statement, ast.Pass):
            continue
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
            raise ValueError("Only straight-line API calls are supported")
        call = statement.value
        if not isinstance(call.func, ast.Attribute) or not isinstance(call.func.value, ast.Name) or call.func.value.id != "api" or call.args:
            raise ValueError("Invalid API call")
        params = {x.arg: literal(x.value) for x in call.keywords}
        if len(params) != len(call.keywords) or None in params or any(type(x) is not int or x < 0 for x in params.values()):
            raise ValueError("Address parameters must be unique nonnegative integers")
        calls.append(Call(call.func.attr, params))
    return tree, body, metadata, calls


def dtype_bytes(dtype):
    name = dtype.removeprefix("jnp.")
    return {"bf16": 2, "bfloat16": 2, "f32": 4, "float32": 4, "u8": 1,
            "uint8": 1, "s32": 4, "int32": 4}[name]
