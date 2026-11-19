"""Read ACT's static assembly and check its exact byte/row address assignments."""

import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from GemminiDual import ACC_ROWS, DIM, SP_ROWS


@dataclass
class Program:
    name: str
    metadata: dict
    calls: list


def literal(node):
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        if node.value.id == "jnp" and node.attr in ("int8", "int32"):
            return node.attr
    if isinstance(node, (ast.List, ast.Tuple)):
        return [literal(item) for item in node.elts]
    if isinstance(node, ast.Dict):
        return {literal(key): literal(value) for key, value in zip(node.keys, node.values)}
    return ast.literal_eval(node)


def read_program(path):
    tree = ast.parse(Path(path).read_text())
    outer, = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    inner, = [node for node in outer.body if isinstance(node, ast.FunctionDef)]
    decorator, = inner.decorator_list
    if not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Name):
        raise ValueError("Expected the generated kernel decorator")
    if decorator.func.id != "kernel" or decorator.args:
        raise ValueError("Unexpected kernel invocation")
    metadata = {item.arg: literal(item.value) for item in decorator.keywords}
    calls = []
    for node in inner.body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            raise ValueError("Only static instruction calls are supported")
        call = node.value
        if (not isinstance(call.func, ast.Attribute)
                or not isinstance(call.func.value, ast.Name)
                or call.func.value.id != "api" or call.args):
            raise ValueError("Unexpected instruction call")
        kwargs = {item.arg: literal(item.value) for item in call.keywords}
        if any(type(value) is not int or value < 0 for value in kwargs.values()):
            raise ValueError("Expected concrete nonnegative ACT addresses")
        calls.append((call.func.attr, kwargs))
    return Program(outer.name, metadata, calls)


def run_address_model(program, inputs):
    """An independent integer interpreter, not a timing or hardware simulator.

    Track initialization at byte granularity. Read every input before writing any
    result of a multi-result instruction. The actual hardware must independently
    establish the completion boundary assumed here.
    """
    meta = program.metadata
    if meta["constant"]:
        raise ValueError("Constants have not been added to the native ABI")
    if len(inputs) != len(meta["input"]):
        raise ValueError("Wrong input count")
    buffers = {"hbm": np.zeros(meta["hbm"], dtype=np.uint8)}
    for lane in (0, 1):
        buffers[f"sp{lane}"] = np.zeros(SP_ROWS * DIM, dtype=np.uint8)
        buffers[f"acc{lane}"] = np.zeros(ACC_ROWS * DIM * 4, dtype=np.uint8)
    written = {name: np.zeros(value.size, dtype=bool) for name, value in buffers.items()}

    def region(name, addr, size):
        if type(addr) is not int or addr < 0 or addr + size > buffers[name].size:
            raise ValueError(f"Out-of-bounds {name}[{addr}:{addr + size}]")
        return slice(addr, addr + size)

    def read(name, addr, size, dtype):
        part = region(name, addr, size)
        if not written[name][part].all():
            raise ValueError(f"Read-before-write: {name}[{addr}:{addr + size}]")
        return buffers[name][part].copy().view(dtype).reshape(DIM, DIM)

    def write(name, addr, value):
        raw = np.ascontiguousarray(value).view(np.uint8).reshape(-1)
        part = region(name, addr, raw.size)
        buffers[name][part] = raw
        written[name][part] = True

    occupied = np.zeros(meta["hbm"], dtype=bool)
    for descriptor, value in zip(meta["input"], inputs):
        if descriptor["shape"] != [DIM, DIM] or descriptor["dtype"] != "int8":
            raise ValueError("Unexpected native input shape/type")
        if value.shape != (DIM, DIM) or value.dtype != np.int8:
            raise ValueError("Input data type/shape mismatch")
        part = region("hbm", descriptor["addr"], 256)
        if occupied[part].any():
            raise ValueError("Overlapping input ABI slots")
        occupied[part] = True
        write("hbm", descriptor["addr"], value)
    for descriptor in meta["output"]:
        if descriptor["shape"] != [DIM, DIM] or descriptor["dtype"] != "int32":
            raise ValueError("Unexpected native output shape/type")
        if descriptor["addr"] % 4:
            raise ValueError("Unaligned INT32 output")
        part = region("hbm", descriptor["addr"], 1024)
        if occupied[part].any():
            raise ValueError("Overlapping input/output ABI slots")
        occupied[part] = True

    for name, args in program.calls:
        if name == "double_gemm":
            if set(args) != {"i0", "i1", "i2", "i3", "o0", "o1"}:
                raise ValueError("Wrong DoubleGEMM ABI")
            results = []
            for lane in (0, 1):
                left = read(f"sp{lane}", args[f"i{2 * lane}"] * DIM, 256, np.int8)
                right = read(f"sp{lane}", args[f"i{2 * lane + 1}"] * DIM, 256, np.int8)
                results.append((left.astype(np.int64) @ right.astype(np.int64)).astype(np.int32))
            for lane, result in enumerate(results):
                write(f"acc{lane}", args[f"o{lane}"] * DIM * 4, result)
            continue
        try:
            operation, suffix = name.rsplit("_", 1)
            lane = int(suffix)
        except (ValueError, TypeError) as error:
            raise ValueError(f"Unknown instruction {name}") from error
        if lane not in (0, 1):
            raise ValueError("Invalid Gemmini engine")
        if operation in ("mvin", "mvout", "requantize_store") and set(args) == {"i0", "o0"}:
            if operation == "mvin":
                value = read("hbm", args["i0"], 256, np.int8)
                write(f"sp{lane}", args["o0"] * DIM, value)
            else:
                value = read(f"acc{lane}", args["i0"] * DIM * 4, 1024, np.int32)
                if operation == "requantize_store":
                    value = np.clip(value, -128, 127).astype(np.int8)
                write("hbm", args["o0"], value)
        elif operation == "gemm" and set(args) == {"i0", "i1", "o0"}:
            left = read(f"sp{lane}", args["i0"] * DIM, 256, np.int8)
            right = read(f"sp{lane}", args["i1"] * DIM, 256, np.int8)
            result = (left.astype(np.int64) @ right.astype(np.int64)).astype(np.int32)
            write(f"acc{lane}", args["o0"] * DIM * 4, result)
        else:
            raise ValueError(f"Unsupported instruction/ABI: {name} {args}")
    return [read("hbm", item["addr"], 1024, np.int32) for item in meta["output"]]
