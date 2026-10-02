"""Check ACT C++ allocations against PII edges, ISA ports and recomputed lifetimes.

This module does not select instructions, allocate memory or rewrite assembly.
"""
import ast
import itertools
import math
import re

from .assembly import dtype_bytes, parse
from .isa_ports import resolve_ports


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_pii(text):
    outputs, values, nodes = None, {}, []
    descriptor = re.compile(r"(t\d+(?:\.\d+)?): (\w+)\[(-?\d+)\] = (\w+)\[([\d,]*)\]")
    for line in text.splitlines():
        if line.startswith("# outputs:"):
            require(outputs is None, "Repeated PII outputs")
            outputs = [name.strip() for name in line.split(":", 1)[1].split(",")]
        elif line and not line.startswith("#"):
            matches = list(descriptor.finditer(line))
            require(bool(matches), "Invalid PII result descriptor")
            names = []
            for match in matches:
                name, buffer, offset, dtype, shape = match.groups()
                require(name not in values, "Duplicate PII value")
                values[name] = {"buffer": buffer, "offset": int(offset), "dtype": dtype,
                                "shape": [int(dim) for dim in shape.split(",") if dim]}
                names.append(name)
            tail = line[matches[-1].end():].lstrip(") ")
            operation, arguments = tail.rsplit("(", 1)
            require(arguments.endswith(")"), "Invalid PII arguments")
            inputs = [item.strip() for item in arguments[:-1].split(",") if item.strip()]
            op = operation.split("[", 1)[0]
            nodes.append({"op": op, "operation": operation, "inputs": inputs, "outputs": names})
    require(outputs is not None and outputs, "Missing PII outputs")
    return outputs, values, nodes


def intersects(a, b):
    return a["buffer"] == b["buffer"] and all(
        x < y + ny and y < x + nx
        for x, nx, y, ny in zip(a["offsets"], a["sizes"], b["offsets"], b["sizes"])
    )


def audit(pii_text, allocation, assembly_text, abi):
    outputs, declared, pii_nodes = read_pii(pii_text)
    _, _, metadata, calls = parse(assembly_text)
    tensors, schedule = allocation["tensors"], allocation["instructions"]
    require(set(tensors) == set(declared), "Lost/extra PII results")
    require(allocation["outputs"] == outputs, "Changed output order")
    definitions, uses, aliases = {}, {name: [] for name in tensors}, {name: {name} for name in tensors}
    pii_by_outputs = {tuple(n["outputs"]): n for n in pii_nodes}
    leaves = {name for n in pii_nodes if n["op"] in ("Var", "DCC") for name in n["outputs"]}
    for name in leaves:
        definitions[name] = 0
    call_index = 0
    inplace = set()
    for step, node in enumerate(schedule, 1):
        source = pii_by_outputs[tuple(node["outputs"])]
        require((node["op"], node["inputs"]) == (source["op"], source["inputs"]), "Changed PII instruction/edge")
        for name in node["inputs"]:
            require(name in definitions, "Read-before-definition / dependency cycle")
            uses[name].append(step)
        for name in node["outputs"]:
            require(name not in definitions, "Producer emitted more than once")
            definitions[name] = step
        if node["view"]:
            require(node["op"] in ("slice", "concat") and len(node["outputs"]) == 1, "Invalid storage view")
            out_name = node["outputs"][0]
            out = tensors[out_name]
            for name in node["inputs"]:
                aliases[name].add(out_name)
                aliases[out_name].add(name)
                require(tensors[name]["buffer"] == out["buffer"], "Cross-buffer view")
            if node["op"] == "slice":
                cuts = re.search(r"'([^']+)'", source["operation"]).group(1)
                cuts = [tuple(map(int, dim.split(":"))) for dim in cuts.split(";")]
                parent = tensors[node["inputs"][0]]
                require(out["offsets"] == [base + cut[0] for base, cut in zip(parent["offsets"], cuts)]
                        and out["sizes"] == [end - start for start, end in cuts], "Incorrect slice placement")
            else:
                dim = int(re.search(r"'([^']+)'", source["operation"]).group(1))
                first, second = [tensors[name] for name in node["inputs"]]
                expected = first["offsets"].copy()
                expected[dim] += first["sizes"][dim]
                size = first["sizes"].copy()
                size[dim] += second["sizes"][dim]
                require(out["offsets"] == first["offsets"] and second["offsets"] == expected
                        and out["sizes"] == size, "Incorrect concat-view placement")
            continue
        require(call_index < len(calls), "Missing assembly instruction")
        call = calls[call_index]
        call_index += 1
        require(call.op == node["op"], "Assembly/PII instruction mismatch")
        emitted = ast.parse("api." + node["assembly"], mode="eval").body
        require(call.params == {kw.arg: ast.literal_eval(kw.value) for kw in emitted.keywords}, "Assembly differs from allocation")
        spec = resolve_ports(abi["instructions"][call.op], call.params)
        for role in ("inputs", "outputs"):
            require(len(node[role]) == len(spec[role]), "ISA/PII arity mismatch")
            for name, port in zip(node[role], spec[role]):
                value = tensors[name]
                require(value["buffer"] == port["buffer"] and value["dtype"] == port["dtype"]
                        and value["shape"] == port["shape"] and value["sizes"] == [port["length"]]
                        and value["offsets"] == [call.params[port["param"]]], "Wrong result port/address/type")
        for o, out in enumerate(spec["outputs"]):
            for i, inp in enumerate(spec["inputs"]):
                if out == inp:
                    inplace.add((node["outputs"][o], node["inputs"][i], step))
    require(call_index == len(calls) and set(definitions) == set(tensors), "Dropped/extra producer")

    raw_lifetimes = {}
    for name, value in tensors.items():
        expected = declared[name]
        require(all(value[key] == expected[key] for key in ("buffer", "dtype", "shape")), "PII result type changed")
        if expected["offset"] != -1:
            require(value["offsets"] == [expected["offset"]], "Fixed address changed")
        require(len(value["offsets"]) == len(value["sizes"]) == len(value["capacity"]), "Invalid address rank")
        require(all(0 <= offset and 0 < size and offset + size <= cap
                    for offset, size, cap in zip(value["offsets"], value["sizes"], value["capacity"])), "Out-of-bounds allocation")
        if value["buffer"] == "HBM":
            require(value["offsets"][0] + value["sizes"][0] <= metadata["hbm"], "HBM exceeds assembly allocation")
            if name in leaves:
                require(any(value["offsets"] == [item["addr"]]
                            and value["sizes"] == [math.prod(item["shape"]) * dtype_bytes(item["dtype"])]
                            for item in metadata["input"]), "Input does not match HBM ABI")
        else:
            require(value["capacity"] == [abi["buffers"][value["buffer"]]["capacity"]], "Capacity differs from ISA")
        first = definitions[name]
        last = len(schedule) + 1 if name in leaves or name in outputs else max([first] + uses[name])
        raw_lifetimes[name] = [first, last]
        require(value["live"] == [first, last], "Incorrect result lifetime")
    components, lifetimes = {}, {}
    for name in tensors:
        if name in components:
            continue
        pending, members = [name], set()
        while pending:
            current = pending.pop()
            if current in members:
                continue
            members.add(current)
            pending.extend(aliases[current] - members)
        span = [min(raw_lifetimes[x][0] for x in members), max(raw_lifetimes[x][1] for x in members)]
        for member in members:
            components[member], lifetimes[member] = name, span
    checked_pairs = 0
    for a, b in itertools.combinations(tensors, 2):
        if components[a] == components[b] or not intersects(tensors[a], tensors[b]):
            continue
        first = max(lifetimes[a][0], lifetimes[b][0])
        last = min(lifetimes[a][1], lifetimes[b][1])
        allowed = any((out, inp, first) in inplace and lifetimes[inp][1] == lifetimes[out][0]
                      for out, inp in ((a, b), (b, a)))
        require(first > last or allowed, f"Live values overlap: {a}, {b}")
        checked_pairs += 1
    require(len(outputs) == len(metadata["output"]), "Program output count mismatch")
    for name, output in zip(outputs, metadata["output"]):
        value = tensors[name]
        require(value["buffer"] == "HBM" and value["offsets"] == [output["addr"]]
                and value["sizes"] == [math.prod(output["shape"]) * dtype_bytes(output["dtype"])], "Wrong output ABI binding")
    return {"passed": True, "instructions": len(calls), "results": len(tensors),
            "outputs": len(outputs), "reused_address_pairs": checked_pairs}
