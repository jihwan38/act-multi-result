"""Mutate real compiler output to ensure address audits reject broken records."""
from copy import deepcopy
import json
from pathlib import Path
import re
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _internal.audit import audit
from _internal.assembly import parse
from _internal.isa_ports import describe


def main():
    result = ROOT / "build/multi-result/runs/double_gemm/two_gemm_concat"
    allocation = json.loads((result / "assembly.py.allocation.json").read_text())
    assembly = (result / "assembly.py").read_text()
    selected_pii = re.search(r"^# PII number: (\d+)$", assembly, re.M)
    assert selected_pii, "Assembly must identify its selected PII"
    pii = (result / "compiler" / f"{selected_pii[1]}.pii").read_text()
    isa = describe(runpy.run_path(str(ROOT / "cases/double_gemm/QKVDoubleGemm.py"))["qkv"])
    audit(pii, allocation, assembly, isa)
    # This fixture exercises a two-result instruction. Keep that expectation
    # in the regression test; ordinary runs can select any valid assembly.
    calls = parse(assembly)[3]
    selected = sum(call.op == "double_gemm" for call in calls)
    if selected != 1:
        raise AssertionError(f"Regression fixture expected one DoubleGEMM, got {selected}")
    print("PASS: regression fixture selected one DoubleGEMM")
    joint = next(node for node in allocation["instructions"] if node["op"] == "double_gemm")
    first, second = joint["outputs"]
    mutants = []
    missing = deepcopy(allocation)
    del missing["tensors"][second]
    mutants.append(("missing-result", missing))
    address = deepcopy(allocation)
    address["tensors"][second]["offsets"][0] += 1
    mutants.append(("wrong-port-address", address))
    lifetime = deepcopy(allocation)
    lifetime["tensors"][first]["live"] = [0, 0]
    mutants.append(("wrong-lifetime", lifetime))
    duplicate = deepcopy(allocation)
    index = duplicate["instructions"].index(joint)
    duplicate["instructions"].insert(index + 1, deepcopy(joint))
    mutants.append(("duplicate-producer", duplicate))
    for label, altered in mutants:
        try:
            audit(pii, altered, assembly, isa)
        except (ValueError, KeyError) as error:
            print(f"PASS: rejected {label}: {error}")
        else:
            raise AssertionError(f"Audit accepted {label}")


if __name__ == "__main__":
    main()
