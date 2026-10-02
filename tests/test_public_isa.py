"""Public examples preserve the tutorial instructions and expose only two additions."""
from dataclasses import asdict
import json
from pathlib import Path
import re
import runpy
import unittest
from unittest.mock import patch

from _internal.isa_ports import describe, length_value, resolve_ports

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("instruction", "comp_attr", "parameters", "constraints", "cost", "update",
          "instr_inputs", "instr_outputs", "instr_semantics_str")


def signature(target):
    return {"models": [asdict(model) for model in target.data_model],
            "instructions": [{field: getattr(inst, field) for field in FIELDS} for inst in target.instructions]}


class PublicIsaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch("act.generators.generate_oracle"), patch("act.generators.generate_backend"):
            cls.tutorial = runpy.run_path(str(ROOT / "third_party/act/tutorials/pldi26/exercise3/solution/QKV.py"))["qkv"]
        cls.single = runpy.run_path(str(ROOT / "cases/qkv/QKV.py"))["qkv"]
        cls.double = runpy.run_path(str(ROOT / "cases/double_gemm/QKVDoubleGemm.py"))["qkv"]

    def test_qkv_preserves_all_tutorial_declarations(self):
        self.assertEqual(signature(self.single), signature(self.tutorial))

    def test_each_lane_preserves_the_tutorial_instructions(self):
        baseline = {i.instruction: i for i in self.tutorial.instructions}
        public = {i.instruction: i for i in self.double.instructions}
        self.assertEqual(set(public), {f"{op}_{lane}" for lane in (0, 1) for op in baseline}
                         | {"double_gemm", "join_store"})
        for lane in (0, 1):
            for op, inst in baseline.items():
                data = {field: getattr(public[f"{op}_{lane}"], field) for field in FIELDS}
                text = json.dumps(data)
                for name in (op, "d1", "d2"):
                    text = re.sub(r"\b" + name + f"_{lane}" + r"\b", name, text)
                self.assertEqual(json.loads(text), json.loads(json.dumps({field: getattr(inst, field) for field in FIELDS})))
            for original in self.tutorial.data_model[1:]:
                actual = next(m for m in self.double.data_model if m.var_name == f"{original.var_name}_{lane}")
                self.assertEqual((actual.access_dim, actual.unit_dim, actual.var_type),
                                 (original.access_dim, original.unit_dim, original.var_type))

    def test_tutorial_n_is_resolved_per_call_in_the_audit(self):
        isa = describe(self.double)
        for n in (32, 64):
            ports = resolve_ports(isa["instructions"]["load_rm_0"], {"n": n, "addr_in": 0, "addr_out": 64})
            self.assertEqual(ports["inputs"][0]["shape"], [n * 128])
            self.assertEqual(ports["outputs"][0]["shape"], [n, 64])
        ports = resolve_ports(isa["instructions"]["softmax_1"], {"n": 64, "addr": 0})
        self.assertEqual(ports["inputs"], ports["outputs"])

    def test_audit_rejects_unknown_attributes_and_unsupported_length_expressions(self):
        instr = describe(self.double)["instructions"]["load_rm_0"]
        for params in ({"addr_in": 0, "addr_out": 0}, {"n": 64, "extra": 1, "addr_in": 0, "addr_out": 0}):
            with self.assertRaises(ValueError):
                resolve_ports(instr, params)
        for text in ("@c.m * 128", "n()", "n.real", "0", "n - 64"):
            with self.assertRaises(ValueError):
                length_value(text, {"n": 64})


if __name__ == "__main__":
    unittest.main()
