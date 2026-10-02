"""Prefix 0004: generator/template output preserves separate buffer lists."""

import unittest
from types import SimpleNamespace

from act_backend.ir_buffer_rs_generator import (
    extract_instruction_info,
    generate_buffer_signature_arms,
)
from act_backend.template_loader import get_backend_template_loader


class BufferSignatureGeneration(unittest.TestCase):
    def render(self, outputs, inputs, comp_attrs=()):
        instruction = SimpleNamespace(
            instruction="probe",
            instr_outputs=[(name, ["@a.addr"], ["64"]) for name in outputs],
            instr_inputs=[(name, ["@a.addr"], ["64"]) for name in inputs],
            comp_attr=comp_attrs,
        )
        info = extract_instruction_info([instruction])
        templates = get_backend_template_loader()
        arms = generate_buffer_signature_arms(info, templates)
        return arms[0]

    def test_same_buffer_does_not_erase_the_boundary(self):
        rendered = self.render(["d1", "d1"], ["d1", "d1", "d1", "d1"])
        self.assertIn("outputs: vec![Buffer::D1, Buffer::D1]", rendered)
        self.assertIn(
            "inputs: vec![Buffer::D1, Buffer::D1, Buffer::D1, Buffer::D1]",
            rendered,
        )

    def test_declared_buffer_order_and_hbm_mapping(self):
        rendered = self.render(["d2", "d1", "d0"], ["d0", "d2"])
        self.assertIn(
            "outputs: vec![Buffer::D2, Buffer::D1, Buffer::HBM]", rendered,
        )
        self.assertIn("inputs: vec![Buffer::HBM, Buffer::D2]", rendered)

    def test_zero_inputs_and_computation_attributes(self):
        rendered = self.render(["d2"], [], ["n"])
        self.assertIn("TensorOp::Probe(_, _)", rendered)
        self.assertIn("outputs: vec![Buffer::D2]", rendered)
        self.assertIn("inputs: vec![]", rendered)

    def test_no_result_instruction_is_not_a_value_candidate(self):
        self.assertIn("=> None,", self.render([], ["d1"]))


if __name__ == "__main__":
    unittest.main()
