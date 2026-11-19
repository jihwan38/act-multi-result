"""Check literal contents, not merely matching dimensions or parser acceptance."""

import unittest

from act.taidl import Accelerator
from act_backend.generator import _get_instruction_metadata
from act_backend.isel_ir2isa_rewrites_rs_generator import generate_ir2isa_rust_functions
from act_backend.semantics import lower_semantics
from taidl_to.semantic_generator import generate_semantic_code


def fixture(literal):
    target = Accelerator("IntegerLiteralProbe")
    target.add_data_model("scratch", [16], [2], "s32")
    instruction = target.add_instruction("lower_bound", [], ["i", "o"])
    instruction.set_inputs([("scratch", ["@a.i"], ["2"])])
    instruction.set_outputs([("scratch", ["@a.o"], ["2"])])
    instruction.add_semantics(f"""ENTRY lower_bound {{
%In1 = s32[2,2] parameter(0);
%bound = s32[2,2] constant(`{literal}`);
ROOT %Out0 = s32[2,2] maximum(%In1, %bound);
}}""")
    return _get_instruction_metadata(target)[0], instruction.instr_semantics_ast


class IntegerSemanticsTests(unittest.TestCase):
    def test_preserves_all_integer_elements(self):
        metadata, tree = fixture("{{-128, 127}, {0, 1}}")
        semantics = lower_semantics(metadata)
        constant, = [node for node in semantics.postorder if node.operation == "constant"]
        self.assertEqual(constant.attrs["literal"], "{{-128,127},{0,1}}")
        rust = generate_ir2isa_rust_functions(metadata, None)
        self.assertIn('"{{-128,127},{0,1}}".chars()', rust)
        oracle = generate_semantic_code(tree)
        self.assertIn("const = '{{-128, 127}, {0, 1}}'", oracle)
        self.assertNotIn("simplify_vals", oracle)

    def test_rejects_ambiguous_scalar_array_and_invalid_literals(self):
        for literal in ("-128", "{-128,127}", "{{-128},{127}}",
                        "{{0,0},{0,2147483648}}", "{{1.5,0},{0,0}}", "@a.i"):
            with self.subTest(literal=literal):
                metadata, _ = fixture(literal)
                with self.assertRaises(ValueError):
                    lower_semantics(metadata)


if __name__ == "__main__":
    unittest.main()
