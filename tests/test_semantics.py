"""One explicit tuple return: ordered ports shared by TAIDL, Backend and Oracle."""
import unittest

from act_backend.generator import _get_instruction_metadata
from act_backend.semantics import lower_semantics
from act_backend.isel_ir2isa_rewrites_rs_generator import generate_ir2isa_rust_functions
from act_backend.template_loader import get_backend_template_loader
from taidl import Accelerator
from taidl.semantics import semantic_interface
from taidl_to.instruction_codegen import generate_api_function, generate_semantic_function


def instruction(body, count=4, outputs=2, attrs=()):
    target = Accelerator("Test", base_dir="/unused")
    target.add_data_model("scratch", [256], [64], "bf16")
    inst = target.add_instruction("joint", list(attrs), ["a"])
    inst.set_inputs([("scratch", ["@a.a"], ["64"])] * count)
    inst.set_outputs([("scratch", ["@a.a"], ["64"])] * outputs)
    inst.add_semantics("ENTRY joint { " + body + " }")
    return target, inst


def metadata(body, **kwargs):
    return _get_instruction_metadata(instruction(body, **kwargs)[0])[0]


PARAMETERS = " ".join(f"%arg{i} = bf16[64,64] parameter({i});" for i in range(4))
LEFT = "%left = bf16[64,64] dot(%arg0, %arg1), lhs_contracting_dims={1}, rhs_contracting_dims={0};"
RIGHT = "%right = bf16[64,64] dot(%arg2, %arg3), lhs_contracting_dims={1}, rhs_contracting_dims={0};"
RETURN = "ROOT %result = (bf16[64,64], bf16[64,64]) tuple(%left, %right);"
BODY = PARAMETERS + LEFT + RIGHT + RETURN


class SemanticsTests(unittest.TestCase):
    def test_return_order_overrides_declaration_order(self):
        lowered = lower_semantics(metadata(PARAMETERS + RIGHT + LEFT + RETURN))
        self.assertEqual(lowered.pattern, "(tuple (dot ?p0 ?p1) (dot ?p2 ?p3))")
        self.assertEqual(lowered.parameters, ["?p0", "?p1", "?p2", "?p3"])
        self.assertEqual(len(lowered.postorder), 6)
        reversed_body = BODY.replace("tuple(%left, %right)", "tuple(%right, %left)")
        self.assertEqual(lower_semantics(metadata(reversed_body)).pattern,
                         "(tuple (dot ?p2 ?p3) (dot ?p0 ?p1))")

    def test_output_names_do_not_encode_port_numbers(self):
        body = BODY.replace("%left", "%Out9").replace("%right", "%Out0")
        self.assertEqual(lower_semantics(metadata(body)).pattern,
                         "(tuple (dot ?p0 ?p1) (dot ?p2 ?p3))")

    def test_parameter_index_overrides_declaration_order(self):
        parameters = " ".join(f"%arg{i} = bf16[64,64] parameter({i});" for i in reversed(range(4)))
        target, inst = instruction(parameters + LEFT + RIGHT + RETURN)
        interface = semantic_interface(inst.instr_semantics_ast)
        self.assertEqual([node.IDENTIFIER().getText() for node in interface.inputs],
                         [f"%arg{i}" for i in range(4)])
        self.assertEqual(lower_semantics(_get_instruction_metadata(target)[0]).parameters,
                         ["?p0", "?p1", "?p2", "?p3"])

    def test_shared_input_keeps_the_same_pattern_variable(self):
        body = BODY.replace("%arg3 = bf16[64,64] parameter(3);", "")
        body = body.replace("%arg2, %arg3", "%arg0, %arg2")
        lowered = lower_semantics(metadata(body, count=3))
        self.assertEqual(lowered.pattern, "(tuple (dot ?p0 ?p1) (dot ?p0 ?p2))")

    def test_internal_semantic_dependency_is_preserved(self):
        body = BODY.replace("%arg3 = bf16[64,64] parameter(3);", "")
        body = body.replace("%arg2, %arg3", "%left, %arg2")
        self.assertEqual(lower_semantics(metadata(body, count=3)).pattern,
                         "(tuple (dot ?p0 ?p1) (dot (dot ?p0 ?p1) ?p2))")

    def test_result_count_matches_physical_outputs(self):
        with self.assertRaisesRegex(ValueError, "result/output count mismatch"):
            metadata(BODY, outputs=1)

    def test_parameter_count_matches_physical_inputs(self):
        with self.assertRaisesRegex(ValueError, "parameter/input count mismatch"):
            metadata(BODY, count=3)

    def test_legacy_multiple_roots_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly one ROOT.*tuple"):
            metadata(PARAMETERS + "ROOT " + LEFT + "ROOT " + RIGHT)

    def test_missing_root_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly one ROOT"):
            metadata(PARAMETERS + LEFT + RIGHT)

    def test_invalid_result_tuple_is_rejected(self):
        cases = [
            (RETURN.replace("%right);", "%missing);"), "Undefined tuple"),
            ("ROOT %result = (bf16[64,64]) tuple(%left, %right);", "arity"),
            ("ROOT %result = () tuple();", "at least two"),
            ("ROOT %result = (bf16[64,64]) tuple(%left);", "at least two"),
            (RETURN.replace("(bf16[64,64],", "(f32[64,64],"), "type does not match"),
            (RETURN.replace("(bf16[64,64],", "(bf16[32,64],"), "type does not match"),
            (RETURN.replace("(bf16[64,64],", "((bf16[64,64], bf16[64,64]),"), "nested tuple"),
            ("ROOT %result = bf16[64,64] tuple(%left, %right);", "tuple result type"),
            (RETURN.replace(");", "), dimensions={0};"), "without attributes"),
        ]
        for returned, error in cases:
            with self.subTest(returned=returned), self.assertRaisesRegex(ValueError, error):
                metadata(PARAMETERS + LEFT + RIGHT + returned)

    def test_forward_return_operand_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Undefined tuple"):
            metadata(PARAMETERS + LEFT + RETURN + RIGHT)

    def test_duplicate_symbols_and_parameters_are_rejected(self):
        for body, error in ((PARAMETERS + LEFT + LEFT + RIGHT + RETURN, "Repeated semantics symbol"),
                            (BODY.replace("parameter(3)", "parameter(2)"), "Repeated semantics parameter"),
                            (BODY.replace("parameter(3)", "parameter(4)"), "contiguous")):
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                metadata(body)

    def test_intermediate_tuple_is_not_an_isa_tensor(self):
        body = PARAMETERS + LEFT + RIGHT + RETURN.replace("ROOT ", "") + "ROOT %out = bf16[64,64] copy(%left);"
        with self.assertRaisesRegex(ValueError, "Only the ROOT"):
            metadata(body, outputs=1)

    def test_unused_isa_operand_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "contribute"):
            lower_semantics(metadata(BODY.replace("%arg2, %arg3", "%arg2, %arg2")))

    def test_uninferrable_shape_attribute_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "attributes"):
            lower_semantics(metadata(BODY, attrs=["n"]))

    def test_nonstandard_dot_dimensions_not_silently_rewritten(self):
        with self.assertRaisesRegex(ValueError, "matrix product"):
            lower_semantics(metadata(BODY.replace("lhs_contracting_dims={1}", "lhs_contracting_dims={0}")))

    def test_single_tensor_root_still_works_with_arbitrary_name(self):
        body = " ".join(f"%arg{i} = bf16[64,64] parameter({i});" for i in range(2)) + "ROOT " + LEFT
        lowered = lower_semantics(metadata(body, count=2, outputs=1))
        self.assertEqual(lowered.pattern, "(dot ?a ?b)")
        self.assertEqual(lowered.parameters, ["?a", "?b"])
        source = generate_ir2isa_rust_functions(metadata(body, count=2, outputs=1), get_backend_template_loader())
        self.assertNotIn("joint::inputs_available", source)

    def test_scalar_tensor_type_has_empty_shape(self):
        body = "%x = bf16[] parameter(0); ROOT %answer = bf16[] add(%x, %x);"
        self.assertEqual(lower_semantics(metadata(body, count=1, outputs=1)).results[0].shape, [])

    def test_one_root_with_multiple_results_keeps_input_guard(self):
        inst = metadata(BODY)
        self.assertEqual(sum(node.ROOT() is not None for node in inst.semantics_ast.instruction()), 1)
        source = generate_ir2isa_rust_functions(inst, get_backend_template_loader())
        self.assertIn("joint::inputs_available", source)
        self.assertIn("lhs_eclasses.len() != 7", source)
        self.assertIn("vec![None; 5]", source)
        self.assertNotIn("set_analysis_data", source)

    def test_oracle_binds_names_in_input_and_return_order(self):
        body = BODY.replace("tuple(%left, %right)", "tuple(%right, %left)")
        _, inst = instruction(body)
        for generate in (generate_api_function, generate_semantic_function):
            source = generate(inst)
            compile(source, "generated-oracle", "exec")
            self.assertIn("'arg0', 'bf16'", source)
            self.assertNotIn("lvars['In1']", source)
            self.assertNotIn("lvars['Out0']", source)
            self.assertNotIn("lvars['result']", source)
            stores = [line for line in source.splitlines() if 'rhs_update =' in line]
            self.assertEqual(len(stores), 2)
            self.assertIn("'right'", stores[0])
            self.assertIn("'left'", stores[1])


if __name__ == "__main__":
    unittest.main()
