"""Generator regressions for common constraints and shared result semantics."""
import ast
import inspect
from pathlib import Path
import re
import unittest

from act.taidl import Accelerator
from act_backend import isel_ir2isa_rewrites_rs_generator as rust_generator
from act_backend import semantics
from act_backend.generator import _get_instruction_metadata
from act_backend.isel_ir2isa_rewrites_rs_generator import generate_ir2isa_rust_functions
from act_backend.isel_ir2isa_rewrites_txt_generator import generate_rewrite_rule
from act_backend.malloc_instructions_h_generator import generate_instruction_class
from act_backend.template_loader import get_backend_template_loader
from taidl_to.instruction_codegen import generate_semantic_function


def fixture(op="multiply", dynamic=False, outputs=3):
    accelerator = Accelerator("CorrectnessProbe")
    accelerator.add_data_model("scratch", [128], [8], "bf16")
    instruction = accelerator.add_instruction("compute", ["n"] if dynamic else [],
                                              ["i", "o"])
    size = "@c.n" if dynamic else "4"
    shape = "`@c.n`,8" if dynamic else "4,8"
    instruction.set_inputs([("scratch", ["@a.i"], [size])])
    instruction.set_outputs([("scratch", ["@a.o"], [size])] * outputs)
    body = f"%In1 = bf16[{shape}] parameter(0);\n"
    body += "\n".join(f"{'ROOT ' if outputs == 1 else ''}%Out{k} = bf16[{shape}] {op}(%In1, %In1);"
                      for k in range(outputs))
    if outputs > 1:
        types = ", ".join([f"bf16[{shape}]"] * outputs)
        values = ", ".join(f"%Out{k}" for k in range(outputs))
        body += f"\nROOT %result = ({types}) tuple({values});"
    instruction.add_semantics("ENTRY compute {\n" + body + "\n}")
    return accelerator, instruction


class CorrectnessGenerators(unittest.TestCase):
    def test_signed_integer_rewrites_match_backend_dtype_names(self):
        # Generate the INT8 -> INT32 rule which the BF16 examples do not cover.
        # Check against the backend's actual HLO-name -> Rust-variant contract.
        dtype_source = (Path(inspect.getfile(rust_generator)).parent
                        / "generic/src/ir/dtype.rs").read_text()
        variants = dict(re.findall(r'"(\w+)"\s*=>\s*Dtype::(\w+)', dtype_source))
        for outputs in (1, 2):
            with self.subTest(outputs=outputs):
                target = Accelerator("SignedGemmProbe")
                target.add_data_model("inputs", [64], [16], "s8")
                target.add_data_model("results", [64], [16], "s32")
                instruction = target.add_instruction("gemm", [], ["a", "b", "c", "d", "x", "y"])
                addresses = ["a", "b", "c", "d"][:outputs * 2]
                instruction.set_inputs([("inputs", [f"@a.{addr}"], ["16"]) for addr in addresses])
                instruction.set_outputs([("results", [f"@a.{addr}"], ["16"])
                                         for addr in ["x", "y"][:outputs]])
                body = "\n".join(f"%arg{i} = s8[16,16] parameter({i});" for i in range(outputs * 2))
                for i in range(outputs):
                    body += (f"\n{'ROOT ' if outputs == 1 else ''}%out{i} = s32[16,16] "
                             f"dot(%arg{i * 2}, %arg{i * 2 + 1}), "
                             "lhs_contracting_dims={1}, rhs_contracting_dims={0};")
                if outputs == 2:
                    body += "\nROOT %pair = (s32[16,16], s32[16,16]) tuple(%out0, %out1);"
                instruction.add_semantics("ENTRY gemm {\n" + body + "\n}")
                metadata = _get_instruction_metadata(target)[0]
                rust = generate_ir2isa_rust_functions(metadata, get_backend_template_loader())
                referenced = set(re.findall(r"Dtype::(\w+)", rust))
                self.assertEqual(referenced, {variants["s8"], variants["s32"]})

    def test_semantics_does_not_depend_on_code_generators(self):
        imports = [node.module for node in ast.walk(ast.parse(inspect.getsource(semantics)))
                   if isinstance(node, ast.ImportFrom)]
        self.assertFalse(any(name and "generator" in name for name in imports))
        for name in ("dim_rust", "shape_checks", "rust_functions"):
            self.assertFalse(hasattr(semantics, name))
            self.assertEqual(getattr(rust_generator, name).__module__, rust_generator.__name__)

    def test_rust_dimension_rendering(self):
        for text, expected in (("64", "64"), ("`@c.n`", "(n * 1 + 0)"),
                               ("`@c.n * 128`", "(n * 128 + 0)"),
                               ("`@c.n + 1`", "(n * 1 + 1)")):
            with self.subTest(dimension=text):
                self.assertEqual(rust_generator.dim_rust(text), expected)

    def test_existing_generators_consume_the_common_semantics(self):
        accelerator, _ = fixture(dynamic=True)
        metadata = _get_instruction_metadata(accelerator)[0]
        lowered = semantics.lower_semantics(metadata)
        self.assertEqual(rust_generator.rust_functions(metadata, lowered),
                         generate_ir2isa_rust_functions(metadata, get_backend_template_loader()))
        rule = generate_rewrite_rule(metadata, get_backend_template_loader())
        self.assertIn(lowered.match_pattern, rule)

    def test_invalid_constraints_fail_at_generation(self):
        for constraint in ("unknown == 0", "o % 0 == 0", "o % -8 == 0",
                           "__import__('os').system('true')", "o / 8 == 0",
                           "@c.o == 0", "i[0] == 0"):
            with self.subTest(constraint=constraint):
                _, instruction = fixture()
                instruction.add_constraints(constraint)
                with self.assertRaises((ValueError, SyntaxError)):
                    generate_instruction_class(instruction, get_backend_template_loader())

    def test_constraints_reach_cpp_solver(self):
        _, instruction = fixture()
        instruction.add_constraints("o % 8 == 0\no >= i + 4")
        cpp = generate_instruction_class(instruction, get_backend_template_loader())
        self.assertIn("address_modulo", cpp)
        self.assertIn("constraints.push_back", cpp)

    def test_scalar_and_multiple_results_use_supported_multiply(self):
        for outputs in (1, 2, 3):
            accelerator, _ = fixture(outputs=outputs)
            metadata = _get_instruction_metadata(accelerator)[0]
            text = generate_rewrite_rule(metadata, get_backend_template_loader())
            self.assertIn("multiply", text)
            self.assertEqual("(tuple " in text, outputs > 1)
            rust = generate_ir2isa_rust_functions(metadata, get_backend_template_loader())
            self.assertIn("precond_compute", rust)

    def test_dynamic_multiple_results(self):
        accelerator, _ = fixture(dynamic=True)
        metadata = _get_instruction_metadata(accelerator)[0]
        text = generate_rewrite_rule(metadata, get_backend_template_loader())
        self.assertIn("compute_?", text)
        rust = generate_ir2isa_rust_functions(metadata, get_backend_template_loader())
        self.assertIn("n", rust)
        self.assertNotIn("@c.", rust)

    def test_mixed_outputs_keep_return_order_across_generators(self):
        accelerator = Accelerator("MixedOutputs")
        accelerator.add_data_model("matrices", [64], [8], "bf16")
        accelerator.add_data_model("vectors", [64], [4], "f32")
        instruction = accelerator.add_instruction("mixed", [], ["a", "b", "c", "d", "sum", "product"])
        instruction.set_inputs([
            ("matrices", ["@a.a"], ["4"]), ("matrices", ["@a.b"], ["8"]),
            ("vectors", ["@a.c"], ["2"]), ("vectors", ["@a.d"], ["2"]),
        ])
        instruction.set_outputs([
            ("vectors", ["@a.sum"], ["2"]), ("matrices", ["@a.product"], ["4"]),
        ])
        instruction.add_semantics("""ENTRY mixed {
            %a = bf16[4,8] parameter(0);
            %b = bf16[8,8] parameter(1);
            %c = f32[2,4] parameter(2);
            %d = f32[2,4] parameter(3);
            %product = bf16[4,8] dot(%a, %b), lhs_contracting_dims={1}, rhs_contracting_dims={0};
            %sum = f32[2,4] add(%c, %d);
            ROOT %result = (f32[2,4], bf16[4,8]) tuple(%sum, %product);
        }""")
        metadata = _get_instruction_metadata(accelerator)[0]
        lowered = semantics.lower_semantics(metadata)
        self.assertEqual(lowered.pattern, "(tuple (add ?p2 ?p3) (dot ?p0 ?p1))")
        self.assertEqual([(node.dtype, node.shape) for node in lowered.results],
                         [("f32", ["2", "4"]), ("bf16", ["4", "8"])])
        loader = get_backend_template_loader()
        self.assertIn(lowered.match_pattern, generate_rewrite_rule(metadata, loader))
        rust = generate_ir2isa_rust_functions(metadata, loader)
        self.assertIn("joint::inputs_available", rust)
        source = generate_semantic_function(instruction)
        compile(source, "mixed-output-oracle", "exec")
        stores = [line for line in source.splitlines() if "rhs_update =" in line]
        self.assertEqual(len(stores), 2)
        self.assertIn("'sum'", stores[0])
        self.assertIn("'product'", stores[1])

    def test_shared_operand_occurrences_are_constrained_equal(self):
        for outputs in (1, 3):
            accelerator, _ = fixture(outputs=outputs)
            metadata = _get_instruction_metadata(accelerator)[0]
            text = generate_rewrite_rule(metadata, get_backend_template_loader())
            rust = generate_ir2isa_rust_functions(metadata, get_backend_template_loader())
            self.assertIn("__use", text)
            self.assertIn("egraph.find(lhs_eclasses[0]) != egraph.find(lhs_eclasses[1])", rust)


if __name__ == "__main__":
    unittest.main()
