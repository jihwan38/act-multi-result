"""Verification must catch bad values and preserve its source artifacts."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from verify import arguments, compare_outputs, example_defaults, input_sets

ROOT = Path(__file__).resolve().parents[1]


class VerificationTests(unittest.TestCase):
    def test_comparison_rejects_wrong_shape_dtype_value_and_nonfinite_results(self):
        expected = np.array([1, 2], dtype=np.float32)
        self.assertTrue(compare_outputs("same", [expected], [expected])["outputs"][0]["exact_bytes"])
        for actual in (expected.reshape(1, 2), expected.astype(np.float64), expected + 1):
            with self.subTest(actual=actual):
                output = compare_outputs("bad", [actual], [expected])["outputs"][0]
                self.assertFalse(output["exact_bytes"])
                self.assertEqual(output["actual_shape"], list(actual.shape))
                self.assertEqual(output["expected_dtype"], "float32")
        for value in (np.inf, np.nan):
            output = compare_outputs("nonfinite", [np.array([value])], [np.array([value])])["outputs"][0]
            self.assertFalse(output["finite"])
        with self.assertRaisesRegex(ValueError, "output count"):
            compare_outputs("missing", [], [expected])

    def test_integer_inputs_are_reproducible_and_not_all_zero(self):
        metadata = [{"shape": [64], "dtype": "jnp.int32"},
                    {"shape": [64], "dtype": "jnp.uint8"}]
        first, second = list(input_sets(metadata)), list(input_sets(metadata))
        self.assertEqual([label for label, _ in first], ["seed0", "seed1", "seed2", "zero"])
        for (label, values), (_, repeated) in zip(first, second):
            for value, repeat in zip(values, repeated):
                np.testing.assert_array_equal(value, repeat)
                self.assertEqual(bool(np.any(value)), label != "zero")

    def test_both_multi_result_examples_infer_an_allocation_audit(self):
        for case, filename in (("qkv", "QKV.py"), ("double_gemm", "QKVDoubleGemm.py")):
            oracle, isa = example_defaults(ROOT / "build/multi-result/runs" / case / "example/assembly.py")
            self.assertIsNotNone(oracle)
            self.assertEqual(isa, ROOT / "cases" / case / filename)
        self.assertIsNone(example_defaults(ROOT / "build/baseline/runs/qkv/example/assembly.py")[1])

    def test_report_cannot_replace_an_inferred_pii_or_allocation_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            assembly = directory / "assembly.py"
            assembly.write_text("# PII number: 3\n")
            pii = directory / "compiler/3.pii"
            pii.parent.mkdir()
            pii.write_text("selected PII\n")
            allocation = directory / "assembly.py.allocation.json"
            allocation.write_text('{"tensors": {}}\n')
            for report in (pii, allocation):
                before = report.read_bytes()
                with self.assertRaisesRegex(ValueError, "must not overwrite"):
                    arguments(["--hlo", str(directory / "input.hlo"), "--assembly", str(assembly),
                               "--oracle", str(directory / "oracle"), "--isa", str(directory / "ISA.py"),
                               "--report", str(report)])
                self.assertEqual(report.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
