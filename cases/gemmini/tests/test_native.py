"""Auditor negative controls. These are not substitutes for RTL execution."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assembly import Program, run_address_model
from emit_native import lower_calls, manual_calls
from verify import make_inputs, reference
from verify_rtl import inspect_trace, intersection, intervals


def fixture():
    meta = {"hbm": 3072, "constant": [],
            "input": [{"addr": i * 256, "shape": [16, 16], "dtype": "int8"} for i in range(4)],
            "output": [{"addr": 1024 + i * 1024, "shape": [16, 16], "dtype": "int32"} for i in range(2)]}
    program = Program("two_gemm", meta, [])
    program.calls = manual_calls(program, parallel=True)
    return program


class NativeTests(unittest.TestCase):
    def test_independent_schedule_and_all_signed_values(self):
        inputs = make_inputs(4, 901)
        expected = reference("two_gemm", inputs)
        self.assertLess(min(v.min() for v in inputs), 0)
        self.assertGreater(max(v.max() for v in inputs), 0)
        for actual, golden in zip(run_address_model(fixture(), inputs), expected):
            np.testing.assert_array_equal(actual, golden)

    def test_uninitialized_input_is_rejected(self):
        program = fixture()
        program.calls.pop(0)
        with self.assertRaisesRegex(ValueError, "Read-before-write"):
            run_address_model(program, make_inputs(4, 901))

    def test_private_memory_overflow_is_rejected(self):
        program = fixture()
        program.calls[0][1]["o0"] = 16380
        with self.assertRaisesRegex(ValueError, "Out-of-bounds"):
            run_address_model(program, make_inputs(4, 901))

    def test_aliasing_output_abi_is_rejected(self):
        program = fixture()
        program.metadata["output"][1]["addr"] = 1024
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            run_address_model(program, make_inputs(4, 901))

    def test_copying_the_same_lane_twice_fails_numerics(self):
        program = fixture()
        program.calls[-1] = ("mvout_0", deepcopy(program.calls[-1][1]))
        inputs = make_inputs(4, 901)
        actual = run_address_model(program, inputs)
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(actual[1], reference("two_gemm", inputs)[1])

    def test_lowering_does_not_group_individual_gemms(self):
        program = fixture()
        calls = manual_calls(program, parallel=False)
        lowered = "\n".join(lower_calls(calls))
        self.assertNotIn("act_double_gemm", lowered)
        self.assertEqual(lowered.count("act_gemm("), 2)

    def test_busy_alone_is_not_compute_overlap(self):
        trace = "\n".join([
            "ACT_COMMAND opcode=123 cycle=1 funct=2 rs1=0 rs2=0",
            "ACT_GEMMINI opcode=123 cycle=2 busy=1 mac00_nonzero=0",
            "ACT_GEMMINI opcode=91 cycle=3 busy=1 mac00_nonzero=0",
            "ACT_GEMMINI opcode=123 cycle=8 busy=0 mac00_nonzero=0",
            "ACT_GEMMINI opcode=91 cycle=9 busy=0 mac00_nonzero=0"])
        _, busy, mac = inspect_trace(trace)
        self.assertEqual(intersection(busy[123], busy[91]), 5)
        self.assertEqual(intersection(mac[123], mac[91]), 0)

    def test_missing_or_unfinished_trace_is_not_a_pass(self):
        with self.assertRaisesRegex(AssertionError, "Missing"):
            inspect_trace("ACT_PASS")
        with self.assertRaisesRegex(AssertionError, "active"):
            intervals([(2, 1, 1)], 1)


if __name__ == "__main__":
    unittest.main()
