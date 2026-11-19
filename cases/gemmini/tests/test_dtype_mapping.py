"""Guard generated Rust names against the actual Backend enum, not a mock."""

from pathlib import Path
import re
import unittest

import act_backend
from act_backend.isel_ir2isa_rewrites_rs_generator import map_dtype_to_rust


class DtypeMappingTest(unittest.TestCase):
    def test_actual_enum(self):
        source = Path(act_backend.__file__).parent / "generic/src/ir/dtype.rs"
        text = source.read_text()
        conversions = dict(re.findall(r'"(\w+)" => Dtype::(\w+)', text))
        for hlo in ("s8", "s32", "u8", "u32", "f32", "bf16"):
            with self.subTest(hlo=hlo):
                self.assertEqual(map_dtype_to_rust(hlo), f"Dtype::{conversions[hlo]}")

    def test_unsupported_is_rejected(self):
        with self.assertRaises(ValueError):
            map_dtype_to_rust("invalid")


if __name__ == "__main__":
    unittest.main()
