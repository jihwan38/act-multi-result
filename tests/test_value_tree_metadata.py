"""The independent assembly reader preserves and validates logical ABI trees."""
import unittest

from _internal.assembly import parse


def source(extra=""):
    return ("def example(kernel, api):\n"
            f"    @kernel(hbm=2, input=[], output=[{{'addr': 0, 'shape': (), 'dtype': jnp.bfloat16}}], constant=[]{extra})\n"
            "    def body():\n        pass\n    return body\n")


class ValueTreeTests(unittest.TestCase):
    def test_old_metadata_remains_valid(self):
        _, _, metadata, _ = parse(source())
        self.assertNotIn("output_tree", metadata)

    def test_nested_and_empty_structure_is_preserved_when_reading(self):
        text = source(", input_tree=[], output_tree=[[], [0]]")
        metadata = parse(text)[2]
        self.assertEqual(metadata["input_tree"], [])
        self.assertEqual(metadata["output_tree"], [[], [0]])

    def test_rejects_bad_tree_shape_count_order_or_type(self):
        for value in ("[]", "[0,0]", "[1]", "[True]", "[-1]", "None", "'0'", "(0,)"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse(source(", output_tree=" + value))
        with self.assertRaises(ValueError):
            parse(source(", input_tree=0"))

    def test_rejects_unknown_or_duplicate_metadata(self):
        for extra in (", mystery=[]", ", hbm=2"):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                parse(source(extra))


if __name__ == "__main__":
    unittest.main()
