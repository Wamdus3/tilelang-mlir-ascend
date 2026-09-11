"""Host-only interface tests with layouts unrelated to the SFA allocation plan."""

import unittest

from ub_pool_adapter import place


SOURCE = """module {
  func.func @example() {
    %a = memref.alloc() : memref<3x4xf32, strided<[4, 1]>, #hivm.address_space<ub>>
    return
  }
}
"""
KEY = ((3, 4), "f32")


class PlacementInterfaceTests(unittest.TestCase):
    def test_independent_layouts(self):
        for pool, capacity, offset, alignment in (
            ("scratch", 64, 0, 32),
            ("temporary", 128, 16, 16),
        ):
            with self.subTest(pool=pool):
                output, receipt = place(
                    SOURCE,
                    pools={pool: capacity},
                    layout={KEY: ((pool, offset),)},
                    alignment=alignment,
                )
                self.assertEqual(receipt["physical_bytes"], capacity)
                self.assertEqual(receipt["placements"][0]["offset"], offset)
                self.assertEqual(receipt["placements"][0]["bytes"], 48)
                self.assertIn(f"%ubpool_{pool} = memref.alloc()", output)
                self.assertIn(f"arith.constant {offset} : index", output)

    def test_bounded_wildcard(self):
        key = ((None, 4), "f32")
        config = dict(
            pools={"scratch": 64},
            layout={key: (("scratch", 0),)},
            dimension_limits={key: (3, 4)},
        )
        _, receipt = place(SOURCE, **config)
        self.assertEqual(receipt["placements"][0]["shape"], (3, 4))
        with self.assertRaisesRegex(ValueError, "dimension out of bounds"):
            place(SOURCE.replace("3x4xf32", "4x4xf32"), **config)

    def test_optional_entry(self):
        optional = ((8,), "i32")
        config = dict(
            pools={"scratch": 128},
            layout={KEY: (("scratch", 0),), optional: (("scratch", 64),)},
        )
        with self.assertRaisesRegex(ValueError, "allocation count"):
            place(SOURCE, **config)
        place(SOURCE, optional_allocations={optional}, **config)

    def test_ambiguous_wildcards(self):
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            place(
                SOURCE,
                pools={"scratch": 64},
                layout={
                    ((None, 4), "f32"): (("scratch", 0),),
                    ((3, None), "f32"): (("scratch", 0),),
                },
            )

    def test_exact_key_precedence(self):
        wildcard = ((None, 4), "f32")
        _, receipt = place(
            SOURCE,
            pools={"scratch": 128},
            layout={KEY: (("scratch", 64),), wildcard: (("scratch", 0),)},
            optional_allocations={wildcard},
        )
        self.assertEqual(receipt["placements"][0]["offset"], 64)

    def test_invalid_configuration(self):
        for extra in (
            {"alignment": 0},
            {"dimension_limits": {KEY: (3,)}},
            {"optional_allocations": {((1,), "i32")}},
            {"layout": {KEY: (("missing_pool", 0),)}},
        ):
            config = dict(pools={"scratch": 64}, layout={KEY: (("scratch", 0),)})
            config.update(extra)
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                place(SOURCE, **config)


if __name__ == "__main__":
    unittest.main()
