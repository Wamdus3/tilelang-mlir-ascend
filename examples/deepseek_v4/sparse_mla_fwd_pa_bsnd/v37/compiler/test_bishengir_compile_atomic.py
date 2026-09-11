"""The compatibility driver must preserve explicit atomic operations."""

import unittest
from unittest.mock import patch

from bishengir_compile_atomic import prepare_ir
from sfa_ub_layout import UB_LAYOUT


class CompilerAdapterTests(unittest.TestCase):
    def test_atomic_operations_are_forwarded_to_ub_adapter_unchanged(self):
        source = """func.func @arbitrary_name() {
  hivm.hir.pipe_barrier[<PIPE_MTE3>]
  hivm.hir.set_atomic kind = <add>[type = i32]
  hivm.hir.store ins(%a : memref<32xi32>) outs(%b : memref<32xi32>)
  hivm.hir.pipe_barrier[<PIPE_MTE3>]
  hivm.hir.set_atomic kind = <none>[type = i32]
}
"""
        with patch(
            "bishengir_compile_atomic.place", return_value=(source, {})
        ) as place:
            self.assertEqual(prepare_ir(source), (source, {}))
            place.assert_called_once_with(source, **UB_LAYOUT)

    def test_only_incompatible_sync_attribute_is_removed(self):
        source = "op syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>\n"
        with patch(
            "bishengir_compile_atomic.place", return_value=("op\n", {})
        ) as place:
            prepare_ir(source)
            place.assert_called_once_with("op\n", **UB_LAYOUT)


if __name__ == "__main__":
    unittest.main()
