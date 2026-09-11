"""Host-only checks for the explicit atomic-mode API."""

import pytest
import tilelang.language as T


@pytest.mark.parametrize("kind", ["add", "max", "min", "none"])
@pytest.mark.parametrize(
    "dtype", ["float16", "float32", "bfloat16", "int8", "int16", "int32"]
)
def test_atomic_mode_intrinsic(kind, dtype):
    op = T.set_atomic(kind, dtype)
    assert op.op.name == "tl.npuir_set_atomic"
    assert [value.value for value in op.args] == [kind, dtype]


@pytest.mark.parametrize(
    "kind,dtype",
    [("xor", "int32"), ("add", "uint32"), ("add", "float64"), ("none", "bool")],
)
def test_invalid_atomic_mode(kind, dtype):
    with pytest.raises(ValueError):
        T.set_atomic(kind, dtype)


def test_atomic_mode_helpers():
    assert T.set_atomic_add("int32").args[0].value == "add"
    assert T.set_atomic_none("int32").args[0].value == "none"
