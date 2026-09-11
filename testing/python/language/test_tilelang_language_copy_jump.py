# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Host tests using the NPU TVM fork; no device execution or precision test."""

import pytest
from tilelang import tvm
import tilelang.language as T
from tilelang.engine.phase import LowerAndLegalize, OptimizeForTarget


def copies(mod):
    result = []
    for func in mod.functions.values():
        tvm.tir.stmt_functor.post_order_visit(
            func.body,
            lambda node: (
                result.append(node)
                if isinstance(node, tvm.tir.Call)
                and isinstance(node.op, tvm.ir.Op)
                and node.op.name == "tl.copy"
                else None
            ),
        )
    return result


def kernel(width=512, parallel=False, dtype="float16"):
    @T.prim_func
    def main(
        A: T.Tensor((8, 4, 1, width), dtype),
        Params: T.Tensor((4,), "int64"),
        Out: T.Tensor((2, width), dtype),
    ):
        with T.Kernel(1, is_npu=True):
            params = T.alloc_ub((4,), "int64")
            ub = T.alloc_ub((2, width), dtype)
            T.copy(Params, params)
            if parallel:
                for i in T.Parallel(2):
                    T.copy(A[i, 0, 0, 0], ub, jump=params[2])
            else:
                # 用于 SFA 单指令双搬运，jump 在设备端由 UB 标量读取。
                T.copy(A[params[0], params[1], 0, 0], ub, jump=params[2])
            T.copy(ub, Out)

    return main


@pytest.mark.parametrize("pitch", [0, 1, 64, 320, -320, 2**32])
def test_jump_frontend_contract(pitch):
    tir = tvm.tir
    a = tir.decl_buffer((8, 4, 1, 64), "float16")
    ub = tir.decl_buffer((2, 64), "float16", scope="shared")
    call = T.copy(a[1, 2, 0, 0], ub, jump=pitch)
    assert call.args[2].value == "jump_v1"
    assert int(call.args[3]) == 384
    assert int(call.args[4]) == pitch
    assert str(call.args[3].dtype) == str(call.args[4].dtype) == "int64"
    assert [int(x) for x in call.args[0].args[2:]] == [8, 4, 1, 64]
    assert [int(x) for x in call.args[1].args[2:]] == [2, 64]
    explicit = T.copy(a[1, 2, 0, 0], ub[0, 0], size=[2, 64], jump=pitch)
    assert tvm.ir.structural_equal(call, explicit)


def test_flatten_offset_is_int64():
    tir = tvm.tir
    a = tir.decl_buffer((5_000_000, 512), "float16")
    ub = tir.decl_buffer((2, 512), "float16", scope="shared")
    call = T.copy(a[4_500_000, 0], ub, jump=512)
    assert int(call.args[3]) == 4_500_000 * 512 > 2**31


@pytest.mark.parametrize("width", [64, 512])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
def test_device_jump_survives_npu_tir_passes(width, dtype, monkeypatch):
    monkeypatch.setenv("TILELANG_ENABLE_SIMT", "0")
    monkeypatch.setenv("TILELANG_ASCEND_DEVICE_NAME", "Ascend910B")
    mod = tvm.IRModule({"main": kernel(width, dtype=dtype)})
    original = [op for op in copies(mod) if len(op.args) == 5]
    assert len(original) == 1
    target = tvm.target.Target("npuir", host="stackvm")
    with tvm.transform.PassContext(opt_level=3):
        lowered = OptimizeForTarget(LowerAndLegalize(mod, target), target)
    result = [op for op in copies(lowered) if len(op.args) == 5]
    assert len(result) == 1
    assert result[0].args[2].value == "jump_v1"
    assert str(result[0].args[4].dtype) == "int64"
    loads = []
    tvm.tir.stmt_functor.post_order_visit(
        result[0].args[4],
        lambda node: (
            loads.append(node) if isinstance(node, tvm.tir.BufferLoad) else None
        ),
    )
    assert loads and loads[0].buffer.scope() == "shared"
    assert [int(x) for x in result[0].args[0].args[2:]] == [8, 4, 1, width]
    assert [int(x) for x in result[0].args[1].args[2:]] == [2, width]


def test_jump_destination_slice():
    tir = tvm.tir
    a = tir.decl_buffer((8, 64), "float16")
    ub = tir.decl_buffer((4, 96), "float16", scope="shared")
    region = tir.BufferRegion(ub, [tvm.ir.Range(1, 3), tvm.ir.Range(16, 80)])
    call = T.copy(a[2, 0], region, jump=128)
    dst = call.args[1]
    assert [int(x) for x in dst.args[0].indices] == [1, 16]
    assert [int(x) for x in dst.args[2:]] == [2, 64]


def test_jump_rejects_parallel():
    mod = tvm.IRModule({"main": kernel(64, parallel=True)})
    with pytest.raises(ValueError, match="T.Parallel"):
        LowerAndLegalize(mod, tvm.target.Target("npuir"))


def test_jump_rejects_other_target():
    mod = tvm.IRModule({"main": kernel(64)})
    with pytest.raises(ValueError, match="target='npuir'"):
        LowerAndLegalize(mod, tvm.target.Target("c"))


def test_jump_argument_validation_and_legacy_copy():
    tir = tvm.tir
    a = tir.decl_buffer((8, 64), "float16")
    ub = tir.decl_buffer((2, 64), "float16", scope="shared")
    old_dst = tir.decl_buffer((1, 64), "float16", scope="shared")
    assert len(T.copy(a[0, 0], old_dst).args) == 2
    assert len(T.copy(a[0, 0], old_dst, coalesced_width=4).args) == 3
    with pytest.raises(ValueError, match="coalesced_width"):
        T.copy(a[0, 0], ub, coalesced_width=4, jump=64)
    with pytest.raises(TypeError, match="signed scalar"):
        T.copy(a[0, 0], ub, jump=1.5)
    with pytest.raises(TypeError, match="bool"):
        T.copy(a[0, 0], ub, jump=True)
    bad_rows = tir.decl_buffer((3, 64), "float16", scope="shared")
    with pytest.raises(ValueError, match="exactly two"):
        T.copy(a[0, 0], bad_rows, jump=64)
