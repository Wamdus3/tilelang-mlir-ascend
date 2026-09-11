# SFA 单指令双搬运 jump 回归；设备测试需要完整 NPUIR 工具链。
import pytest
import random
import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.language as T
from testcommon import gen_tensor

pytestmark = [pytest.mark.op("copy"), pytest.mark.mode("Expert")]


def pair_copy(width, dtype, padded=True):
    rows, ub_pitch = (4, width + 32) if padded else (2, width)
    row_start, col_start = (1, 16) if padded else (0, 0)

    @T.prim_func
    def main(
        A: T.Tensor((8, 4, 1, width), dtype),
        Zero: T.Tensor((rows, ub_pitch), dtype),
        Out: T.Tensor((rows, ub_pitch), dtype),
        Params: T.Tensor((4,), "int64"),
    ):
        with T.Kernel(1, is_npu=True):
            ub = T.alloc_ub((rows, ub_pitch), dtype)
            params = T.alloc_ub((4,), "int64")
            T.copy(Params, params)
            p = params[0]
            lane = params[1]
            pitch = params[2]
            T.copy(Zero, ub)
            T.copy(
                A[p // 4, p % 4, 0, lane],
                ub[row_start : row_start + 2, col_start : col_start + width],
                jump=pitch,
            )
            T.copy(ub, Out)

    return main


def random_pair_copy(width, dtype):
    @T.prim_func
    def main(
        A: T.Tensor((4096 * width,), dtype),
        Indices: T.Tensor((4,), "int64"),
        Out: T.Tensor((2, width), dtype),
    ):
        with T.Kernel(1, is_npu=True):
            indices = T.alloc_ub((4,), "int64")
            ub = T.alloc_ub((2, width), dtype)
            T.copy(Indices, indices)
            # 用于 SFA 单指令双搬运：两个独立随机 GM offset 在设备端求 jump。
            T.copy(A[indices[0]], ub, jump=indices[1] - indices[0])
            T.copy(ub, Out)

    return main


@pytest.mark.parametrize("width", [64, 512])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
def test_two_random_gm_offsets(width, dtype):
    kernel = tilelang.compile(random_pair_copy(width, dtype), target="npuir")
    a = gen_tensor((4096 * width,), dtype, kind="randn")
    a_cpu = a.cpu()
    out = torch.empty((2, width), dtype=a.dtype, device=a.device)
    rng = random.Random(20260909)
    counts = [0, 0]
    for _ in range(32):
        first = rng.randrange(0, 2046) * width + rng.randrange(width)
        second = rng.randrange(2048, 4095) * width + rng.randrange(width)
        if rng.randrange(2):
            first, second = second, first
        counts[int(first > second)] += 1
        indices = torch.tensor(
            [first, second, 0, 0], dtype=torch.int64, device=a.device
        )
        kernel(a, indices, out)
        low, high = sorted((first, second))
        ref = torch.stack((a_cpu[low : low + width], a_cpu[high : high + width]))
        torch.testing.assert_close(
            out.cpu(),
            ref,
            rtol=0,
            atol=0,
            msg=f"dtype={dtype}, W={width}, offset0={first}, offset1={second}",
        )
    assert min(counts) > 0


@pytest.mark.parametrize("width", [64, 512])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
@pytest.mark.parametrize("padded", [False, True])
def test_pair_copy_dynamic_pitch(width, dtype, padded):
    kernel = tilelang.compile(pair_copy(width, dtype, padded), target="npuir")
    a = gen_tensor((8, 4, 1, width), dtype, kind="randn")
    shape = (4, width + 32) if padded else (2, width)
    row, col = (1, 16) if padded else (0, 0)
    zero = gen_tensor(shape, dtype, kind="zeros")
    out = torch.empty_like(zero)
    # Forward page crossing, contiguous, reversed, duplicate, overlapping,
    # last legal source row, and a nonzero scalar lane offset.
    cases = [
        (3, 0, 5 * width),
        (3, 0, width),
        (8, 0, -5 * width),
        (8, 0, 0),
        (8, 0, 1),
        (8, 0, -1),
        (30, 0, width),
        (3, 7, 5 * width),
    ]
    for p, lane, pitch in cases:
        params = torch.tensor([p, lane, pitch, 0], dtype=torch.int64, device=a.device)
        kernel(a, zero, out, params)
        ref = zero.cpu().clone()
        flat = a.cpu().flatten()
        start = p * width + lane
        low, high = sorted((start, start + pitch))
        ref[row, col : col + width] = flat[low : low + width]
        ref[row + 1, col : col + width] = flat[high : high + width]
        torch.testing.assert_close(
            out.cpu(),
            ref,
            rtol=0,
            atol=0,
            msg=f"dtype={dtype}, width={width}, padded={padded}, p={p}, lane={lane}, jump={pitch}",
        )


def paired_kv_rope_copy(dtype, padded_kv):
    rows, kv_pitch = (4, 544) if padded_kv else (2, 512)
    row, col = (1, 16) if padded_kv else (0, 0)

    @T.prim_func
    def main(
        KV: T.Tensor((128, 512), dtype),
        Rope: T.Tensor((128, 64), dtype),
        Indices: T.Tensor((4,), "int64"),
        Zero: T.Tensor((rows, kv_pitch), dtype),
        KVOut: T.Tensor((rows, kv_pitch), dtype),
        RopeOut: T.Tensor((2, 64), dtype),
    ):
        with T.Kernel(1, is_npu=True):
            indices = T.alloc_ub((4,), "int64")
            kv = T.alloc_ub((rows, kv_pitch), dtype)
            rope = T.alloc_ub((2, 64), dtype)
            T.copy(Indices, indices)
            # Read the shared token pair once, before subsequent UB DMA writes.
            # CANN 9 can misread repeated UB scalar loads across these copies.
            index0 = indices[0]
            index1 = indices[1]
            T.copy(Zero, kv)
            # 用于 SFA 单指令双搬运：共享 token indices，分别按 KV/RoPE
            # 的元素行距求 jump。KV 回退时也须与 RoPE 的升序输出对应。
            T.copy(
                KV[index0, 0],
                kv[row : row + 2, col : col + 512],
                jump=(index1 - index0) * T.int64(512),
            )
            T.copy(
                Rope[index0, 0],
                rope,
                jump=(index1 - index0) * T.int64(64),
            )
            T.copy(kv, KVOut)
            T.copy(rope, RopeOut)

    return main


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
@pytest.mark.parametrize("padded_kv", [False, True])
def test_paired_kv_rope_order(dtype, padded_kv):
    kernel = tilelang.compile(paired_kv_rope_copy(dtype, padded_kv), target="npuir")
    kv = gen_tensor((128, 512), dtype, kind="randn")
    rope = gen_tensor((128, 64), dtype, kind="randn")
    shape = (4, 544) if padded_kv else (2, 512)
    row, col = (1, 16) if padded_kv else (0, 0)
    zero = gen_tensor(shape, dtype, kind="zeros")
    kv_out, rope_out = torch.empty_like(zero), torch.empty_like(rope[:2])
    kv_cpu, rope_cpu = kv.cpu(), rope.cpu()
    qkv, qr = torch.arange(512).double() / 512, torch.arange(64).double() / 64

    def attention(k, r):
        # A two-token CPU attention check, not a full SFA kernel benchmark.
        k, r = k.double(), r.double()
        weights = torch.softmax((k @ qkv + r @ qr) / (576**0.5), dim=0)
        return (weights[:, None] * k[:, :8]).sum(dim=0)

    rng = random.Random(20260909)
    pairs = [(100, 3), (3, 100), (127, 0), (0, 127), (7, 7), (65, 64)]
    pairs += [(rng.randrange(128), rng.randrange(128)) for _ in range(8)]
    for first, second in pairs:
        indices = torch.tensor(
            [first, second, 0, 0], dtype=torch.int64, device=kv.device
        )
        kernel(kv, rope, indices, zero, kv_out, rope_out)
        ordered = sorted((first, second))
        expected = zero.cpu().clone()
        expected[row : row + 2, col : col + 512] = kv_cpu[ordered]
        actual_kv, actual_rope = kv_out.cpu(), rope_out.cpu()
        torch.testing.assert_close(
            actual_kv,
            expected,
            rtol=0,
            atol=0,
            msg=f"KV dtype={dtype}, padded={padded_kv}, indices=({first}, {second})",
        )
        torch.testing.assert_close(
            actual_rope,
            rope_cpu[ordered],
            rtol=0,
            atol=0,
            msg=f"RoPE dtype={dtype}, padded={padded_kv}, indices=({first}, {second})",
        )
        torch.testing.assert_close(
            attention(actual_kv[row : row + 2, col : col + 512], actual_rope),
            attention(kv_cpu[[first, second]], rope_cpu[[first, second]]),
            rtol=1e-12,
            atol=1e-12,
        )


def test_jump_frontend_contract():
    from tilelang import tvm

    tir = tvm.tir
    a = tir.decl_buffer((8, 4, 1, 64), "float16")
    ub = tir.decl_buffer((2, 64), "float16", scope="shared")
    call = T.copy(a[1, 2, 0, 0], ub, jump=320)
    assert len(call.args) == 5
    assert call.args[2].value == "jump_v1"
    assert int(call.args[3]) == 6 * 64
    assert int(call.args[4]) == 320
    assert [int(x) for x in call.args[0].args[2:]] == [8, 4, 1, 64]
    assert [int(x) for x in call.args[1].args[2:]] == [2, 64]
    explicit = T.copy(a[1, 2, 0, 0], ub[0, 0], size=[2, 64], jump=320)
    assert tvm.ir.structural_equal(call, explicit)
    old_dst = tir.decl_buffer((1, 64), "float16", scope="shared")
    assert len(T.copy(a[0, 0, 0, 0], old_dst).args) == 2
    with pytest.raises(ValueError, match="coalesced_width"):
        T.copy(a[0, 0, 0, 0], ub, coalesced_width=4, jump=64)
    bad_rows = tir.decl_buffer((3, 64), "float16", scope="shared")
    with pytest.raises(ValueError, match="exactly two"):
        T.copy(a[0, 0, 0, 0], bad_rows, jump=64)
    with pytest.raises(TypeError, match="signed scalar"):
        T.copy(a[0, 0, 0, 0], ub, jump=1.5)
