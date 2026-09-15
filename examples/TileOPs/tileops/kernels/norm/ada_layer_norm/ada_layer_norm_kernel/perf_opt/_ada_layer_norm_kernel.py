# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""AdaLayerNorm forward kernel, has_gate=False path (NPU, Developer mode).

Stage 4 TUNED version (perf_opt, 2026-09-10; baseline: the Stage 3
precision-passed ``../_ada_layer_norm_kernel.py``). Two changes over the
baseline, both precision-neutral (identical math, identical
vsqrt+vdiv rstd contract, identical tail semantics):

1. block_m defaults: the deployed wrapper default was the GPU
   structural ``block_m=1``; this module replaces it with a tuned
   per-(M, N) dispatch table (``TUNED_DEFAULT_BLOCK_M``) plus the
   UB-budget general rule (``_select_block_m``). See the table's
   comment for the per-shape evidence.

2. MTE2/VEC decoupling via per-input staging + up-front loads (the
   round-2 structural win). Round-1 diagnosis: at every workload the
   per-core pipe busy times summed to ~the wall time (prefill fp16
   bm=1: vec 38.8 + scalar 13.2 + mte2 42.4 + mte3 6.5 ~= 101us vs
   90.5us wall) -- MTE2 and Vector pipes executed nearly serialized.
   Root cause: the single ``stage`` buffer was reused x -> scale ->
   shift, forcing MTE2(x) -> VEC(vcast x) -> MTE2(scale) -> VEC ->
   MTE2(shift) with no cross-pipe overlap. The same-traffic-shape
   lerp_tensor precedent (pattern-library PL-1.6: 3 loads up-front,
   vector chain fully hidden in the MTE2 window) uses dedicated
   per-input staging. This module restructures both dtype paths:
     - fp16/bf16 transit: ``stage_x`` (x-in / y-out dual role) +
       ``stage_s``/``stage_h`` dedicated scale/shift staging, all
       three loads hoisted to the top of the serial body, then one
       uninterrupted fp32 vector chain, then a single store. UB: 14
       B/elem resident before auto-multi-buffer inflation (measured
       20.01 B/elem multi-buffered -- probe_ub_v2.py -- so the full
       baseline bm range survives: N=1152 -> bm<=8, N=4096 -> bm<=2).
     - fp32 direct: ``x_f32`` (x -> d -> result) + ``sq_f32`` (d^2)
       + ``scale_f32``/``shift_f32`` dedicated buffers, all three
       loads up-front. UB: 16 B/elem resident, 26 B/elem
       multi-buffered (N=1152 -> bm<=6, N=4096 -> bm<=1).
   Measured (msprof op Task Duration, median of 20): prefill fp16
   58.3 -> 39.8us at bm=2 (-31.7% vs the bm-swept baseline), dit fp16
   9.55 -> 8.76us at bm=7, decode bf16 3.16 -> 2.65us.

Rejected during tuning (see perf_opt/opt_log.md): the static-extent
body specialization (v3_op2: prefill +1.5-3.3%, dit worse than the
bm-7 reshape alone), the num_kernels=32 balance cap (v3_op3: UB
requirement grows to 22 B/elem at num_local_tasks=4 -- compile
failure at the target config), and block_m beyond the transit UB cap
(compile overflow is the natural guard).

Migrated from the GPU repo's ``_ada_layer_norm_kernel`` (TileOPs
AdaLayerNormFwdOp, non-gated variant) to TileLang ``target="npuir"``.

Semantics (DESIGN.md section 0.1, frozen):
    y_ij = scale_ij * (x_ij - mean_i) * rstd_i + shift_ij
    mean_i = sum_j(x_ij) / N          (biased, ddof=0)
    rstd_i = rsqrt(sum_j(x_ij - mean_i)^2 / N + eps)
    fp32 opmath throughout, single round-back to the input dtype.

NPU redesign (DESIGN.md section 0.6):
    R1: persistent core split: ``num_kernels = min(num_logical,
        vector_cores)`` with an in-core ``T.serial`` grid-stride loop
        and an if-guard for out-of-range row blocks (static bounds).
    R2: block_m driven by the UB budget table (DESIGN.md section 4.5),
        not the GPU structural block_m=1.
    R3: tail handling via ``T.min`` + src/dst dual explicit slices +
        garbage-row drop (rows are independent; reduce dim=1 and the
        row-broadcast v-ops never mix rows; output slices copy valid
        rows only).
    R4: explicit fp32 transit chain for fp16/bf16:
        ``T.vcast(rint)`` up to fp32 -> fp32 v-prefix chain ->
        ``T.vcast(rint)`` single round-back (bf16 has no Vector
        arithmetic; fp16 transit matches the golden's fp32 opmath).

Dropped GPU mechanisms (DESIGN.md section 0.5, intent handed over):
    256-alignment padding + pad variance correction (replaced by exact
    N + identity O4), cp.async prefetch (compiler auto multi-buffer),
    threads/block two-level parallelism (Vector lanes).

Implementation Notes (attempt-1 precision fix, 2026-09-10):
    Deviation from DESIGN.md section 3.2 (rstd row): the raw
    ``T.vrsqrt`` instruction on this toolchain (tilelang
    0.1.2+a83118285a, Ascend910B2C) is an approximate vector op with a
    measured max relative error of 2.7e-3 across var in [9e-7, 8.7e5]
    (layer_norm.py precedent passes only because its tolerance is
    1e-2). That error propagates to y as ~2.7e-3 * |d*rstd*scale| and
    violates the fp32 1e-5 / fp16 1e-3 gates (~87% / ~6% of elements;
    bf16 1.6e-2 masked it). Fix, preserving the DESIGN.md section 3.1
    math exactly (rstd = 1/sqrt(var + eps)): compose rstd from
    ``T.vsqrt`` + ``T.vdiv`` on the (block_m, 1) buffer --
    sqrt(var)/var == 1/sqrt(var) in real arithmetic, and both ops are
    correctly rounded on this device (measured max rel err 5.8e-8 /
    7.9e-8; the dead s2 buffer serves as the sqrt scratch, no
    in-place aliasing). Net cost: one extra (block_m, 1) vector op.
    Evidence: repro/TRAP-vrsqrt-plain-precision.py (self-contained,
    with assertions).

Implementation Notes (attempt-1 UB budget correction, 2026-09-10):
    DESIGN.md section 4.5 budgeted the UB resident block at 192 KB /
    1.7 (~115 KB, i.e. ~11500 elements at 10 B/elem). Compile probes
    on this toolchain show the auto-multi-buffer pass materializes
    ~20 B per resident element whenever the persistent serial loop is
    active (num_local_tasks >= 2), on BOTH dtype paths: fp32 bm=3
    N=4096 overflowed at exactly 245760 B = 20*3*4096 and fp32 bm=4
    N=3000 at exactly 240000 B = 20*4*3000; num_local_tasks == 1 does
    not inflate. The effective factors are therefore 2.0x (fp16/bf16
    transit, 3 buffers) and 2.5x (fp32 direct, 2 buffers), not 1.7x.
    The guard and the default table now use the measured 20 B/elem
    budget with a 9216-element cap (= the design's largest sanctioned
    config bm=8 x N=1152, >= 12 KB margin); per the DESIGN.md section
    9.1 contingency, fp32 N=4096 drops one table step (bm 3 -> 2).

Factory contract (source interface unchanged):
    _ada_layer_norm_kernel(M, N, eps, dtype, has_gate=False,
                           use_cp_async=False)
      -> _func(block_m=UB-table default) -> main
    ``main(x, scale, shift, _dummy)`` returns ``y`` (``out_idx=[4]``,
    source contract: _dummy keeps the output at index 4 so the gated
    and non-gated variants share the signature).
    ``use_cp_async`` is accepted but ignored (CUDA cp.async does not
    exist on npuir; the single slice path covers aligned/unaligned N).
    ``has_gate=True`` (AdaLN-Zero) is out of scope -> NotImplementedError.

Run the embedded hierarchical tests:
    python _ada_layer_norm_kernel.py --level L0
    python _ada_layer_norm_kernel.py --level all
"""

import os

# Developer mode: alloc_shared maps to UB, compiler auto-sync
# (DESIGN.md sections 2 and 7).
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse
import time

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401  (registers the "npu" device)
from tilelang.utils.npu_utils import NPUUtils

__all__ = ["ALIGNMENT", "_ada_layer_norm_kernel", "_align_up", "golden_ada_layer_norm"]


# ---------------------------------------------------------------------------
# GPU padding-policy helpers (integration glue, Stage 5)
#
# Re-exported verbatim from the GPU extracted module
# (``_ada_layer_norm_fwd_kernels.py``, extract_tl_kernel.py pattern B) for
# the wrapper's extracted-import contract: ``ada_layer_norm.py`` (and the
# pytest policy tests) import ``ALIGNMENT`` / ``_align_up`` alongside the
# kernel factory and drive ``_should_use_cp_async`` through them. The NPU
# kernel itself drops the 256-alignment padding (DESIGN.md section 0.5,
# exact-N redesign); these helpers only feed the wrapper's ignored
# ``use_cp_async`` flag and the config selector, so re-adding them here
# changes no kernel semantics.
# ---------------------------------------------------------------------------

ALIGNMENT = 256


def _align_up(n: int, alignment: int) -> int:
    return ((n + alignment - 1) // alignment) * alignment


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Supported dtypes (factory validation).
_SUPPORTED_DTYPES = ("float16", "bfloat16", "float32")

# UB capacity per Vector core (docs/开发指南.md: 192 KB).
_UB_CAPACITY_BYTES = 196_608

# Measured multi-buffered UB cost per element of the resident block.
# Baseline structure (attempt-1 finding, see module Implementation Notes
# in the Stage 3 baseline): serial loop active (num_local_tasks >= 2)
# -> auto-multi-buffer materializes ~20 B/elem (2.0x of the 10 B/elem
# transit set; 2.5x of the 8 B/elem fp32 set).
# v2_op1 restructured sets (see module docstring): transit 14 B/elem
# resident (stage_x 2 + stage_s 2 + stage_h 2 + x_f32 4 + sq_f32 4),
# fp32 16 B/elem (x_f32 4 + sq_f32 4 + scale_f32 4 + shift_f32 4).
# Inflation factors probe-measured on this structure (probe_ub_v2.py,
# 2026-09-10, compile overflow "requires N bits" readings):
#   transit: exactly 20.01 B/elem multi-buffered (bm=8 N=1152 = 184.4 KB
#     compiles; bm=9 N=1152 overflows at 207424 B) -- the SAME absolute
#     footprint as the 10 B/elem baseline set, so the full baseline bm
#     range survives the restructure (N=1152 -> bm=8, N=4096 -> bm=2).
#   fp32 direct: exactly 26.00 B/elem (bm=2 N=4096 overflows at 213024 B;
#     bm=5 N=1152 = 149.8 KB compiles) -> N=1152 caps at bm=6, N=4096 at
#     bm=1.
# Constants carry a small margin over the measured 20.01 / 26.00.
_UB_BYTES_PER_ELEM_TRANSIT_V2 = 21
_UB_BYTES_PER_ELEM_FP32_V2 = 27
_UB_CAPACITY_ELEMS = _UB_CAPACITY_BYTES // max(
    _UB_BYTES_PER_ELEM_TRANSIT_V2, _UB_BYTES_PER_ELEM_FP32_V2
)  # conservative shared cap; refined per-dtype in the factory guard

# block_m default cap (DESIGN.md section 4.5).
_BLOCK_M_CAP = 8

# Precision tolerances (DESIGN.md section 8.2, identical to the source
# repo test_ada_layer_norm.py ``_get_tolerances()``).
_TOLERANCE = {
    "float16": (torch.float16, 1e-3, 1e-3),
    "bfloat16": (torch.bfloat16, 1.6e-2, 1.6e-2),
    "float32": (torch.float32, 1e-5, 1e-5),
}


# ---------------------------------------------------------------------------
# Golden (PyTorch CPU reference implementation)
# ---------------------------------------------------------------------------
def golden_ada_layer_norm(x, scale, shift, eps=1e-5):
    """PyTorch reference: ``y = scale * LayerNorm(x) + shift``.

    Port of the source-repo test benchmark (examples/TileOPs/tests/ops/
    test_ada_layer_norm.py ``AdaLayerNormTest.ref_program``). LayerNorm
    over the last dim with no weight/bias, fp32 opmath
    (``F.layer_norm(x.float(), ...)``), then the affine modulation in
    fp32, and a single round-back to the input dtype. Independent of
    the NPU algorithm (no vcast / fp32-transit / persistent structure).
    Inputs: x/scale/shift same shape (..., N), same dtype; output same
    shape and dtype. Runs on CPU and does not require an NPU device.
    """
    N = x.shape[-1]
    normed = F.layer_norm(x.float(), (N,), weight=None, bias=None, eps=eps)
    y = scale.float() * normed + shift.float()
    return y.to(x.dtype)


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Tuned per-(M, N) block_m table (Stage 4, msprof op Task Duration,
# Ascend910B2C / tilelang 0.1.2+a83118285a / CANN 8.5.0, 2026-09-10).
#
# Empirical winners over the per-shape block_m sweeps (see
# perf_opt/opt_log.md): these override the general UB-table rule for
# the manifest workloads. Exposed at module level for the wrapper's
# baseline/perf_opt switch block to reference in pairs (S4-5 dispatch
# table pattern).
#
#   (64, 1152)   -> 4  : 16 tasks / 16 cores (nl=1) beats bm=8 (8
#                        cores, +13-20%) and bm=2 (32 cores, +7-9%);
#                        U-shaped task-concurrency sweet spot.
#   (1024, 1152) -> 7  : task-geometry local optimum (147 tasks, nl=4,
#                        multi-buffered UB 82%) beats the UB-table bm=8
#                        (128 tasks, nl=3, UB 94%): -8.3% fp16 (single
#                        run), -6.3% bf16 (A/B merged, direction
#                        consistent). NOT a general UB-slack law: at
#                        (4096, 1152) bm=8 wins (+4.0% over bm=7) --
#                        hence the explicit table entry, not a rule.
#   (2048, 4096) -> 2  : the transit UB cap (bm=3 needs 245.8 KB >
#                        192 KB); bm=1 measures +20% (copy granularity).
#   (1, 4096)    -> 1  : single row; bm=2 adds garbage-row vector work
#                        (+8.9%).
TUNED_DEFAULT_BLOCK_M = {
    (64, 1152): 4,
    (1024, 1152): 7,
    (2048, 4096): 2,
    (1, 4096): 1,
}


def _select_block_m(M, N, use_fp32_transit=True):
    """Tuned block_m: manifest-shape dispatch table first (only when the
    entry fits the dtype's multi-buffered UB budget -- the table was
    tuned on the transit path and e.g. (2048, 4096) fp32 caps at bm=1),
    then the general UB-budget rule."""
    tuned = TUNED_DEFAULT_BLOCK_M.get((M, N))
    if tuned is not None:
        bpe = _UB_BYTES_PER_ELEM_TRANSIT_V2 if use_fp32_transit else _UB_BYTES_PER_ELEM_FP32_V2
        if tuned * N * bpe <= _UB_CAPACITY_BYTES:
            return tuned
    return _default_block_m(N, use_fp32_transit)


def select_row_config(M, N, use_fp32_transit=True):
    """Drop-in tuned config for the wrapper's row-config selection
    (same return shape as ada_layer_norm.py ``_select_row_config``)."""
    return {"block_m": _select_block_m(M, N, use_fp32_transit)}


def _default_block_m(N, use_fp32_transit=True):
    """UB-budget default block_m for the v2_op1 restructured buffer set
    (per-dtype bytes/elem, see Constants)."""
    bpe = _UB_BYTES_PER_ELEM_TRANSIT_V2 if use_fp32_transit else _UB_BYTES_PER_ELEM_FP32_V2
    return min(_BLOCK_M_CAP, max(1, _UB_CAPACITY_BYTES // (bpe * N)))


def _ada_layer_norm_kernel(M, N, eps, dtype, has_gate=False, use_cp_async=False):
    """Build the AdaLayerNorm forward kernel factory (source signature).

    Returns a JIT callable taking a single ``block_m`` (threads removed,
    DESIGN.md section 3.3 note), which compiles and returns ``main``;
    ``main(x, scale, shift, _dummy)`` returns ``y`` (``out_idx=[4]``,
    source contract). ``use_cp_async`` is kept for interface
    compatibility and ignored (CUDA cp.async is NPU-absent).
    """
    if has_gate:
        raise NotImplementedError("has_gate=True (AdaLN-Zero) is out of scope for this migration")
    if M < 1 or N < 1:
        raise ValueError(f"M and N must be positive, got M={M}, N={N}")
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"unsupported dtype {dtype!r}; expected one of {sorted(_SUPPORTED_DTYPES)}"
        )

    # O1 (DESIGN.md section 1.6.1): compile-time reciprocal, used as the
    # scalar src2 of T.vmul (division -> multiplication by fl(1/N)).
    N_inv = 1.0 / float(N)

    # R4: fp32 transit for both half-precision dtypes (bf16 has no
    # Vector arithmetic; fp16 transit matches the golden's fp32 opmath
    # rounding path, TRAP-fp16-opmath-golden).
    use_fp32_transit = dtype in ("float16", "bfloat16")
    block_m_default = _select_block_m(M, N, use_fp32_transit)

    # Physical Vector cores (DESIGN.md section 5.5: AI cores x 2 for a
    # pure-Vector op; measured 24 -> 48 on this device). Re-queried per
    # factory call so deployments on other devices adapt automatically.
    vector_cores = NPUUtils.get().get_aicore_num() * 2

    @tilelang.jit(out_idx=[4], target="npuir")
    def _func(block_m=block_m_default):
        # Host-side UB budget guard (v2_op1 restructured buffer set;
        # per-dtype bytes/elem, see Constants). ValueError (not a bare
        # assert) so the guard survives `python -O`.
        bpe = _UB_BYTES_PER_ELEM_TRANSIT_V2 if use_fp32_transit else _UB_BYTES_PER_ELEM_FP32_V2
        if block_m < 1:
            raise ValueError(f"block_m must be positive, got block_m={block_m}")
        if block_m * N * bpe > _UB_CAPACITY_BYTES:
            raise ValueError(
                f"block_m={block_m} exceeds the UB budget for N={N} "
                f"(max {block_m} x {N} x {bpe} B/elem multi-buffered > "
                f"{_UB_CAPACITY_BYTES} B UB capacity)"
            )

        # Persistent core split (DESIGN.md section 5.5). All quantities
        # are compile-time Python constants (fold to IntImm at trace
        # time, PL-1.5); the in-core serial bound is therefore static.
        num_logical = (M + block_m - 1) // block_m
        num_kernels = min(num_logical, vector_cores)
        num_local_tasks = (num_logical + num_kernels - 1) // num_kernels

        # Compile-time dtype dispatch outside the traced body (avoids the
        # TVM-script parser if-block variable-table scoping limitation,
        # pattern-library C11; lerp same form).
        if use_fp32_transit:

            @T.prim_func
            def main(
                x: T.Tensor[(M, N), dtype],
                scale: T.Tensor[(M, N), dtype],
                shift: T.Tensor[(M, N), dtype],
                # _dummy keeps the output tensor at index 4 so that
                # out_idx=[4] is consistent between the non-gated and
                # gated variants (source contract). It also guarantees
                # the kernel always has >= 1 input tensor.
                _dummy: T.Tensor[(1,), dtype],
                y: T.Tensor[(M, N), dtype],
            ):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    # UB buffers (v2_op1 restructure): dedicated per-input
                    # staging so the three GM loads can stream up-front
                    # (MTE2) before the uninterrupted fp32 vector chain --
                    # breaking the baseline's stage-reuse serialization
                    # MTE2(x) -> VEC -> MTE2(scale) -> VEC -> MTE2(shift).
                    #   stage_x: x in (consumed by the first vcast) -> y
                    #            out staging (dual role, disjoint lives)
                    #   stage_s/stage_h: scale/shift staging
                    #   x_f32  : x fp32 -> d (kept, O3) -> final result
                    #   sq_f32 : d^2 -> scale fp32 -> shift fp32 (3 roles)
                    stage_x = T.alloc_shared((block_m, N), dtype)
                    stage_s = T.alloc_shared((block_m, N), dtype)
                    stage_h = T.alloc_shared((block_m, N), dtype)
                    x_f32 = T.alloc_shared((block_m, N), "float32")
                    sq_f32 = T.alloc_shared((block_m, N), "float32")
                    s1 = T.alloc_shared((block_m, 1), "float32")
                    mean = T.alloc_shared((block_m, 1), "float32")
                    s2 = T.alloc_shared((block_m, 1), "float32")
                    var = T.alloc_shared((block_m, 1), "float32")
                    rstd = T.alloc_shared((block_m, 1), "float32")

                    # Persistent in-core serial task loop (static bound,
                    # grid-stride mapping, if-guard masks out-of-range
                    # row blocks).
                    for i in T.serial(num_local_tasks):
                        block_id = i * num_kernels + cid
                        if block_id < num_logical:
                            off_m = block_id * block_m
                            # Tail row count. The v-ops below run on the
                            # full (block_m, N) buffers; garbage tail rows
                            # are row-isolated (reduce dim=1 / row
                            # broadcast) and never copied out.
                            real_m = T.min(block_m, M - off_m)

                            # --- 1. All three loads up-front (MTE2) ---
                            # Dual explicit slices, dst zero-origin
                            # (TRAP-T-copy-region-semantics /
                            # TRAP-UB-dst-align recommended form).
                            T.copy(
                                x[off_m : off_m + real_m, 0:N],
                                stage_x[0:real_m, 0:N],
                            )
                            T.copy(
                                scale[off_m : off_m + real_m, 0:N],
                                stage_s[0:real_m, 0:N],
                            )
                            T.copy(
                                shift[off_m : off_m + real_m, 0:N],
                                stage_h[0:real_m, 0:N],
                            )

                            # --- 2. Uninterrupted fp32 vector chain ---
                            T.vcast(stage_x, x_f32, round_mode="rint")
                            # Mean (O1: S1 * fl(1/N))
                            T.reduce_sum(x_f32, s1, dim=1)
                            T.vmul(s1, N_inv, mean)
                            # Centered two-pass variance; d kept (O3).
                            T.vsub(x_f32, mean, x_f32)  # d = x - mean
                            T.vmul(x_f32, x_f32, sq_f32)  # sq = d^2
                            T.reduce_sum(sq_f32, s2, dim=1)
                            T.vmul(s2, N_inv, var)
                            T.vadd(var, eps, var)
                            # rstd = 1/sqrt(var): vsqrt+vdiv composition
                            # (the raw T.vrsqrt instruction carries
                            # ~2.7e-3 relative error on this toolchain;
                            # see module Implementation Notes).
                            T.vsqrt(var, s2)  # s2 reused: sqrt(var)
                            T.vdiv(s2, var, rstd)  # rstd = sqrt(var)/var
                            # Epilogue: y = ((d*rstd)*scale) + shift
                            T.vmul(x_f32, rstd, x_f32)  # d * rstd
                            T.vcast(stage_s, sq_f32, round_mode="rint")
                            T.vmul(x_f32, sq_f32, x_f32)  # * scale
                            T.vcast(stage_h, sq_f32, round_mode="rint")
                            T.vadd(x_f32, sq_f32, x_f32)  # + shift
                            # Single round-back to the input dtype (R4);
                            # stage_x's input role is dead since the first
                            # vcast, reuse it as the output staging.
                            T.vcast(x_f32, stage_x, round_mode="rint")

                            # --- 3. Store (valid rows only) ---
                            T.copy(
                                stage_x[0:real_m, 0:N],
                                y[off_m : off_m + real_m, 0:N],
                            )

        else:

            @T.prim_func
            def main(
                x: T.Tensor[(M, N), dtype],
                scale: T.Tensor[(M, N), dtype],
                shift: T.Tensor[(M, N), dtype],
                _dummy: T.Tensor[(1,), dtype],
                y: T.Tensor[(M, N), dtype],
            ):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    # fp32 path (v2_op1 restructure): no dtype staging;
                    # dedicated scale/shift buffers so all three loads
                    # stream up-front before the compute chain.
                    #   x_f32    : x -> d (kept, O3) -> final result
                    #   sq_f32   : d^2
                    #   scale_f32/shift_f32: modulation, loaded up-front
                    x_f32 = T.alloc_shared((block_m, N), "float32")
                    sq_f32 = T.alloc_shared((block_m, N), "float32")
                    scale_f32 = T.alloc_shared((block_m, N), "float32")
                    shift_f32 = T.alloc_shared((block_m, N), "float32")
                    s1 = T.alloc_shared((block_m, 1), "float32")
                    mean = T.alloc_shared((block_m, 1), "float32")
                    s2 = T.alloc_shared((block_m, 1), "float32")
                    var = T.alloc_shared((block_m, 1), "float32")
                    rstd = T.alloc_shared((block_m, 1), "float32")

                    for i in T.serial(num_local_tasks):
                        block_id = i * num_kernels + cid
                        if block_id < num_logical:
                            off_m = block_id * block_m
                            real_m = T.min(block_m, M - off_m)

                            # All three loads up-front (same-dtype
                            # GM -> UB copies, fp32 direct path).
                            T.copy(
                                x[off_m : off_m + real_m, 0:N],
                                x_f32[0:real_m, 0:N],
                            )
                            T.copy(
                                scale[off_m : off_m + real_m, 0:N],
                                scale_f32[0:real_m, 0:N],
                            )
                            T.copy(
                                shift[off_m : off_m + real_m, 0:N],
                                shift_f32[0:real_m, 0:N],
                            )
                            T.reduce_sum(x_f32, s1, dim=1)
                            T.vmul(s1, N_inv, mean)
                            T.vsub(x_f32, mean, x_f32)
                            T.vmul(x_f32, x_f32, sq_f32)
                            T.reduce_sum(sq_f32, s2, dim=1)
                            T.vmul(s2, N_inv, var)
                            T.vadd(var, eps, var)
                            # rstd = 1/sqrt(var): vsqrt+vdiv composition
                            # (raw T.vrsqrt is approximate; see module
                            # Implementation Notes).
                            T.vsqrt(var, s2)  # s2 reused: sqrt(var)
                            T.vdiv(s2, var, rstd)  # rstd = sqrt(var)/var
                            T.vmul(x_f32, rstd, x_f32)
                            T.vmul(x_f32, scale_f32, x_f32)
                            T.vadd(x_f32, shift_f32, x_f32)
                            # Direct fp32 write-back, valid rows only.
                            T.copy(
                                x_f32[0:real_m, 0:N],
                                y[off_m : off_m + real_m, 0:N],
                            )

        return main

    return _func


# ---------------------------------------------------------------------------
# Precision comparing func + hierarchical testing
# ---------------------------------------------------------------------------

_FACTORY_CACHE = {}


def _get_factory(M, N, eps, dtype_str, use_cp_async=False):
    """Cache factory instances (mirrors the wrapper's one-factory-per-op
    usage; the jit wrapper then caches compiled mains per block_m)."""
    key = (M, N, eps, dtype_str, use_cp_async)
    if key not in _FACTORY_CACHE:
        _FACTORY_CACHE[key] = _ada_layer_norm_kernel(
            M, N, eps, dtype_str, has_gate=False, use_cp_async=use_cp_async
        )
    return _FACTORY_CACHE[key]


# Per-dtype worst max_diff across the whole run (for the final summary).
_STATS_MAX_DIFF = {}


def _compare(out, ref, atol, rtol, equal_nan=False):
    """Return (max_diff, violation_count) between out and ref.

    NaN/Inf aware: positions where both are NaN count as equal when
    equal_nan is set; equal infinities count as equal.
    """
    eq = out == ref
    if equal_nan:
        eq = eq | (torch.isnan(out) & torch.isnan(ref))
    diff = torch.where(
        eq,
        torch.zeros((), dtype=torch.float32),
        (out.float() - ref.float()).abs(),
    )
    max_diff = diff.max().item()
    violations = int((diff > (atol + rtol * ref.float().abs())).sum().item())
    return max_diff, violations


def _run_case(
    dtype_str,
    M=None,
    N=None,
    block_m=None,
    tag="L0",
    x=None,
    scale=None,
    shift=None,
    eps=1e-5,
    equal_nan=False,
    use_cp_async=False,
):
    """Run one kernel case and compare against the golden.

    Tensors x/scale/shift may be pre-built as CPU tensors (special-value
    cases); by default randn inputs are generated on CPU, cast, and
    moved to NPU (device-independent data generation, mish.py pattern).
    Raises AssertionError on tolerance violation.
    """
    torch_dtype, atol, rtol = _TOLERANCE[dtype_str]
    if x is None:
        x = torch.randn(M, N, dtype=torch.float32).to(torch_dtype)
        scale = torch.randn(M, N, dtype=torch.float32).to(torch_dtype)
        shift = torch.randn(M, N, dtype=torch.float32).to(torch_dtype)
    else:
        assert scale is not None and shift is not None, "need all three inputs"
        M, N = x.shape[0], x.shape[1]
        assert x.dtype == torch_dtype, f"input dtype {x.dtype} != expected {torch_dtype}"

    if block_m is None:
        block_m = _select_block_m(M, N, use_fp32_transit=dtype_str in ("float16", "bfloat16"))

    x_npu = x.npu()
    scale_npu = scale.npu()
    shift_npu = shift.npu()
    dummy = torch.empty(1, dtype=torch_dtype).npu()

    kernel = _get_factory(M, N, eps, dtype_str, use_cp_async)(block_m)
    y = kernel(x_npu, scale_npu, shift_npu, dummy)
    ref = golden_ada_layer_norm(x, scale, shift, eps)
    y_cpu = y.cpu()

    max_diff, violations = _compare(y_cpu, ref, atol, rtol, equal_nan=equal_nan)
    if max_diff == max_diff:  # NaN guard (all-equal cases give 0.0)
        best = _STATS_MAX_DIFF.get(dtype_str, 0.0)
        if max_diff > best:
            _STATS_MAX_DIFF[dtype_str] = max_diff

    try:
        torch.testing.assert_close(y_cpu, ref, rtol=rtol, atol=atol, equal_nan=equal_nan)
    except AssertionError:
        print(
            f"[{tag}] FAIL: shape=({M},{N}) dtype={dtype_str} "
            f"block_m={block_m} max_diff={max_diff:.3e} "
            f"violations={violations}/{M * N}"
        )
        raise
    print(
        f"[{tag}] PASS: shape=({M},{N}) dtype={dtype_str} block_m={block_m} max_diff={max_diff:.3e}"
    )
    return max_diff, violations


def _try_case(failures, case_id, **kwargs):
    """Run _run_case, collecting (not re-raising) per-case failures.

    AssertionError is treated as a precision failure; any other
    exception is recorded as a runtime error (both block the gate).
    """
    try:
        _run_case(**kwargs)
    except AssertionError as exc:
        failures.append((case_id, exc))
        first_line = str(exc).splitlines()[0] if str(exc) else repr(exc)
        print(f"[collected] {case_id} FAILED: {first_line}")
        return False
    except Exception as exc:  # noqa: BLE001
        failures.append((case_id, exc))
        print(f"[collected] {case_id} RUNTIME-ERROR: {exc!r}")
        return False
    return True


def _run_L0_3d_reshape(failures):
    """L0-13: 3D input (2, 512, 4096) fp16 via the Op-layer reshape path.

    The kernel is 2D (M, N); the Op layer reshapes any leading dims to
    (M, N) as a host metadata view (DESIGN.md sections 4.6 and 9.3).
    The golden runs on the original 3D tensors (last-dim LayerNorm).
    """
    case_id = "L0-13/3d-reshape"
    try:
        torch_dtype, atol, rtol = _TOLERANCE["float16"]
        B, S, H = 2, 512, 4096
        M = B * S
        x3 = torch.randn(B, S, H, dtype=torch.float32).to(torch_dtype)
        s3 = torch.randn(B, S, H, dtype=torch.float32).to(torch_dtype)
        h3 = torch.randn(B, S, H, dtype=torch.float32).to(torch_dtype)
        # Host metadata view: contiguous reshape, no data movement.
        x2 = x3.reshape(M, H).npu()
        s2 = s3.reshape(M, H).npu()
        h2 = h3.reshape(M, H).npu()
        dummy = torch.empty(1, dtype=torch_dtype).npu()
        block_m = 2
        kernel = _get_factory(M, H, 1e-5, "float16")(block_m)
        y = kernel(x2, s2, h2, dummy).cpu().reshape(B, S, H)
        ref = golden_ada_layer_norm(x3, s3, h3, 1e-5)
        max_diff, violations = _compare(y, ref, atol, rtol)
        torch.testing.assert_close(y, ref, rtol=rtol, atol=atol)
        best = _STATS_MAX_DIFF.get("float16", 0.0)
        if max_diff > best:
            _STATS_MAX_DIFF["float16"] = max_diff
        print(
            f"[L0-13] PASS: shape=(2,512,4096)->(1024,4096) dtype=float16 "
            f"block_m={block_m} max_diff={max_diff:.3e}"
        )
    except AssertionError as exc:
        failures.append((case_id, exc))
        first_line = str(exc).splitlines()[0] if str(exc) else repr(exc)
        print(f"[collected] {case_id} FAILED: {first_line}")
    except Exception as exc:  # noqa: BLE001
        failures.append((case_id, exc))
        print(f"[collected] {case_id} RUNTIME-ERROR: {exc!r}")


def run_L0():
    """L0 gate suite (DESIGN.md section 8.3, all 14 items).

    Returns the list of (case_id, exception) failures.
    """
    failures = []

    # (M, N, dtype, block_m, case_id) -- DESIGN.md section 8.3 table,
    # v2_op1 budget adjustments: L0-1 fp32 bm 2 -> 1 and L0-9/fp32 bm
    # 8 -> 6 (fp32 v2 multi-buffered budget 26 B/elem, probe-measured;
    # 2x4096 and 8x1152 exceed 192 KB). Transit-path cases keep the
    # baseline bm values (transit budget 20.01 B/elem, unchanged cap).
    cases = [
        (1024, 4096, "float32", 1, "L0-1"),  # aligned fp32 direct path
        (1024, 4096, "float16", 2, "L0-2"),  # aligned fp32-transit path
        (1024, 4096, "bfloat16", 2, "L0-3"),  # bf16 -> fp32 -> bf16 path
        (1024, 3000, "float16", 3, "L0-4"),  # N not a power of two
        (1025, 4096, "float16", 2, "L0-5"),  # M tail (1025 % 2 = 1)
        (1025, 4096, "bfloat16", 2, "L0-6"),  # M tail + bf16
        (16, 1152, "float16", 8, "L0-7"),  # naturally unaligned N (smoke)
        (17, 514, "float16", 4, "L0-8"),  # row tail + tiny N
        (64, 1152, "float32", 4, "L0-9/fp32"),  # manifest smoke-dit (tuned bm=4)
        (64, 1152, "float16", 4, "L0-9/fp16"),
        (64, 1152, "bfloat16", 4, "L0-9/bf16"),
        (1024, 1152, "float16", 7, "L0-10"),  # manifest dit-xl-2 (tuned bm=7)
        (2048, 4096, "float16", 2, "L0-11"),  # manifest llama-prefill (tuned bm=2)
        (1, 4096, "bfloat16", 1, "L0-12"),  # manifest llama-decode (tuned bm=1)
        # L0-13 (3D reshape) handled separately below.
        (5, 13, "float16", 4, "L0-14"),  # tiny odd shape (defensive)
    ]
    for M, N, dtype_str, block_m, case_id in cases:
        _try_case(
            failures,
            case_id,
            M=M,
            N=N,
            dtype_str=dtype_str,
            block_m=block_m,
            tag=case_id.split("/")[0],
        )

    # L0-13: 3D input via the Op-layer reshape path.
    _run_L0_3d_reshape(failures)

    if failures:
        print(f"[L0] {len(failures)} failing case(s): " + ", ".join(cid for cid, _ in failures))
    else:
        print("[L0] ALL PASS (14 items, 16 runs)")
    return failures


def _run_L1_contract(failures):
    """L1 contract checks (factory guards, no precision comparison)."""
    # UB budget guard rejects over-budget block_m values (measured
    # 20 B/elem multi-buffered budget, see module Constants).
    for M, N, dtype_str, bm in (
        (64, 4096, "float32", 3),  # 3*4096 = 12288 > 9216 (245760 B)
        (64, 4096, "float16", 3),  # 12288 > 9216
        (64, 4096, "float32", 4),  # 16384 > 9216
    ):
        try:
            _ada_layer_norm_kernel(M, N, 1e-5, dtype_str)(bm)
        except ValueError as exc:
            print(f"[L1] PASS: UB guard rejects {dtype_str} bm={bm} N={N} ({exc})")
        else:
            failures.append(
                (
                    f"L1/UB-guard/{dtype_str}/bm={bm}",
                    AssertionError(f"{dtype_str} bm={bm} not rejected"),
                )
            )

    # Out-of-domain N (beyond the residency bound) is intercepted by the
    # same guard via the default block_m (DESIGN.md section 9.3). The
    # guard fires when _func(block_m) is invoked, not at factory build.
    try:
        _ada_layer_norm_kernel(4, 20000, 1e-5, "float16")()
    except ValueError as exc:
        print(f"[L1] PASS: domain guard rejects N=20000 fp16 ({exc})")
    else:
        failures.append(("L1/domain-guard-N20000", AssertionError("N=20000 fp16 not rejected")))

    # has_gate=True is explicitly out of scope.
    try:
        _ada_layer_norm_kernel(64, 1152, 1e-5, "float16", has_gate=True)
    except NotImplementedError:
        print("[L1] PASS: has_gate=True raises NotImplementedError")
    else:
        failures.append(("L1/has_gate", AssertionError("has_gate=True did not raise")))

    # Non-positive M / N are rejected.
    for bad_m, bad_n in ((0, 1152), (64, 0)):
        try:
            _ada_layer_norm_kernel(bad_m, bad_n, 1e-5, "float16")
        except ValueError:
            print(f"[L1] PASS: M/N guard rejects M={bad_m}, N={bad_n}")
        else:
            failures.append(
                (
                    f"L1/mn-guard/{bad_m}x{bad_n}",
                    AssertionError(f"M={bad_m}, N={bad_n} not rejected"),
                )
            )

    # Unsupported dtype is rejected.
    try:
        _ada_layer_norm_kernel(64, 1152, 1e-5, "int32")
    except ValueError:
        print("[L1] PASS: dtype guard rejects int32")
    else:
        failures.append(("L1/dtype-guard", AssertionError("int32 not rejected")))


def run_L1():
    """L1: full functional coverage (dtype x shape x block_m + contracts).

    Returns the list of (case_id, exception) failures.
    """
    failures = []

    # dtype x {aligned, M-tail, unaligned-N, tiny} grid.
    for dtype_str in ("float32", "float16", "bfloat16"):
        for M, N, bm in ((128, 512, 4), (130, 256, 8), (33, 1024, 2), (17, 514, 4)):
            _try_case(
                failures,
                f"L1/{dtype_str}/{M}x{N}/bm={bm}",
                M=M,
                N=N,
                dtype_str=dtype_str,
                block_m=bm,
                tag="L1",
            )

    # Persistent core-count switch boundary (num_logical 47/48/50 with
    # 48 Vector cores; 50 also carries an M tail: 397 = 49*8 + 5).
    for M, logical in ((376, 47), (384, 48), (397, 50)):
        _try_case(
            failures,
            f"L1/persistent/num_logical={logical}",
            M=M,
            N=1152,
            dtype_str="float16",
            block_m=8,
            tag="L1",
        )

    # block_m sweep on one shape (wrapper config coverage).
    for bm in (1, 2, 4, 8):
        _try_case(
            failures,
            f"L1/fp16/96x1152/bm={bm}",
            M=96,
            N=1152,
            dtype_str="float16",
            block_m=bm,
            tag="L1",
        )

    # Factory-default block_m path (UB-budget table value).
    _try_case(
        failures,
        "L1/fp16/50x3000/default-bm",
        M=50,
        N=3000,
        dtype_str="float16",
        tag="L1",
    )
    _try_case(
        failures,
        "L1/fp32/50x3000/default-bm",
        M=50,
        N=3000,
        dtype_str="float32",
        tag="L1",
    )

    # Non-default eps (exercises the eps plumbing: var + eps placement).
    _try_case(
        failures,
        "L1/fp16/64x1152/eps=1e-2",
        M=64,
        N=1152,
        dtype_str="float16",
        block_m=8,
        eps=1e-2,
        tag="L1",
    )

    # use_cp_async is accepted and ignored (interface compatibility).
    _try_case(
        failures,
        "L1/fp16/64x1152/use_cp_async=True",
        M=64,
        N=1152,
        dtype_str="float16",
        block_m=8,
        use_cp_async=True,
        tag="L1",
    )

    # E2 factory contract: one factory callable serving two block_m
    # values (the jit wrapper caches the compiled main per block_m).
    try:
        dtype_str = "float16"
        torch_dtype, atol, rtol = _TOLERANCE[dtype_str]
        M, N = 96, 1152
        x = torch.randn(M, N, dtype=torch.float32).to(torch_dtype)
        scale = torch.randn(M, N, dtype=torch.float32).to(torch_dtype)
        shift = torch.randn(M, N, dtype=torch.float32).to(torch_dtype)
        ref = golden_ada_layer_norm(x, scale, shift, 1e-5)
        fn = _get_factory(M, N, 1e-5, dtype_str)
        dummy = torch.empty(1, dtype=torch_dtype).npu()
        for bm in (8, 4):
            out = fn(bm)(x.npu(), scale.npu(), shift.npu(), dummy).cpu()
            torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)
        print("[L1] PASS: E2 factory(block_m) dual-config reuse")
    except AssertionError as exc:
        failures.append(("L1/E2-reuse", exc))
        print(f"[collected] L1/E2-reuse FAILED: {exc}")
    except Exception as exc:  # noqa: BLE001
        failures.append(("L1/E2-reuse", exc))
        print(f"[collected] L1/E2-reuse RUNTIME-ERROR: {exc!r}")

    # Factory contract guards (no NPU run).
    _run_L1_contract(failures)

    if failures:
        print(f"[L1] {len(failures)} failing case(s): " + ", ".join(cid for cid, _ in failures))
    else:
        print("[L1] ALL PASS")
    return failures


def run_L2():
    """L2: extreme sizes (warn only, non-blocking)."""
    cases = [
        (1, 1, "float16", 1),  # minimal
        (2, 3, "bfloat16", 1),  # tiny odd
        (1, 4096, "float32", None),  # single row, default bm
        (3, 8192, "float16", 1),  # N beyond the manifest max
        (5000, 64, "float16", 8),  # large M, small N (deep serial)
        (1, 8192, "bfloat16", 1),  # single row bf16, large N
    ]
    for M, N, dtype_str, bm in cases:
        try:
            _run_case(
                M=M,
                N=N,
                dtype_str=dtype_str,
                block_m=bm,
                tag="L2",
            )
        except Exception as e:  # noqa: BLE001
            print(f"[L2] WARN (Record without blocking): shape=({M},{N}) {dtype_str}: {e}")


def run_boundary():
    """Boundary: special-value inputs (warn only, non-blocking)."""
    M, N = 32, 512

    # Zeros: var = 0 -> rstd = rsqrt(eps), y = shift exactly.
    try:
        _run_case(
            dtype_str="float16",
            tag="Boundary-zeros",
            x=torch.zeros(M, N, dtype=torch.float16),
            scale=torch.randn(M, N, dtype=torch.float32).to(torch.float16),
            shift=torch.randn(M, N, dtype=torch.float32).to(torch.float16),
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): zeros: {e}")

    # Constant rows (var = 0 as well, non-zero mean).
    try:
        _run_case(
            dtype_str="float32",
            tag="Boundary-const-rows",
            x=torch.full((M, N), 3.7, dtype=torch.float32),
            scale=torch.randn(M, N, dtype=torch.float32),
            shift=torch.randn(M, N, dtype=torch.float32),
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): const-rows: {e}")

    # Large fp16 magnitudes (x100 scale).
    try:
        _run_case(
            dtype_str="float16",
            tag="Boundary-large",
            x=(torch.randn(M, N, dtype=torch.float32) * 100.0).to(torch.float16),
            scale=torch.randn(M, N, dtype=torch.float32).to(torch.float16),
            shift=torch.randn(M, N, dtype=torch.float32).to(torch.float16),
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): large: {e}")

    # Subnormal fp16 magnitudes (~1e-5 scale; exact fp32 upcast).
    try:
        _run_case(
            dtype_str="float16",
            tag="Boundary-subnormal",
            x=(torch.randn(M, N, dtype=torch.float32) * 1e-5).to(torch.float16),
            scale=torch.randn(M, N, dtype=torch.float32).to(torch.float16),
            shift=torch.randn(M, N, dtype=torch.float32).to(torch.float16),
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): subnormal: {e}")

    # Extreme scale magnitude (fp16 max 65504, cancellation corner from
    # the D-1 verify_equiv coverage; most products saturate to inf on
    # both sides, equal infinities compare equal).
    try:
        _run_case(
            dtype_str="float16",
            tag="Boundary-extreme-scale",
            x=torch.randn(M, N, dtype=torch.float32).to(torch.float16),
            scale=torch.where(
                torch.rand(M, N) > 0.5,
                torch.full((M, N), 65504.0),
                torch.full((M, N), -65504.0),
            ).to(torch.float16),
            shift=torch.randn(M, N, dtype=torch.float32).to(torch.float16),
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): extreme-scale: {e}")

    # inf/NaN natural propagation (DESIGN.md section 0.1: no special
    # handling; a full-inf row yields NaN output on both sides).
    try:
        x = torch.randn(M, N, dtype=torch.float32)
        x[0, :] = float("inf")  # full-inf row -> NaN row
        x[1, ::4] = float("nan")  # scattered NaN -> NaN row
        x[2, ::4] = float("inf")  # mixed inf row -> NaN row
        _run_case(
            dtype_str="float16",
            tag="Boundary-inf-nan",
            x=x.to(torch.float16),
            scale=torch.randn(M, N, dtype=torch.float32).to(torch.float16),
            shift=torch.randn(M, N, dtype=torch.float32).to(torch.float16),
            equal_nan=True,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): inf-nan: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "all"])
    args, _ = parser.parse_known_args()

    failures = []

    t0 = time.time()
    failures += run_L0()
    print(f"[L0] elapsed {time.time() - t0:.1f}s")

    if args.level == "all":
        t1 = time.time()
        failures += run_L1()
        print(f"[L1] elapsed {time.time() - t1:.1f}s")
        t2 = time.time()
        run_L2()
        print(f"[L2] elapsed {time.time() - t2:.1f}s")
        t3 = time.time()
        run_boundary()
        print(f"[Boundary] elapsed {time.time() - t3:.1f}s")

    print(
        "Summary max_diff per dtype: "
        + ", ".join(f"{k}={v:.3e}" for k, v in sorted(_STATS_MAX_DIFF.items()))
    )
    if failures:
        raise AssertionError(
            f"{len(failures)} case(s) failed: " + "; ".join(cid for cid, _ in failures)
        )
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    main()
