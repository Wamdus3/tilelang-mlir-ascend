import os
import torch
import tilelang
from tilelang import language as T

dtype = "bfloat16"
accum_dtype = "float32"


@tilelang.jit(target="npuir")
def get_engram_gate_fwd_kernel(
        num_tokens: int,
        hidden_size: int,
        eps: float,
        scalar: float,
        clamp_value: float = 1e-6,
        hc_mult: int = 4,
        save_for_backward: bool = True,
):
    """Forward kernel. When save_for_backward=True, saves dot/gate_score/rstd_x/rstd_k for backward."""
    threads = 32
    vec_size = 8

    # NOTE Performance only tuned for hidden_size in {4096, 7168}
    def _choose_blk_d(hidden_size):
        for blk in [1024, 768, 512, 256]:
            if hidden_size % blk == 0 and hidden_size >= 2 * blk:
                return blk
        raise ValueError(f"No valid blk_d for hidden_size={hidden_size}")

    blk_d = _choose_blk_d(hidden_size)
    num_persistent_blocks = 6

    assert hidden_size % blk_d == 0
    assert hidden_size >= 2 * blk_d
    num_blk = hidden_size // blk_d
    reduce_blk = threads * vec_size
    sub_blks = blk_d // reduce_blk
    v_start_phase = num_blk % 2

    @T.prim_func
    def engram_gate_fwd_kernel(
            hidden_states: T.Tensor([num_tokens, hc_mult, hidden_size], dtype),
            k: T.Tensor([num_tokens, hc_mult, hidden_size], dtype),
            v: T.Tensor([num_tokens, hidden_size], dtype),
            weight_fused: T.Tensor([hc_mult, hidden_size], accum_dtype),
            output: T.Tensor([num_tokens, hc_mult, hidden_size], dtype),
            dot_out: T.Tensor([num_tokens, hc_mult], accum_dtype),
            gate_score: T.Tensor([num_tokens, hc_mult], accum_dtype),
            rstd_x: T.Tensor([num_tokens, hc_mult], accum_dtype),
            rstd_k: T.Tensor([num_tokens, hc_mult], accum_dtype),
            kk_out: T.Tensor([num_tokens, hc_mult, hidden_size], accum_dtype),
    ):
        with T.Kernel(hc_mult * num_persistent_blocks, is_npu=True) as (cid, _):
            pid_h = cid % hc_mult
            pid_b = cid // hc_mult
            x_local_1 = T.alloc_fragment((vec_size,), accum_dtype)
            k_local_1 = T.alloc_fragment((vec_size,), accum_dtype)
            w_local_1 = T.alloc_fragment((vec_size,), accum_dtype)
            x_local_2 = T.alloc_fragment((vec_size,), accum_dtype)
            k_local_2 = T.alloc_fragment((vec_size,), accum_dtype)
            w_local_2 = T.alloc_fragment((vec_size,), accum_dtype)
            v_local = T.alloc_fragment((vec_size,), accum_dtype)
            tmp_local = T.alloc_fragment((vec_size,), accum_dtype)

            gate_score_local = T.alloc_shared((1,), accum_dtype)
            rstd_x_local = T.alloc_shared((1,), accum_dtype)
            rstd_k_local = T.alloc_shared((1,), accum_dtype)

            gate_score_reducer = T.alloc_shared((1,), accum_dtype)
            rstd_x_reducer = T.alloc_shared((1,), accum_dtype)
            rstd_k_reducer = T.alloc_shared((1,), accum_dtype)

            x_smem = T.alloc_shared((hidden_size,), dtype)
            # 原二维kv_smem拆分为两个独立的一维共享内存
            kv_smem_0 = T.alloc_shared((blk_d,), dtype)
            kv_smem_1 = T.alloc_shared((blk_d,), dtype)
            tmpfp = T.alloc_fragment((vec_size,), dtype)

            per_block = T.ceildiv(num_tokens, num_persistent_blocks)
            t_start = T.min(per_block * pid_b, num_tokens)
            t_end = T.min(per_block * (pid_b + 1), num_tokens)

            tmp_val = T.alloc_fragment((vec_size,), accum_dtype)

            for i_s in T.serial(t_start, t_end):
                # Correctness-first variant: make each token self-contained instead
                # of relying on pass2 to prefetch the next token into shared memory.
                T.copy(hidden_states[i_s, pid_h, 0:blk_d], x_smem[0:blk_d])
                T.copy(k[i_s, pid_h, 0:blk_d], kv_smem_0[:])

                T.clear(rstd_k_local)
                T.clear(rstd_x_local)
                T.clear(gate_score_local)

                for i_b in T.serial(1, num_blk):
                    phase = i_b % 2
                    prev_phase = (i_b - 1) % 2
                    T.copy(
                        hidden_states[i_s, pid_h, i_b * blk_d: (i_b + 1) * blk_d],
                        x_smem[i_b * blk_d: (i_b + 1) * blk_d],
                    )
                    # 动态phase写入：直接访问对应一维共享内存
                    if phase == 0:
                        T.copy(k[i_s, pid_h, i_b * blk_d: (i_b + 1) * blk_d], kv_smem_0[:])
                    else:
                        T.copy(k[i_s, pid_h, i_b * blk_d: (i_b + 1) * blk_d], kv_smem_1[:])

                    for i_sub in T.serial(sub_blks):
                        sub_base = (i_b - 1) * blk_d + i_sub * reduce_blk
                        for tid in T.serial(threads):
                            for i_k in T.Parallel(vec_size):
                                x_local_1[i_k] = x_smem[sub_base + tid * vec_size + i_k]
                                # 动态(num_blk-1)%2读取：直接访问对应一维共享内存
                                if prev_phase == 0:
                                    tmpfp[i_k] = kv_smem_0[i_sub * reduce_blk + tid * vec_size + i_k]
                                else:
                                    tmpfp[i_k] = kv_smem_1[i_sub * reduce_blk + tid * vec_size + i_k]
                            # if prev_phase == 0:
                            #     T.copy(kv_smem_0[
                            #            i_sub * reduce_blk + tid * vec_size: i_sub * reduce_blk + tid * vec_size + vec_size],
                            #            tmpfp)
                            # else:
                            #     T.copy(kv_smem_1[
                            #            i_sub * reduce_blk + tid * vec_size: i_sub * reduce_blk + tid * vec_size + vec_size],
                            #            tmpfp)
                            T.vcast(tmpfp, k_local_1)

                            T.copy(weight_fused[pid_h, sub_base + tid * vec_size: sub_base + tid * vec_size + vec_size],
                                   w_local_1)
                            T.vmul(x_local_1, x_local_1, tmp_local)
                            T.reduce(
                                tmp_local,
                                rstd_x_local,
                                dims=0,
                                reduce_mode="sum",
                            )
                            rstd_x_reducer[0] = rstd_x_reducer[0] + rstd_x_local[0]

                            # T.copy(tmp_local, kk_out[i_s, pid_h, i_b * blk_d + i_sub * reduce_blk + tid * vec_size: i_b * blk_d + i_sub * reduce_blk + tid * vec_size + vec_size])
                            for i_k in T.Parallel(vec_size):
                                kk_out[i_s, pid_h, sub_base + tid * vec_size + i_k] = tmp_local[i_k]


                            T.reduce(
                                tmp_local,
                                rstd_k_local,
                                dims=0,
                                reduce_mode="sum",
                                clear=False,
                            )
                            rstd_k_reducer[0] = rstd_k_reducer[0] + rstd_k_local[0]

                            T.vmul(x_local_1, w_local_1, tmp_local)
                            T.vmul(tmp_local, k_local_1, tmp_local)
                            T.reduce(
                                tmp_local,
                                gate_score_local,
                                dims=0,
                                reduce_mode="sum",
                                clear=False,
                            )
                            gate_score_reducer[0] = gate_score_reducer[0] + gate_score_local[0]

                # Prefetch v[0] into freed kv_smem bank
                # T.copy(v[i_s, 0:blk_d], kv_smem[v_start_phase, :])  # 完全保留注释

                for i_sub in T.serial(sub_blks):
                    sub_base = (num_blk - 1) * blk_d + i_sub * reduce_blk
                    for tid in T.serial(threads):
                        for i_k in T.Parallel(vec_size):
                            x_local_2[i_k] = x_smem[sub_base + tid * vec_size + i_k]
                            # 动态(num_blk-1)%2读取：直接访问对应一维共享内存
                            if (num_blk - 1) % 2 == 0:
                                k_local_2[i_k] = kv_smem_0[i_sub * reduce_blk + tid * vec_size + i_k]
                            else:
                                k_local_2[i_k] = kv_smem_1[i_sub * reduce_blk + tid * vec_size + i_k]

                        T.copy(weight_fused[pid_h, sub_base + tid * vec_size: sub_base + tid * vec_size + vec_size],
                               w_local_2)
                        T.vmul(x_local_2, x_local_2, tmp_local)
                        T.reduce(
                            tmp_local,
                            rstd_x_local,
                            dims=0,
                            reduce_mode="sum",
                        )
                        rstd_x_reducer[0] = rstd_x_reducer[0] + rstd_x_local[0]

                        T.vmul(k_local_2, k_local_2, tmp_local)
                        # T.copy(tmp_local, kk_out[i_s, pid_h, (num_blk - 1) * blk_d + i_sub * reduce_blk + tid * vec_size: (num_blk - 1) * blk_d + i_sub * reduce_blk + tid * vec_size + vec_size])

                        for i_k in T.Parallel(vec_size):
                            kk_out[i_s, pid_h, sub_base + tid * vec_size + i_k] = tmp_local[i_k]

                        T.reduce(
                            tmp_local,
                            rstd_k_local,
                            dims=0,
                            reduce_mode="sum",
                        )
                        rstd_k_reducer[0] = rstd_k_reducer[0] + rstd_k_local[0]

                        T.vmul(x_local_2, w_local_2, tmp_local)
                        T.vmul(tmp_local, k_local_2, tmp_local)
                        T.reduce(
                            tmp_local,
                            gate_score_local,
                            dims=0,
                            reduce_mode="sum",
                        )
                        gate_score_reducer[0] = gate_score_reducer[0] + gate_score_local[0]

                # rstd_x_reducer[0] = T.rsqrt(rstd_x_reducer[0] / hidden_size + eps)

                rstd_x_reducer[0] = rstd_x_reducer[0] / hidden_size + eps
                T.vrsqrt(rstd_x_reducer, rstd_x_reducer)
                # rstd_k_reducer[0] = T.rsqrt(rstd_k_reducer[0] / hidden_size + eps)
                rstd_k_reducer[0] = rstd_k_reducer[0] / hidden_size + eps
                T.vrsqrt(rstd_k_reducer, rstd_k_reducer)

                if save_for_backward:
                    dot_out[i_s, pid_h] = gate_score_reducer[0]
                    # T.copy(gate_score_reducer, dot_out[i_s, pid_h: pid_h + 1])
                    rstd_x[i_s, pid_h]  = rstd_x_reducer[0]
                    # T.copy(rstd_x_reducer, rstd_x[i_s, pid_h: pid_h + 1])
                    rstd_k[i_s, pid_h]  = rstd_k_reducer[0]
                    # T.copy(rstd_k_local, rstd_k[i_s, pid_h: pid_h + 1])

                gate_score_reducer[0] = (
                        gate_score_reducer[0]
                        * rstd_x_reducer[0]
                        * rstd_k_reducer[0]
                        * scalar
                )

                # gate_score_reducer[0] = T.sigmoid(T.copysign(T.sqrt(T.clamp(T.abs(gate_score_reducer[0]), clamp_value, float('inf'))), gate_score_reducer[0]))
                gate_raw = T.alloc_fragment((1,), accum_dtype)
                gate_abs = T.alloc_fragment((1,), accum_dtype)
                gate_sqrt = T.alloc_fragment((1,), accum_dtype)
                gate_raw[0] = gate_score_reducer[0]
                T.vabs(gate_raw, gate_abs)
                T.vclamp(gate_abs, gate_abs, clamp_value, float("inf"))
                T.vsqrt(gate_abs, gate_sqrt)
                if gate_raw[0] < 0:
                    gate_score_reducer[0] = -gate_sqrt[0]
                else:
                    gate_score_reducer[0] = gate_sqrt[0]
                T.vsigmoid(gate_score_reducer, gate_score_reducer)

                if save_for_backward:
                    # gate_score[i_s, pid_h] = gate_score_reducer[0]
                    T.copy(gate_score_reducer, gate_score[i_s, pid_h: pid_h + 1])

                # 动态v_start_phase写入：直接访问对应一维共享内存
                if v_start_phase == 0:
                    T.copy(v[i_s, 0:blk_d], kv_smem_0[:])
                else:
                    T.copy(v[i_s, 0:blk_d], kv_smem_1[:])

                if num_blk > 1:
                    if (1 - v_start_phase) == 0:
                        T.copy(v[i_s, blk_d: 2 * blk_d], kv_smem_0[:])
                    else:
                        T.copy(v[i_s, blk_d: 2 * blk_d], kv_smem_1[:])

                # === Pass 2: Output — x from smem, v from kv_smem (tiles 0,1 already prefetched) ===
                for i_b in T.serial(num_blk):
                    tile_phase = (v_start_phase + i_b) % 2
                    for i_sub in T.serial(sub_blks):
                        sub_base = i_b * blk_d + i_sub * reduce_blk
                        for tid in T.serial(threads):
                            for i_k in T.Parallel(vec_size):
                                tmp_val[i_k] = x_smem[sub_base + tid * vec_size + i_k]
                                # 动态tile_phase读取v：直接访问对应一维共享内存
                                if tile_phase == 0:
                                    v_local[i_k] = kv_smem_0[i_sub * reduce_blk + tid * vec_size + i_k]
                                else:
                                    v_local[i_k] = kv_smem_1[i_sub * reduce_blk + tid * vec_size + i_k]
                            T.vmul(v_local, gate_score_reducer, v_local)
                            T.vadd(tmp_val, v_local, tmp_val)
                            T.copy(tmp_val,
                                   output[i_s, pid_h, sub_base + tid * vec_size: sub_base + tid * vec_size + vec_size])

                    # Prefetch v[i_b+2] into freed kv_smem bank
                    if i_b + 2 < num_blk:
                        # 动态tile_phase写入v：直接访问对应一维共享内存
                        if tile_phase == 0:
                            T.copy(
                                v[i_s, (i_b + 2) * blk_d: (i_b + 3) * blk_d],
                                kv_smem_0[:],
                            )
                        else:
                            T.copy(
                                v[i_s, (i_b + 2) * blk_d: (i_b + 3) * blk_d],
                                kv_smem_1[:],
                            )

    return engram_gate_fwd_kernel


def engram_gate_ref(
        hidden_states: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        weight_hidden: torch.Tensor,
        weight_embed: torch.Tensor,
        clamp_value: float,
        eps: float,
        save_for_backward: bool = False,
) -> (
        torch.Tensor
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Pure PyTorch reference implementation of engram gate (vectorized, supports autograd).

    Computes: output = x + sigmoid(signed_sqrt(dot(RMSNorm(x, wh), RMSNorm(k, we)) * scalar)) * v

    Args:
        hidden_states: Input of shape (num_tokens, hc_mult, hidden_size), bfloat16.
        k: Key embeddings of shape (num_tokens, hc_mult, hidden_size), bfloat16.
        v: Value embeddings of shape (num_tokens, hidden_size), bfloat16.
        weight_hidden: RMSNorm weight for hidden states, shape (hc_mult, hidden_size), bfloat16.
        weight_embed: RMSNorm weight for key embeddings, shape (hc_mult, hidden_size), bfloat16.
        clamp_value: Clamp threshold for signed-sqrt gate activation.
        eps: Epsilon for RMSNorm numerical stability.
        save_for_backward: If True, also return (dot, gate_score, rstd_x, rstd_k).

    Returns:
        If save_for_backward is False: output tensor of shape (num_tokens, hc_mult, hidden_size), bfloat16.
        If save_for_backward is True: tuple of (output, dot, gate_score, rstd_x, rstd_k).
    """
    hidden_size = hidden_states.shape[-1]
    scalar = hidden_size ** -0.5

    x = hidden_states.float()
    k_f = k.float()

    kk = k_f.pow(2)

    wh = weight_hidden.float().unsqueeze(0)
    we = weight_embed.float().unsqueeze(0)

    # RMSNorm
    rstd_x = torch.rsqrt(x.pow(2).mean(-1) + eps)
    # rstd_k = torch.rsqrt(k_f.pow(2).mean(-1) + eps)
    rstd_k = k_f.pow(2).sum(-1)

    # Dot -> sqrt-gate -> sigmoid
    # raw_dot is the unnormalized sum(x * wh * k * we), matching the kernel's dot_out
    raw_dot = torch.einsum("...d,...d->...", x * wh, k_f * we)
    dot = raw_dot * rstd_x * rstd_k * scalar
    signed_sqrt = dot.abs().clamp_min(clamp_value).sqrt() * dot.sign()
    gate_score = signed_sqrt.sigmoid()

    output = x + gate_score.unsqueeze(-1) * v.unsqueeze(-2)
    output = output.bfloat16()

    if save_for_backward:
        return output, raw_dot, gate_score, rstd_x, rstd_k, kk
    return output


def engram_gate_fwd(
        hidden_states: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        weight_fused: torch.Tensor,
        eps: float,
        clamp_value: float,
        save_for_backward: bool = True,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    num_tokens, hc_mult, hidden_size = hidden_states.shape
    scalar = hidden_size ** -0.5

    kernel = get_engram_gate_fwd_kernel(
        num_tokens, hidden_size, eps, scalar, clamp_value, hc_mult, save_for_backward
    )

    output = torch.empty_like(hidden_states)
    if save_for_backward:
        dot = torch.empty(
            (num_tokens, hc_mult), dtype=torch.float32, device=hidden_states.device
        )
        gate_score = torch.empty(
            (num_tokens, hc_mult), dtype=torch.float32, device=hidden_states.device
        )
        rstd_x = torch.empty(
            (num_tokens, hc_mult), dtype=torch.float32, device=hidden_states.device
        )
        rstd_k = torch.empty(
            (num_tokens, hc_mult), dtype=torch.float32, device=hidden_states.device
        )
        kk = torch.empty(
            (num_tokens, hc_mult, hidden_size), dtype=torch.float32, device=hidden_states.device
        )
    else:
        dot = gate_score = rstd_x = rstd_k = kk = None

    kernel(hidden_states, k, v, weight_fused, output, dot, gate_score, rstd_x, rstd_k, kk)

    return output, dot, gate_score, rstd_x, rstd_k, kk


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim if denominator != 0 else 0


def assert_equal(
        x: torch.Tensor,
        y: torch.Tensor,
        check_dtype: bool = True,
        check_shape: bool = True,
        check_stride: bool = True,
) -> None:
    assert not check_dtype or x.dtype == y.dtype, (
        f"Tensor dtypes are not equal: {x.dtype} vs {y.dtype}"
    )
    assert not check_shape or x.shape == y.shape, (
        f"Tensor shapes are not equal: {x.shape} vs {y.shape}"
    )
    assert not check_stride or x.numel() == 0 or x.stride() == y.stride(), (
        f"Tensor strides are not equal: {x.stride()} vs {y.stride()}"
    )
    assert x.device == y.device, (
        f"Tensor devices are not equal: {x.device} vs {y.device}"
    )
    # Hints: The tensor with a size of [32768, 1] and a stride of [1, 32768] is considered contiguous,
    # but using .view will cause an error. Therefore, .flatten is used to ensure the stride of the last dimension is 1.
    mask = x != y
    assert torch.equal(
        x.contiguous().flatten().view(torch.uint8),
        y.contiguous().flatten().view(torch.uint8),
    ), (
        f"Tensor values are not equal: {x.shape=} vs {y.shape=}\n"
        f"mask={torch.nonzero(mask)}\n"
        f"{x[mask]}\nvs\n{y[mask]}"
    )


# def run_test():
#     seed = 2026
#     torch.manual_seed(seed)
#
#     device = "npu"
#     hc_mult = 4
#     num_tokens = 7
#     hidden_size = 512
#
#     eps = 1e-20
#     clamp_value = 1e-6
#
#     hidden_states = torch.randn(
#         (num_tokens, hc_mult, hidden_size),
#         device=device,
#         dtype=torch.bfloat16,
#     )
#
#     k = torch.randn(
#         (num_tokens, hc_mult, hidden_size),
#         device=device,
#         dtype=torch.bfloat16,
#     )
#
#     v = torch.randn(
#         (num_tokens, hidden_size),
#         device=device,
#         dtype=torch.bfloat16,
#     )
#
#     weight_hidden = torch.randn(
#         (hc_mult, hidden_size),
#         device=device,
#         dtype=torch.bfloat16,
#     )
#
#     weight_embed = torch.randn(
#         (hc_mult, hidden_size),
#         device=device,
#         dtype=torch.bfloat16,
#     )
#
#     weight_fused = (weight_hidden.float() * weight_embed.float()).contiguous()
#
#     out_save, dot, gate_score, rstd_x, rstd_k, kk = engram_gate_fwd(
#         hidden_states, k, v, weight_fused, eps, clamp_value, save_for_backward=True
#     )
#
#     out_ref, dot_ref, gate_score_ref, rstd_x_ref, rstd_k_ref, kk_ref = engram_gate_ref(
#         hidden_states,
#         k,
#         v,
#         weight_hidden,
#         weight_embed,
#         clamp_value,
#         eps,
#         save_for_backward=True,
#     )
#
#     assert (
#         dot is not None
#         and gate_score is not None
#         and rstd_x is not None
#         and rstd_k is not None
#     )
#     diff_out = calc_diff(out_save, out_ref)
#     assert diff_out < 1e-2, f"out_save mismatch: {diff_out:.6e}"
#     diff_dot = calc_diff(dot, dot_ref)
#     assert diff_dot < 1e-2, f"dot mismatch: {diff_dot:.6e}"
#     diff_gate = calc_diff(gate_score, gate_score_ref)
#     assert diff_gate < 1e-2, f"gate_score mismatch: {diff_gate:.6e}"
#     diff_rstd_x = calc_diff(rstd_x, rstd_x_ref)
#     assert diff_rstd_x < 1e-2, f"rstd_x mismatch: {diff_rstd_x:.6e}"
#     diff_rstd_k = calc_diff(rstd_k, rstd_k_ref)
#     assert diff_rstd_k < 1e-2, f"rstd_k mismatch: {diff_rstd_k:.6e}"
#     # kk <-> kk_ref
#
#     # Correctness: save_for_backward=False
#     out_no_save, dot_n, gate_score_n, rstd_x_n, rstd_k_n = engram_gate_fwd(
#         hidden_states,
#         k,
#         v,
#         weight_fused,
#         eps,
#         clamp_value,
#         save_for_backward=False,
#     )
#     assert (
#         dot_n is None and gate_score_n is None and rstd_x_n is None and rstd_k_n is None
#     )
#     diff_out = calc_diff(out_no_save, out_ref)
#     assert diff_out < 1e-2, f"out_no_save mismatch: {diff_out:.6e}"
#     assert_equal(out_no_save, out_save)
#     print("All check pass!")

def calc_mismatch_count(x: torch.Tensor, y: torch.Tensor, rtol: float = 1e-3, atol: float = 1e-3) -> tuple[
    int, int, float]:
    """统计不匹配的元素数量、总元素数、不匹配比例"""
    x, y = x.double(), y.double()
    mask = ~torch.isclose(x, y, rtol=rtol, atol=atol)
    mismatch_count = mask.sum().item()
    total_count = x.numel()
    mismatch_ratio = mismatch_count / total_count if total_count > 0 else 0.0
    return mismatch_count, total_count, mismatch_ratio


def run_test():
    seed = 2026
    torch.manual_seed(seed)

    device = "npu"
    hc_mult = 4
    num_tokens = 7
    hidden_size = 512

    eps = 1e-20
    clamp_value = 1e-6

    hidden_states = torch.randn(
        (num_tokens, hc_mult, hidden_size),
        device=device,
        dtype=torch.bfloat16,
    )

    k = torch.randn(
        (num_tokens, hc_mult, hidden_size),
        device=device,
        dtype=torch.bfloat16,
    )

    v = torch.randn(
        (num_tokens, hidden_size),
        device=device,
        dtype=torch.bfloat16,
    )

    weight_hidden = torch.randn(
        (hc_mult, hidden_size),
        device=device,
        dtype=torch.bfloat16,
    )

    weight_embed = torch.randn(
        (hc_mult, hidden_size),
        device=device,
        dtype=torch.bfloat16,
    )

    weight_fused = (weight_hidden.float() * weight_embed.float()).contiguous()

    out_save, dot, gate_score, rstd_x, rstd_k, kk = engram_gate_fwd(
        hidden_states, k, v, weight_fused, eps, clamp_value, save_for_backward=True
    )

    out_ref, dot_ref, gate_score_ref, rstd_x_ref, rstd_k_ref, kk_ref = engram_gate_ref(
        hidden_states,
        k,
        v,
        weight_hidden,
        weight_embed,
        clamp_value,
        eps,
        save_for_backward=True,
    )

    assert (
            dot is not None
            and gate_score is not None
            and rstd_x is not None
            and rstd_k is not None
    )

    # ========== 修改为统计mismatch数量 ==========
    rtol, atol = 1e-5, 1e-5

    out_mismatch, out_total, out_ratio = calc_mismatch_count(out_save, out_ref, rtol, atol)
    dot_mismatch, dot_total, dot_ratio = calc_mismatch_count(dot, dot_ref, rtol, atol)
    gate_mismatch, gate_total, gate_ratio = calc_mismatch_count(gate_score, gate_score_ref, rtol, atol)
    rstd_x_mismatch, rstd_x_total, rstd_x_ratio = calc_mismatch_count(rstd_x, rstd_x_ref, rtol, atol)
    rstd_k_mismatch, rstd_k_total, rstd_k_ratio = calc_mismatch_count(rstd_k, rstd_k_ref, rtol, atol)
    rstd_kk_mismatch, rstd_kk_total, rstd_kk_ratio = calc_mismatch_count(kk, kk_ref, rtol, atol)

    print(rstd_k)
    print(rstd_k_ref)

    print(f"=== Mismatch Statistics (rtol={rtol}, atol={atol}) ===")
    print(f"output:      {out_mismatch}/{out_total} ({out_ratio:.2%}) mismatch")
    print(f"dot:         {dot_mismatch}/{dot_total} ({dot_ratio:.2%}) mismatch")
    print(f"gate_score:  {gate_mismatch}/{gate_total} ({gate_ratio:.2%}) mismatch")
    print(f"rstd_x:      {rstd_x_mismatch}/{rstd_x_total} ({rstd_x_ratio:.2%}) mismatch")
    print(f"rstd_k:      {rstd_k_mismatch}/{rstd_k_total} ({rstd_k_ratio:.2%}) mismatch")
    print(f"rstd_k:      {rstd_kk_mismatch}/{rstd_kk_total} ({rstd_kk_ratio:.2%}) mismatch")

    # 可以设置一个宽松的阈值，比如允许0.1%以内的数值误差
    max_allowed_ratio = 0.001  # 0.1%
    assert out_ratio < max_allowed_ratio, f"out_save mismatch too high: {out_mismatch}/{out_total} ({out_ratio:.2%})"
    assert dot_ratio < max_allowed_ratio, f"dot mismatch too high: {dot_mismatch}/{dot_total} ({dot_ratio:.2%})"
    assert gate_ratio < max_allowed_ratio, f"gate_score mismatch too high: {gate_mismatch}/{gate_total} ({gate_ratio:.2%})"
    assert rstd_x_ratio < max_allowed_ratio, f"rstd_x mismatch too high: {rstd_x_mismatch}/{rstd_x_total} ({rstd_x_ratio:.2%})"
    assert rstd_k_ratio < max_allowed_ratio, f"rstd_k mismatch too high: {rstd_k_mismatch}/{rstd_k_total} ({rstd_k_ratio:.2%})"
    # ==========================================

    # Correctness: save_for_backward=False
    out_no_save, dot_n, gate_score_n, rstd_x_n, rstd_k_n = engram_gate_fwd(
        hidden_states,
        k,
        v,
        weight_fused,
        eps,
        clamp_value,
        save_for_backward=False,
    )
    assert (
            dot_n is None and gate_score_n is None and rstd_x_n is None and rstd_k_n is None
    )
    out_no_save_mismatch, out_no_save_total, out_no_save_ratio = calc_mismatch_count(out_no_save, out_ref, rtol, atol)
    print(f"out_no_save: {out_no_save_mismatch}/{out_no_save_total} ({out_no_save_ratio:.2%}) mismatch")
    assert out_no_save_ratio < max_allowed_ratio, f"out_no_save mismatch too high: {out_no_save_mismatch}/{out_no_save_total} ({out_no_save_ratio:.2%})"
    assert_equal(out_no_save, out_save)
    print("All check pass!")


if __name__ == "__main__":
    os.environ["TILELANG_ASCEND_MODE"] = "Dev"
    tilelang.cache.clear_cache()
    run_test()
# 看看该算子有什么问题，忽略标量循环
