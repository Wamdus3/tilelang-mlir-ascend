"""
Small TileLang NPUIR kernel for Mamba3 Rotary Q + Interchunk Contribution.

This isolates the rotary-Q update and q @ state GEMM from mamba3_mimo_fwd_npu.py
so the mixed Vector + Cube section can be optimized independently.
"""

import tilelang
import tilelang.language as T


@tilelang.jit(target="npuir")
def mamba3_rotary_q_interchunk(
    N: int,
    P: int,
    R: int,
    chunk_size: int,
    rotary_dim_divisor: int = 4,
    dtype: str = "float32",
):
    accum_dtype = "float32"
    fused_chunk_size = chunk_size * R
    rotary_dim = N // rotary_dim_divisor
    pos_half = rotary_dim_divisor // 2

    @T.prim_func
    def mamba3_rotary_q_interchunk_kernel(
        Q: T.Tensor((fused_chunk_size, N), dtype),
        ANGLES: T.Tensor((chunk_size, rotary_dim), "float32"),
        STATES: T.Tensor((N, P), accum_dtype),
        O: T.Tensor((fused_chunk_size, P), accum_dtype),
    ):
        with T.Kernel(1, is_npu=True):
            q_shared = T.alloc_shared((fused_chunk_size, N), accum_dtype)
            q_reshaped = T.alloc_fragment((chunk_size, R, rotary_dim_divisor, rotary_dim), accum_dtype)
            q_worked = T.alloc_fragment((fused_chunk_size, N), accum_dtype)
            q_first_sin = T.alloc_fragment((chunk_size, R, 1, rotary_dim), accum_dtype)
            q_second_sin = T.alloc_fragment((chunk_size, R, 1, rotary_dim), accum_dtype)
            q_first_cos = T.alloc_fragment((chunk_size, R, 1, rotary_dim), accum_dtype)
            q_second_cos = T.alloc_fragment((chunk_size, R, 1, rotary_dim), accum_dtype)

            states_shared = T.alloc_shared((N, P), accum_dtype)

            T.copy(Q, q_shared)
            T.reshape(q_shared, q_reshaped)

            T.copy(STATES, states_shared)

            angles_frag = T.alloc_shared((chunk_size, 1, 1, rotary_dim), "float32")
            T.copy(ANGLES, angles_frag)

            angles_frag_cos = T.alloc_fragment((chunk_size, 1, 1, rotary_dim), "float32")
            T.vcos(angles_frag, angles_frag_cos)
            angles_frag_sin = T.alloc_fragment((chunk_size, 1, 1, rotary_dim), "float32")
            T.vsin(angles_frag, angles_frag_sin)

            T.vmul(q_reshaped[:, :, 0:1, :], angles_frag_cos, q_first_cos)
            T.vmul(q_reshaped[:, :, 0:1, :], angles_frag_sin, q_first_sin)
            T.vmul(q_reshaped[:, :, pos_half : pos_half + 1, :], angles_frag_cos, q_second_cos)
            T.vmul(q_reshaped[:, :, pos_half : pos_half + 1, :], angles_frag_sin, q_second_sin)
            
            T.vsub(q_first_cos, q_second_sin, q_reshaped[:, :, 0:1, :])
            T.vadd(q_first_sin, q_second_cos, q_reshaped[:, :, pos_half : pos_half + 1, :])

            T.reshape(q_reshaped, q_worked)
            
            o_mimo_accum_frag = T.alloc_fragment(
                (fused_chunk_size, P), accum_dtype
            )
            
            T.gemm(
                q_worked,
                states_shared,
                o_mimo_accum_frag,
                [fused_chunk_size, N, P],
                initC=True,
            )

            T.copy(o_mimo_accum_frag, O)

    return mamba3_rotary_q_interchunk_kernel
