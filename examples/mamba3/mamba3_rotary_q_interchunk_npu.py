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

    @T.prim_func
    def mamba3_rotary_q_interchunk_kernel(
        Q: T.Tensor((fused_chunk_size, N), dtype),
        ANGLES: T.Tensor((chunk_size, rotary_dim), "float32"),
        STATES: T.Tensor((N, P), accum_dtype),
        O: T.Tensor((fused_chunk_size, P), accum_dtype),
    ):
        with T.Kernel(1, is_npu=True):
            q_shared = T.alloc_shared((fused_chunk_size, N), accum_dtype)
            states_shared = T.alloc_shared((N, P), accum_dtype)

            T.copy(Q, q_shared)
            T.copy(STATES, states_shared)

            angles_frag = T.alloc_fragment((chunk_size, rotary_dim), "float32")
            T.copy(ANGLES, angles_frag)
            angles_frag_cos = T.alloc_fragment(
                (chunk_size, rotary_dim), "float32"
            )
            T.vcos(angles_frag, angles_frag_cos)
            angles_frag_sin = T.alloc_fragment(
                (chunk_size, rotary_dim), "float32"
            )
            T.vsin(angles_frag, angles_frag_sin)

            q_first_tmp = T.alloc_shared((R, rotary_dim), accum_dtype)
            q_second_tmp = T.alloc_shared((R, rotary_dim), accum_dtype)
            sincos_tmp = T.alloc_shared((R, rotary_dim), "float32")
            angle_cos_brc = T.alloc_shared((R, rotary_dim), "float32")
            angle_sin_brc = T.alloc_shared((R, rotary_dim), "float32")

            for cs in T.serial(chunk_size):
                offset = cs * R
                for r in T.serial(R):
                    T.copy(
                        angles_frag_cos[cs : cs + 1, :],
                        angle_cos_brc[r : r + 1, :],
                    )
                    T.copy(
                        angles_frag_sin[cs : cs + 1, :],
                        angle_sin_brc[r : r + 1, :],
                    )
                T.copy(q_shared[offset : offset + R, :rotary_dim], q_first_tmp)
                T.copy(
                    q_shared[offset : offset + R, N // 2 : N // 2 + rotary_dim],
                    q_second_tmp,
                )

                T.vmul(
                    angle_cos_brc,
                    q_first_tmp,
                    q_shared[offset : offset + R, :rotary_dim],
                )
                T.vmul(
                    angle_sin_brc,
                    q_second_tmp,
                    sincos_tmp,
                )
                T.vsub(
                    q_shared[offset : offset + R, :rotary_dim],
                    sincos_tmp,
                    q_shared[offset : offset + R, :rotary_dim],
                )

                T.vmul(
                    angle_sin_brc,
                    q_first_tmp,
                    q_shared[offset : offset + R, N // 2 : N // 2 + rotary_dim],
                )
                T.vmul(
                    angle_cos_brc,
                    q_second_tmp,
                    sincos_tmp,
                )
                T.vadd(
                    q_shared[offset : offset + R, N // 2 : N // 2 + rotary_dim],
                    sincos_tmp,
                    q_shared[offset : offset + R, N // 2 : N // 2 + rotary_dim],
                )

            o_mimo_accum_frag = T.alloc_fragment(
                (fused_chunk_size, P), accum_dtype
            )
            T.gemm(
                q_shared,
                states_shared,
                o_mimo_accum_frag,
                [fused_chunk_size, N, P],
                initC=True,
            )

            T.copy(o_mimo_accum_frag, O)

    return mamba3_rotary_q_interchunk_kernel
