"""
Small TileLang NPUIR kernel for the Mamba3 intrachunk causal mask.

This isolates the nested csr_i/csr_j loop from mamba3_mimo_fwd_npu.py so it can
be optimized independently.
"""

import tilelang
import tilelang.language as T


@tilelang.jit(target="npuir")
def mamba3_intrachunk_mask(
    chunk_size: int,
    R: int,
    dtype: str = "float32",
):
    fused_chunk_size = chunk_size * R

    @T.prim_func
    def mamba3_intrachunk_mask_kernel(
        QK_INTRACHUNK: T.Tensor((fused_chunk_size, fused_chunk_size), dtype),
        SEGSUM: T.Tensor((chunk_size, chunk_size), "float32"),
        QK_INTRACHUNK_MASKED: T.Tensor(
            (fused_chunk_size, fused_chunk_size), dtype
        ),
    ):
        with T.Kernel(1, is_npu=True):
            qk_intrachunk_frag = T.alloc_fragment(
                (fused_chunk_size, fused_chunk_size), dtype
            )
            segsum_frag = T.alloc_fragment((chunk_size, 1, chunk_size, 1), "float32")
            qk_intrachunk_frag_reshaped = T.alloc_fragment((chunk_size, R, chunk_size, R), dtype)

            qk_intrachunk_masked_frag = T.alloc_fragment(
                (fused_chunk_size, fused_chunk_size), dtype
            )

            T.copy(QK_INTRACHUNK, qk_intrachunk_frag)
            T.copy(SEGSUM, segsum_frag)
            T.clear(qk_intrachunk_masked_frag)
            T.vexp(segsum_frag, segsum_frag)

            T.reshape(qk_intrachunk_frag, qk_intrachunk_frag_reshaped)

            for csr_i in T.serial(chunk_size):
                for csr_j in T.serial(chunk_size):
                    if csr_i < csr_j :
                        segsum_frag[csr_i, 0, csr_j, 0] = 0.0
            
            T.vmul(qk_intrachunk_frag_reshaped, segsum_frag, qk_intrachunk_frag_reshaped)
            T.reshape(qk_intrachunk_frag_reshaped, qk_intrachunk_masked_frag)
                        

            T.copy(qk_intrachunk_masked_frag, QK_INTRACHUNK_MASKED)

    return mamba3_intrachunk_mask_kernel
