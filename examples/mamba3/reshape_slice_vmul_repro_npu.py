"""
Minimal NPUIR repro for consuming a T.reshape slice with T.vmul.

Three paths compute the same result:
1. T.vmul directly consumes a_reshaped[:, :, 0:K].
2. a_slice is filled by T.copy from a_reshaped[:, :, 0:K].
3. a_scalar is filled by scalar reads from a_reshaped.

All paths multiply by the same B tensor broadcasted from (N, 1, K) to
(N, M, K). The scalar path is the control path; a mismatch in either slice path
points at reshape-slice consumption by T.vmul.
"""

import tilelang
import tilelang.language as T


@tilelang.jit(target="npuir")
def reshape_slice_vmul_repro(
    N: int,
    M: int,
    K: int,
    dtype: str = "float32",
):
    L = 4 * K

    @T.prim_func
    def reshape_slice_vmul_repro_kernel(
        A: T.Tensor((N * M, L), dtype),
        B: T.Tensor((N, K), dtype),
        OUT_DIRECT_SLICE_VMUL: T.Tensor((N, M, K), dtype),
        OUT_SLICE_VMUL: T.Tensor((N, M, K), dtype),
        OUT_SCALAR_VMUL: T.Tensor((N, M, K), dtype),
    ):
        with T.Kernel(1, is_npu=True):
            a_shared = T.alloc_shared((N * M, L), dtype)
            a_reshaped = T.alloc_fragment((N, M, L), dtype)
            a_slice = T.alloc_fragment((N, M, K), dtype)
            a_scalar = T.alloc_fragment((N, M, K), dtype)
            out_direct = T.alloc_fragment((N, M, K), dtype)
            out_slice = T.alloc_fragment((N, M, K), dtype)
            out_scalar = T.alloc_fragment((N, M, K), dtype)

            b_shared = T.alloc_shared((N, 1, K), dtype)
            b_broadcast = T.alloc_fragment((N, M, K), dtype)

            T.copy(A, a_shared)
            T.reshape(a_shared, a_reshaped)

            T.copy(B, b_shared)
            T.vbrc(b_shared, b_broadcast)

            T.copy(a_reshaped[:, :, 0:K], a_slice)

            for n in T.serial(N):
                for m in T.serial(M):
                    for k in T.serial(K):
                        a_scalar[n, m, k] = a_reshaped[n, m, k]

            T.vmul(a_reshaped[:, :, 0:K], b_broadcast, out_direct)
            T.vmul(a_slice, b_broadcast, out_slice)
            T.vmul(a_scalar, b_broadcast, out_scalar)

            T.copy(out_direct, OUT_DIRECT_SLICE_VMUL)
            T.copy(out_slice, OUT_SLICE_VMUL)
            T.copy(out_scalar, OUT_SCALAR_VMUL)

    return reshape_slice_vmul_repro_kernel
