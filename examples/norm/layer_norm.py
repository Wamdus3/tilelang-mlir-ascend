import tilelang
import tilelang.language as T
import torch
import os
import torch.nn.functional as F

ALIGNMENT = 256


def _align_up(n: int, alignment: int) -> int:
    return ((n + alignment - 1) // alignment) * alignment


def _layer_norm_kernel(M, N, eps, dtype):
    N_padded = _align_up(N, ALIGNMENT)
    pad_count = N_padded - N  # number of zero-padded elements per row

    @tilelang.jit(out_idx=[3], target="npuir")
    def _func(block_m):

        @T.prim_func
        def main(
            x: T.Tensor[(M, N_padded), dtype],
            weight: T.Tensor[(N_padded,), dtype],
            bias: T.Tensor[(N_padded,), dtype],
            y: T.Tensor[(M, N_padded), dtype],
        ):
            with T.Kernel(T.ceildiv(M, block_m), is_npu=True) as (pid_m, _):
                shared_buf = T.alloc_shared((block_m, N_padded), dtype)
                x_local = T.alloc_fragment((block_m, N_padded), dtype)
                x_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                acc = T.alloc_fragment((block_m, 1), "float32")
                mean_val = T.alloc_fragment((block_m, 1), "float32")
                rstd = T.alloc_fragment((block_m, 1), "float32")

                # Load input row block via shared memory
                T.copy(x[pid_m * block_m, 0], shared_buf)
                T.copy(shared_buf, x_local)

                # Cast to fp32 once — reused across all passes
                for i, j in T.Parallel(block_m, N_padded):
                    x_f32[i, j] = T.cast(x_local[i, j], "float32")

                # --- Mean reduction ---
                T.reduce_sum(x_f32, acc, dim=1)
                for i in T.Parallel(block_m):
                    mean_val[i, 0] = acc[i, 0] / float(N)

                # --- Centered variance reduction ---
                # Rewrite x_f32 in-place with (x - mean)^2.
                # Padded positions (x=0) contribute mean^2; corrected below.
                for i, j in T.Parallel(block_m, N_padded):
                    x_f32[i, j] = (x_f32[i, j] - mean_val[i, 0]) * (
                        x_f32[i, j] - mean_val[i, 0]
                    )
                T.reduce_sum(x_f32, acc, dim=1)
                for i in T.Parallel(block_m):
                    rstd[i, 0] = (
                        acc[i, 0] - float(pad_count) * mean_val[i, 0] * mean_val[i, 0]
                    ) / float(N) + eps
                T.vrsqrt(rstd, rstd)

                # --- Output: y = (x - mean) * rstd * weight + bias ---
                for i, j in T.Parallel(block_m, N_padded):
                    x_local[i, j] = (
                        T.cast(x_local[i, j], "float32") - mean_val[i, 0]
                    ) * rstd[i, 0] * T.cast(weight[j], "float32") + T.cast(
                        bias[j], "float32"
                    )

                # Write output via shared memory
                T.copy(x_local, shared_buf)
                T.copy(shared_buf, y[pid_m * block_m, 0])

        return main

    return _func


def _layer_norm_kernel_high_perf(M, N, eps, dtype):
    N_padded = _align_up(N, ALIGNMENT)
    pad_count = N_padded - N  # number of zero-padded elements per row

    @tilelang.jit(out_idx=[3], target="npuir")
    def _func_high_perf(block_m, block_n):

        @T.prim_func
        def high_perf(
            x: T.Tensor[(M, N_padded), dtype],
            weight: T.Tensor[(N_padded,), dtype],
            bias: T.Tensor[(N_padded,), dtype],
            y: T.Tensor[(M, N_padded), dtype],
        ):
            with T.Kernel(T.ceildiv(M, block_m)) as pid_m:
                x_tile = T.alloc_shared((block_m, block_n), "float32")
                w_tile = T.alloc_shared((block_n,), "float32")
                b_tile = T.alloc_shared((block_n,), "float32")
                y_tile = T.alloc_shared((block_m, block_n), "float32")

                acc = T.alloc_fragment((block_m, 1), "float32")
                mean_val = T.alloc_fragment((block_m, 1), "float32")
                var_val = T.alloc_fragment((block_m, 1), "float32")
                rstd = T.alloc_fragment((block_m, 1), "float32")

                # -------------------------------------------------
                # Pass 1: sum over full N_padded by tiles
                # -------------------------------------------------
                T.clear(acc)

                for no in T.serial(T.ceildiv(N_padded, block_n)):
                    n_start = no * block_n

                    T.copy(x[pid_m * block_m, n_start], x_tile)

                    # tile row-wise reduction, accumulate into acc
                    T.reduce_sum(x_tile, var_val, dim=1)
                    for i in T.Parallel(block_m):
                        acc[i, 0] += var_val[i, 0]

                for i in T.Parallel(block_m):
                    mean_val[i, 0] = acc[i, 0] / float(N)

                # -------------------------------------------------
                # Pass 2: squared diff sum over full N_padded by tiles
                # -------------------------------------------------
                T.clear(acc)

                for no in T.serial(T.ceildiv(N_padded, block_n)):
                    n_start = no * block_n

                    T.copy(x[pid_m * block_m, n_start], x_tile)

                    for i, j in T.Parallel(block_m, block_n):
                        y_tile[i, j] = (x_tile[i, j] - mean_val[i, 0]) * (
                            x_tile[i, j] - mean_val[i, 0]
                        )

                    T.reduce_sum(y_tile, var_val, dim=1)
                    for i in T.Parallel(block_m):
                        acc[i, 0] += var_val[i, 0]

                for i in T.Parallel(block_m):
                    var_val[i, 0] = (
                        acc[i, 0] - float(pad_count) * mean_val[i, 0] * mean_val[i, 0]
                    ) / float(N) + eps
                T.vrsqrt(var_val, rstd)

                # -------------------------------------------------
                # Pass 3: output by tiles
                # -------------------------------------------------
                for no in T.serial(T.ceildiv(N_padded, block_n)):
                    n_start = no * block_n

                    T.copy(x[pid_m * block_m, n_start], x_tile)
                    T.copy(weight[n_start], w_tile)
                    T.copy(bias[n_start], b_tile)

                    for i, j in T.Parallel(block_m, block_n):
                        y_tile[i, j] = (x_tile[i, j] - mean_val[i, 0]) * rstd[
                            i, 0
                        ] * w_tile[j] + b_tile[j]

                    T.copy(y_tile, y[pid_m * block_m, n_start])

        return high_perf

    return _func_high_perf


def layer_norm_ref(x, weight, bias, N, eps):
    return F.layer_norm(
        x.float(),
        (N,),
        weight=weight.float(),
        bias=bias.float(),
        eps=eps,
    ).to(x.dtype)


def run_test(
    M=4096,
    N=4096,
    block_m=64,
    block_n=64,
    eps=1e-5,
    dtype="float16",
    device="npu",
    atol=1e-2,
    rtol=1e-2,
):
    n_padded = _align_up(N, ALIGNMENT)

    torch_dtype = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[dtype]

    x = torch.zeros((M, n_padded), dtype=torch_dtype, device=device)
    x[:, :N] = torch.randn((M, N), dtype=torch_dtype, device=device)
    weight = torch.randn((n_padded,), dtype=torch_dtype, device=device)
    bias = torch.randn((n_padded,), dtype=torch_dtype, device=device)

    y_ref = layer_norm_ref(x, weight, bias, N, eps)
    program = _layer_norm_kernel_high_perf(M, N, eps, dtype)
    y = program(block_m, block_n)(x, weight, bias)

    torch.testing.assert_close(y.float(), y_ref.float(), atol=atol, rtol=rtol)
    print("\033[32;1mPass!\033[0m")


if __name__ == "__main__":
    os.environ["TILELANG_ASCEND_MODE"] = "Dev"
    run_test()
