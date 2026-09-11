# ruff: noqa: F841
# Preserve all measured DSL allocations and scalar bindings for the UB adapter.
# V37: FP32 zero guard before integer exponent updates; preserve manual sync.
# V36: four KV workspace slots, with disjoint per-direction cross-core flags.
# V35: full 16-row reductions; AscendC logical tile512, 7x72KiB L1, C1 N128/K288/K0=96, C2 N128/K256/K0=128.
# V32: paired jump copies on V28 AscendC UB; requires jump build + isolated UB/atomic adapter.
# Copyright (c) Huawei Technologies Co., Ltd. 2025.
# Experimental block512 for H32/D512/Rope64 and topk divisible by512.
# B=4 with uniform S, T=4*S; each call is JIT-specialized for its T.
# Each GM slot is released after its actual last DMA read; token stages overlap.
# V25: AscendC AMLA int32 exponent update + FP32 atomic Fixpipe; one kernel.
# Requires TileLang explicit atomic mode and an atomic-enabled AscendNPU-IR build.
import math
import torch

# Import tilelang modules for NPU kernel development
import tilelang
import tilelang.language as T

from tilelang.utils import NPUUtils


@tilelang.jit(
    target="npuir",
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION: False,
        tilelang.PassConfigKey.NPUIR_ENABLE_AUTO_MULTI_BUFFER: False,
        tilelang.PassConfigKey.NPUIR_DISABLE_HIVM_AUTO_INJECT_SYNC: True,
    },
)
def sparse_mla_fwd_pa_bsnd_kernel(
    batch,
    total_tokens,
    num_pages,
    max_blocks,
    heads,
    dim,
    tail_dim,
    top_k,
    page_size=128,
    sm_scale=None,
    block=64,
    sparse_mode=3,
    multi_ws_kv=4,
    multi_ws_s=2,
    multi_ws_p=2,
    multi_ws_o=2,
):
    # Uniform request lengths: ActualQLengths is a checked caller precondition.
    # T specializes the JIT; request mapping uses S=T/B without a GM lookup.
    assert batch == 4
    assert total_tokens > 0 and total_tokens % batch == 0
    assert page_size == 128
    assert sparse_mode in (0, 3)
    assert 0 < heads <= 32 and heads % 16 == 0
    assert block > 0 and block % 8 == 0 and block <= min(dim, 512)
    assert top_k > 0 and top_k % block == 0
    assert 0 < tail_dim <= 64 and tail_dim % 16 == 0
    assert num_pages > 0 and 0 < max_blocks <= 1565
    assert 0 < dim <= 512
    assert (multi_ws_kv, multi_ws_s, multi_ws_p, multi_ws_o) == (4, 2, 2, 2)
    assert dim == tilelang.math.next_power_of_2(dim), (
        f"haven't check padding correctness yet, dim={dim}"
    )

    # Set softmax scale if not provided
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5
    assert math.isfinite(sm_scale) and sm_scale > 0

    multi_l1_kv = 2
    multi_ub_kv = 2
    gather_rows = 32
    assert block == 512 and dim == 512 and heads == 32
    multi_ub_inner_cross = 2

    # ws_kv: V0S -> C1L -> ...
    flag_base_V0S_C1L_kv = 0

    # ws_s: C1S -> V1L -> ...
    flag_base_C1S_V1L_s = 0

    # ws_p: V1S -> C2L -> ...
    flag_base_V1S_C2L_p = 4

    # ws_o: C2S -> V2L -> ...
    flag_base_C2S_V2L_o = flag_base_C1S_V1L_s + multi_ws_s

    # Reverse acknowledgements prevent a producer from reusing a GM slot
    # before the consumer-side DMA has completed.
    flag_base_V1L_C1S_s_free = 10
    flag_base_V2L_C2S_o_free = flag_base_V1L_C1S_s_free + multi_ws_s
    flag_base_C1L_V0S_kv_free = 10
    flag_base_C2L_V1S_p_free = 4

    num_logic_kernels = total_tokens
    num_kernels = min(NPUUtils.get().get_aicore_num(), num_logic_kernels)
    assert num_kernels > 0
    tokens_per_request = total_tokens // batch
    num_top_k_blocks = (top_k - 1) // block + 1

    # Define data types
    indices_dtype = "int32"
    dtype = "bfloat16"
    accum_dtype = "float"

    # Set block size for head dimension
    heads_half = heads // 2

    # Calculate half block sizes for vector operations
    block_half = block // 2

    block_dim_share = max(block, dim)

    # Cube PIPE_M <-> PIPE_MTE2 events.
    cube_event_kv_free_base = 0
    cube_event_q_free = 2
    cube_event_p_free = 3
    cube_event_c1_ready = 4
    cube_event_c2_ready = 5
    cube_event_l0_ready = 0
    cube_event_l0_free = 1

    # Vector PIPE_V <-> PIPE_MTE2 events.
    vector_event_v0_zero_base = 0
    vector_event_v1_loaded = 4
    vector_event_v1_input_free = 5
    vector_event_merge_loaded = 6
    vector_event_merge_input_free = 7

    # Vector PIPE_V <-> PIPE_MTE3 events.
    vector_event_v0_free_base = 1
    vector_event_v1_store_ready = 4
    vector_event_v1_store_free = 5
    vector_event_output_ready = 6
    vector_event_output_free = 7

    # Vector PIPE_MTE2 <-> PIPE_MTE3 and PIPE_S events.
    vector_event_v0_ready_base = 0
    vector_event_indices_scalar_base = 0
    vector_event_v1_metadata_scalar = 2
    vector_event_v1_length_scalar = 3
    vector_event_stats_free_base = 2

    @T.macro
    def init_cube_flags():
        with T.rs("PIPE_MTE2"):
            for slot in T.serial(multi_ws_kv):
                T.sync_block_set(flag_base_C1L_V0S_kv_free + slot)
            for slot in T.serial(multi_ws_p):
                T.sync_block_set(flag_base_C2L_V1S_p_free + slot)
        with T.rs("PIPE_MTE1"):
            for slot in T.serial(7):
                T.set_flag("PIPE_MTE2", slot)
        with T.rs("PIPE_FIX"):
            for slot in T.serial(2):
                T.set_flag("PIPE_M", slot)
            T.sync_block_set(8)

    @T.macro
    def clear_cube_flags():
        with T.rs("PIPE_FIX"):
            for slot in T.serial(2):
                T.sync_block_wait(flag_base_V1L_C1S_s_free + slot)
        with T.rs("PIPE_MTE2"):
            for slot in T.serial(7):
                T.wait_flag("PIPE_MTE1", slot)
        with T.rs("PIPE_M"):
            for slot in T.serial(2):
                T.wait_flag("PIPE_FIX", slot)

    @T.macro
    def C1Tile(
        QNope,
        QRope,
        workspace_nope,
        workspace_rope,
        workspace_s,
        l1_qp,
        l1_kv,
        l0_c,
        kernel_id,
        task_id,
    ):
        token = task_id // num_top_k_blocks * num_kernels + kernel_id
        # AscendC Q/P ring: first C1 uses slots 0/1; steady C1 uses 2/3,
        # interleaved C2 uses 0/1. Each physical slot is 72 KiB.
        qp_base = T.min(task_id, 1) * 2
        with T.rs("PIPE_MTE2"):
            T.sync_block_wait(flag_base_V0S_C1L_kv + task_id % multi_ws_kv)
        with T.rs("PIPE_FIX"):
            T.sync_block_wait(flag_base_V1L_C1S_s_free + task_id % 2)
        for n1 in T.serial(4):
            cslot = n1 % 2
            with T.rs("PIPE_M"):
                T.wait_flag("PIPE_FIX", cslot)
            for k1 in T.serial(2):
                ka = qp_base + k1
                # Eight KV loads per C1/C2 phase; ring continues across stages.
                kb = ((T.max(task_id * 2 - 1, 0)) * 8 + n1 * 2 + k1) % 3
                with T.rs("PIPE_MTE2"):
                    T.wait_flag("PIPE_MTE1", kb + 4)
                    if n1 == 0:
                        T.wait_flag("PIPE_MTE1", ka)
                    if k1 == 0:
                        T.copy(
                            workspace_nope[
                                kernel_id,
                                task_id % multi_ws_kv,
                                n1 * 128 : (n1 + 1) * 128,
                                :256,
                            ],
                            l1_kv[kb, :, :256],
                        )
                        T.copy(
                            workspace_rope[
                                kernel_id,
                                task_id % multi_ws_kv,
                                n1 * 128 : (n1 + 1) * 128,
                                :32,
                            ],
                            l1_kv[kb, :, 256:288],
                        )
                        if n1 == 0:
                            T.copy(QNope[token, :, :256], l1_qp[ka, :heads, :256])
                            T.copy(QRope[token, :, :32], l1_qp[ka, :heads, 256:288])
                    else:
                        T.copy(
                            workspace_rope[
                                kernel_id,
                                task_id % multi_ws_kv,
                                n1 * 128 : (n1 + 1) * 128,
                                32:64,
                            ],
                            l1_kv[kb, :, :32],
                        )
                        T.copy(
                            workspace_nope[
                                kernel_id,
                                task_id % multi_ws_kv,
                                n1 * 128 : (n1 + 1) * 128,
                                256:512,
                            ],
                            l1_kv[kb, :, 32:288],
                        )
                        if n1 == 0:
                            T.copy(QRope[token, :, 32:64], l1_qp[ka, :heads, :32])
                            T.copy(QNope[token, :, 256:512], l1_qp[ka, :heads, 32:288])
                    T.set_flag("PIPE_MTE1", kb + 4)
                    if n1 == 0:
                        T.set_flag("PIPE_MTE1", ka)
                with T.rs("PIPE_MTE1"):
                    T.wait_flag("PIPE_MTE2", kb + 4)
                    if n1 == 0:
                        T.wait_flag("PIPE_MTE2", ka)
                for k0 in T.serial(3):
                    with T.rs("PIPE_M"):
                        T.gemm(
                            l1_qp[ka, :heads, k0 * 96 : (k0 + 1) * 96],
                            l1_kv[kb, :, k0 * 96 : (k0 + 1) * 96],
                            l0_c[cslot, :heads, :128],
                            size=[heads, 96, 128],
                            initC=(k1 * 3 + k0 == 0),
                            b_transpose=True,
                        )
                with T.rs("PIPE_MTE1"):
                    T.set_flag("PIPE_MTE2", kb + 4)
                    if n1 == 3:
                        T.set_flag("PIPE_MTE2", ka)
            with T.rs("PIPE_M"):
                T.set_flag("PIPE_FIX", cslot)
            with T.rs("PIPE_FIX"):
                T.wait_flag("PIPE_M", cslot)
                T.copy(
                    l0_c[cslot, :heads, :128],
                    workspace_s[kernel_id, task_id % 2, :, n1 * 128 : (n1 + 1) * 128],
                )
                T.set_flag("PIPE_M", cslot)
        with T.rs("PIPE_FIX"):
            T.sync_block_set(flag_base_C1S_V1L_s + task_id % 2)
        # KV workspace remains owned until C2 finishes reloading V.

    @T.macro
    def C2Tile(
        workspace_p, workspace_nope, workspace_o, l1_qp, l1_kv, l0_c, kernel_id, task_id
    ):
        with T.rs("PIPE_MTE2"):
            T.sync_block_wait(flag_base_V1S_C2L_p + task_id % 2)
        for n1 in T.serial(4):
            cslot = n1 % 2
            with T.rs("PIPE_M"):
                T.wait_flag("PIPE_FIX", cslot)
            for k1 in T.serial(2):
                ka = k1
                kb = ((task_id * 2 + 2) * 8 + n1 * 2 + k1) % 3
                with T.rs("PIPE_MTE2"):
                    T.wait_flag("PIPE_MTE1", kb + 4)
                    # Two 128x128 slabs occupy 64 KiB of the 72 KiB KV slot.
                    T.copy(
                        workspace_nope[
                            kernel_id,
                            task_id % multi_ws_kv,
                            k1 * 256 : k1 * 256 + 128,
                            n1 * 128 : (n1 + 1) * 128,
                        ],
                        l1_kv[kb, :, :128],
                    )
                    T.copy(
                        workspace_nope[
                            kernel_id,
                            task_id % multi_ws_kv,
                            k1 * 256 + 128 : (k1 + 1) * 256,
                            n1 * 128 : (n1 + 1) * 128,
                        ],
                        l1_kv[kb, :, 128:256],
                    )
                    if n1 == 0:
                        T.wait_flag("PIPE_MTE1", ka)
                        T.copy(
                            workspace_p[
                                kernel_id, task_id % 2, :, k1 * 256 : (k1 + 1) * 256
                            ],
                            l1_qp[ka, :heads, :256],
                        )
                        T.set_flag("PIPE_MTE1", ka)
                    T.set_flag("PIPE_MTE1", kb + 4)
                    if n1 * 2 + k1 == 7:
                        T.sync_block_set(
                            flag_base_C1L_V0S_kv_free + task_id % multi_ws_kv
                        )
                with T.rs("PIPE_MTE1"):
                    T.wait_flag("PIPE_MTE2", kb + 4)
                    if n1 == 0:
                        T.wait_flag("PIPE_MTE2", ka)
                for k0 in T.serial(2):
                    with T.rs("PIPE_M"):
                        T.gemm(
                            l1_qp[ka, :heads, k0 * 128 : (k0 + 1) * 128],
                            l1_kv[kb, :, k0 * 128 : (k0 + 1) * 128],
                            l0_c[cslot, :heads, :128],
                            size=[heads, 128, 128],
                            initC=(k1 * 2 + k0 == 0),
                        )
                with T.rs("PIPE_MTE1"):
                    T.set_flag("PIPE_MTE2", kb + 4)
                    if n1 == 3:
                        T.set_flag("PIPE_MTE2", ka)
            with T.rs("PIPE_M"):
                T.set_flag("PIPE_FIX", cslot)
            with T.rs("PIPE_FIX"):
                T.wait_flag("PIPE_M", cslot)
                if n1 == 0:
                    T.sync_block_wait(6)
                # Explicit state applies only to the C2 accumulation store.
                T.pipe_barrier("PIPE_FIX")
                T.set_atomic_add("float32")
                T.copy(
                    l0_c[cslot, :heads, :128],
                    workspace_o[
                        kernel_id * 2 * heads : kernel_id * 2 * heads + heads,
                        n1 * 128 : (n1 + 1) * 128,
                    ],
                )
                T.pipe_barrier("PIPE_FIX")
                T.set_atomic_none("float32")
                T.set_flag("PIPE_M", cslot)
        with T.rs("PIPE_MTE2"):
            T.sync_block_set(flag_base_C2L_V1S_p_free + task_id % 2)
        with T.rs("PIPE_FIX"):
            T.sync_block_set(8)
            if task_id % num_top_k_blocks == num_top_k_blocks - 1:
                T.sync_block_set(9)

    @T.macro
    def init_vector_flags():
        with T.rs("PIPE_MTE2"):
            for slot in T.serial(multi_ws_s):
                T.sync_block_set(flag_base_V1L_C1S_s_free + slot)
        # Seed only free events. Ready events are produced by actual work.
        with T.rs("PIPE_MTE3"):
            for slot in T.serial(multi_ub_kv):
                T.set_flag("PIPE_V", vector_event_v0_free_base + slot)
            T.set_flag("PIPE_V", vector_event_v1_store_free)
            T.set_flag("PIPE_V", vector_event_output_free)
        with T.rs("PIPE_V"):
            T.set_flag("PIPE_MTE2", vector_event_v1_input_free)

    @T.macro
    def clear_vector_flags():
        with T.rs("PIPE_MTE3"):
            T.sync_block_wait(8)
            for slot in T.serial(multi_ws_p):
                T.sync_block_wait(flag_base_C2L_V1S_p_free + slot)
        with T.rs("PIPE_MTE3"):
            for slot in T.serial(multi_ws_kv):
                T.sync_block_wait(flag_base_C1L_V0S_kv_free + slot)
        with T.rs("PIPE_V"):
            T.wait_flag("PIPE_MTE3", vector_event_output_free)
            T.wait_flag("PIPE_MTE3", vector_event_v1_store_free)
            for slot in T.serial(multi_ub_kv):
                T.wait_flag("PIPE_MTE3", vector_event_v0_free_base + slot)
        with T.rs("PIPE_MTE2"):
            T.wait_flag("PIPE_V", vector_event_v1_input_free)

    @T.macro
    def GatherChunk(
        KVNope,
        KRope,
        workspace_nope,
        workspace_rope,
        ub_indices,
        ub_block_table,
        ub_actual_kv,
        ub_nope,
        ub_rope,
        kernel_id,
        vid,
        task_id,
        chunk_id,
    ):
        tile = task_id % num_top_k_blocks
        token = task_id // num_top_k_blocks * num_kernels + kernel_id
        query = token % tokens_per_request
        tail = T.min(top_k - tile * block, block)
        half = (tail + 1) // 2
        start = vid * half
        count = T.min(gather_rows, half - (tail % 2) * vid - chunk_id * gather_rows)
        slot = chunk_id % 2
        inner = task_id % multi_ub_inner_cross
        with T.rs("PIPE_V"):
            T.wait_flag("PIPE_MTE3", vector_event_v0_free_base + slot)
            T.vbrc(T.cast(0, dtype), ub_nope)
            T.vbrc(T.cast(0, dtype), ub_rope)
            T.pipe_barrier("PIPE_V")
            T.set_flag("PIPE_MTE2", vector_event_v0_zero_base + slot)
        with T.rs("PIPE_MTE2"):
            T.wait_flag("PIPE_V", vector_event_v0_zero_base + slot)
            # One SSA definition: an assignment inside the static if would be
            # scoped there, leaving pair validity bound to the old length.
            length = ub_actual_kv[inner, 0] - (tokens_per_request - query - 1) * (
                sparse_mode // 3
            )
            for pair in T.serial(T.max(count, 0) // 2):
                i = pair * 2
                # Read once, widen before arithmetic, and share with KV/RoPE.
                index0 = ub_indices[inner, 0, start + chunk_id * gather_rows + i]
                index1 = ub_indices[inner, 0, start + chunk_id * gather_rows + i + 1]
                valid0 = T.cast(index0 >= 0, "int32") * T.cast(index0 < length, "int32")
                valid1 = T.cast(index1 >= 0, "int32") * T.cast(index1 < length, "int32")
                if valid0 + valid1 == 2:
                    page0 = T.cast(ub_block_table[index0 // page_size], "int64")
                    page1 = T.cast(ub_block_table[index1 // page_size], "int64")
                    lane0 = T.cast(index0 % page_size, "int64")
                    lane1 = T.cast(index1 % page_size, "int64")
                    token0 = page0 * page_size + lane0
                    token1 = page1 * page_size + lane1
                    # jump sorts both KV and RoPE by the same physical address.
                    # Both mask entries are valid, so the paired permutation is
                    # shared by QK, softmax and PV, preserving attention semantics.
                    T.copy(
                        KVNope[page0, lane0, 0, 0],
                        ub_nope[i, 0],
                        size=[2, dim],
                        jump=(token1 - token0) * T.int64(dim),
                    )
                    T.copy(
                        KRope[page0, lane0, 0, 0],
                        ub_rope[i, 0],
                        size=[2, tail_dim],
                        jump=(token1 - token0) * T.int64(tail_dim),
                    )
                else:
                    # Preserve positions of invalid/causally masked holes. Their
                    # rows retain the zeros written before this gather stage.
                    if valid0 != 0:
                        page0 = ub_block_table[index0 // page_size]
                        T.copy(
                            KVNope[page0, index0 % page_size, 0, :dim], ub_nope[i, :]
                        )
                        T.copy(
                            KRope[page0, index0 % page_size, 0, :tail_dim],
                            ub_rope[i, :],
                        )
                    if valid1 != 0:
                        page1 = ub_block_table[index1 // page_size]
                        T.copy(
                            KVNope[page1, index1 % page_size, 0, :dim],
                            ub_nope[i + 1, :],
                        )
                        T.copy(
                            KRope[page1, index1 % page_size, 0, :tail_dim],
                            ub_rope[i + 1, :],
                        )
            if count % 2 != 0:
                i = count - 1
                index = ub_indices[inner, 0, start + chunk_id * gather_rows + i]
                if index >= 0 and index < length:
                    physical = ub_block_table[index // page_size]
                    offset = index % page_size
                    T.copy(KVNope[physical, offset, 0, :dim], ub_nope[i, :])
                    T.copy(KRope[physical, offset, 0, :tail_dim], ub_rope[i, :])
            T.set_flag("PIPE_MTE3", vector_event_v0_ready_base + slot)
        with T.rs("PIPE_MTE3"):
            T.wait_flag("PIPE_MTE2", vector_event_v0_ready_base + slot)
            if count > 0:
                T.copy(
                    ub_nope[:count, :],
                    workspace_nope[
                        kernel_id,
                        task_id % multi_ws_kv,
                        start + chunk_id * gather_rows : start
                        + chunk_id * gather_rows
                        + count,
                        :,
                    ],
                )
                T.copy(
                    ub_rope[:count, :],
                    workspace_rope[
                        kernel_id,
                        task_id % multi_ws_kv,
                        start + chunk_id * gather_rows : start
                        + chunk_id * gather_rows
                        + count,
                        :,
                    ],
                )
            T.set_flag("PIPE_V", vector_event_v0_free_base + slot)

    @T.macro
    def MergeKV(
        KVNope,
        KRope,
        BlockTable,
        ActualKVLengths,
        Indices,
        workspace_nope,
        workspace_rope,
        ub_indices,
        ub_block_table_even,
        ub_actual_kv,
        ub_nope_even,
        ub_nope_odd,
        ub_rope_even,
        ub_rope_odd,
        kernel_id,
        vid,
        task_id,
    ):
        token = task_id // num_top_k_blocks * num_kernels + kernel_id
        request = token // tokens_per_request
        tile = task_id % num_top_k_blocks
        inner = task_id % multi_ub_inner_cross
        # inputBuff1 is shared by KV gather, scores and final O input.
        # Transfer its free token to V before either ping-pong slot is zeroed.
        with T.rs("PIPE_MTE2"):
            T.wait_flag("PIPE_V", vector_event_v1_input_free)
            T.copy(
                Indices[token, 0, tile * block : (tile + 1) * block],
                ub_indices[inner, 0, :],
            )
            T.copy(BlockTable[request, :max_blocks], ub_block_table_even)
            T.copy(ActualKVLengths[request : request + 1], ub_actual_kv[inner, :])
            T.set_flag("PIPE_S", vector_event_indices_scalar_base + inner)
            T.set_flag("PIPE_V", 3)
        with T.rs("PIPE_S"):
            T.wait_flag("PIPE_MTE2", vector_event_indices_scalar_base + inner)
        with T.rs("PIPE_V"):
            T.wait_flag("PIPE_MTE2", 3)
        with T.rs("PIPE_MTE3"):
            T.sync_block_wait(flag_base_C1L_V0S_kv_free + task_id % multi_ws_kv)
        for chunk_id in T.serial(T.ceildiv(block_half, gather_rows)):
            if chunk_id % 2 == 0:
                GatherChunk(
                    KVNope,
                    KRope,
                    workspace_nope,
                    workspace_rope,
                    ub_indices,
                    ub_block_table_even,
                    ub_actual_kv,
                    ub_nope_even,
                    ub_rope_even,
                    kernel_id,
                    vid,
                    task_id,
                    chunk_id,
                )
            else:
                GatherChunk(
                    KVNope,
                    KRope,
                    workspace_nope,
                    workspace_rope,
                    ub_indices,
                    ub_block_table_even,
                    ub_actual_kv,
                    ub_nope_odd,
                    ub_rope_odd,
                    kernel_id,
                    vid,
                    task_id,
                    chunk_id,
                )
        with T.rs("PIPE_MTE3"):
            T.sync_block_set(flag_base_V0S_C1L_kv + task_id % multi_ws_kv)
            T.set_flag("PIPE_V", 3)
        with T.rs("PIPE_V"):
            # Both gather stores finish before score/final-input aliases reuse UB.
            T.wait_flag("PIPE_MTE3", 3)
            T.set_flag("PIPE_MTE2", vector_event_v1_input_free)

    @T.macro
    def V1L(
        ActualKVLengths,
        Indices,
        workspace_s,
        ub_v1_indices,
        ub_v1_actual_kv,
        ub_cross_kernel_32,
        kernel_id,
        vid,
        task_id,
    ):
        sub_offset = vid * heads_half
        local_id = task_id // num_top_k_blocks
        block_i_id = task_id % num_top_k_blocks
        logic_kernel_id = local_id * num_kernels + kernel_id
        request_id = logic_kernel_id // tokens_per_request
        block_i_offset = block_i_id * block

        # Load attention scores from workspace
        with T.rs("PIPE_MTE2"):
            T.wait_flag("PIPE_V", vector_event_v1_input_free)
            flag_C1S_V1L_s = flag_base_C1S_V1L_s + task_id % multi_ws_s
            T.sync_block_wait(flag_C1S_V1L_s)
            T.copy(
                workspace_s[
                    kernel_id,
                    task_id % multi_ws_s,
                    sub_offset : sub_offset + heads_half,
                    :block,
                ],
                ub_cross_kernel_32[:heads_half, :block],
            )
            T.copy(
                Indices[
                    logic_kernel_id,
                    0,
                    block_i_offset : block_i_offset + block,
                ],
                ub_v1_indices,
            )
            T.copy(
                ActualKVLengths[request_id : request_id + 1],
                ub_v1_actual_kv[0, :],
            )
            T.sync_block_set(flag_base_V1L_C1S_s_free + task_id % multi_ws_s)
            T.set_flag("PIPE_V", vector_event_v1_loaded)
            T.set_flag("PIPE_S", vector_event_v1_metadata_scalar)
        with T.rs("PIPE_S"):
            T.wait_flag("PIPE_MTE2", vector_event_v1_metadata_scalar)

    @T.macro
    def V1P(
        ub_running_max,
        ub_running_sum,
        ub_previous_max,
        ub_scale,
        ub_tile_scale,
        ub_n,
        ub_prev_n,
        ub_cof,
        ub_prev_cof,
        ub_eps,
        ub_shat,
        ub_shat16,
        ub_delta,
        ub_delta_i32,
        ub_update,
        ub_safe_sum,
        ub_one,
        ub_sum_nonzero,
        ub_tile_max,
        ub_tile_sum,
        ub_v1_indices,
        ub_v1_actual_kv,
        ub_v1_indices_fp32,
        ub_v1_actual_kv_fp32,
        ub_v1_lengths_fp32,
        ub_cross_kernel_16,
        ub_cross_kernel_32,
        ub_valid_nonnegative,
        ub_valid_below_length,
        ub_valid_columns,
        ub_valid_columns_fp32,
        ub_valid_scores_fp32,
        ub_valid_scores,
        ub_invalid_scores,
        kernel_id,
        vid,
        task_id,
    ):
        acc_s_scale = sm_scale
        value_zero = 0
        value_finite_min = -1.0e30

        local_id = task_id // num_top_k_blocks
        logic_kernel_id = local_id * num_kernels + kernel_id
        local_query_id = logic_kernel_id % tokens_per_request
        stats_offset = (task_id % 2) * heads_half

        with T.rs("PIPE_V"):
            T.wait_flag("PIPE_MTE2", vector_event_v1_loaded)
            T.wait_flag("PIPE_MTE3", vector_event_v1_store_free)
            T.vmul(
                ub_cross_kernel_32[:, :block],
                acc_s_scale,
                ub_cross_kernel_32[:, :block],
            )
            T.pipe_barrier("PIPE_V")

            # CANN 9 lowers int32 vcmp to scalar instructions. Compare on V;
            # explicitly synchronize any scalar helper for the length broadcast.
            # Valid page-table indices fit exactly in FP32 (capacity <= 200320).
            T.vcast(ub_v1_indices, ub_v1_indices_fp32)
            T.pipe_barrier("PIPE_V")
            T.vcast(ub_v1_actual_kv, ub_v1_actual_kv_fp32)
            T.pipe_barrier("PIPE_V")
            T.set_flag("PIPE_S", vector_event_v1_length_scalar)
        with T.rs("PIPE_S"):
            T.wait_flag("PIPE_V", vector_event_v1_length_scalar)
            if sparse_mode == 3:
                # Use scalar arithmetic in UB: the vector scalar API only
                # accepts literals, whereas this offset depends on the token.
                ub_v1_actual_kv_fp32[0, 0] = ub_v1_actual_kv_fp32[0, 0] - T.cast(
                    tokens_per_request - local_query_id - 1, accum_dtype
                )
            T.set_flag("PIPE_V", vector_event_v1_length_scalar)
        with T.rs("PIPE_V"):
            T.wait_flag("PIPE_S", vector_event_v1_length_scalar)
            T.vbrc(ub_v1_actual_kv_fp32, ub_v1_lengths_fp32)
            T.pipe_barrier("PIPE_V")

            T.vcmp(
                ub_v1_indices_fp32,
                value_zero,
                ub_valid_nonnegative,
                "ge",
            )
            T.pipe_barrier("PIPE_V")
            T.vcmp(
                ub_v1_indices_fp32,
                ub_v1_lengths_fp32,
                ub_valid_below_length,
                "lt",
            )
            T.pipe_barrier("PIPE_V")
            T.vand(
                ub_valid_nonnegative,
                ub_valid_below_length,
                ub_valid_columns,
            )
            T.pipe_barrier("PIPE_V")
            T.vcast(ub_valid_columns, ub_valid_columns_fp32)
            T.pipe_barrier("PIPE_V")
            T.vbrc(ub_valid_columns_fp32, ub_valid_scores_fp32)
            T.pipe_barrier("PIPE_V")
            T.vcmp(
                ub_valid_scores_fp32,
                value_zero,
                ub_valid_scores,
                "gt",
            )
            T.pipe_barrier("PIPE_V")
            T.vbrc(value_finite_min, ub_invalid_scores)
            T.pipe_barrier("PIPE_V")
            T.vselect(
                ub_valid_scores,
                ub_cross_kernel_32[:, :block],
                ub_invalid_scores,
                ub_cross_kernel_32[:, :block],
            )
            T.pipe_barrier("PIPE_V")

            # V35: one 16-row reduction; split input pools bound scratch to 16 KiB.
            T.reduce(
                ub_cross_kernel_32[:heads_half, :block],
                ub_tile_max[stats_offset : stats_offset + heads_half, :],
                dims=[1],
                reduce_mode="max",
            )
            T.pipe_barrier("PIPE_V")

            if task_id % num_top_k_blocks == 0:
                T.vbrc(value_finite_min, ub_running_max)
                T.vbrc(value_zero, ub_running_sum)
                T.vbrc(value_zero, ub_prev_n)
                T.vbrc(T.cast(1.0, "float"), ub_prev_cof)
                T.pipe_barrier("PIPE_V")
            T.vadd(ub_running_max, 0.0, ub_previous_max)
            T.pipe_barrier("PIPE_V")
            T.vmax(
                ub_running_max,
                ub_tile_max[stats_offset : stats_offset + heads_half, :],
                ub_running_max,
            )
            T.pipe_barrier("PIPE_V")
            T.vsub(ub_previous_max, ub_running_max, ub_scale)
            T.pipe_barrier("PIPE_V")
            T.vexp(ub_scale, ub_scale)
            T.pipe_barrier("PIPE_V")
            T.vmul(ub_running_sum, ub_scale, ub_running_sum)
            T.pipe_barrier("PIPE_V")
            T.vsub(
                ub_cross_kernel_32[:, :block],
                ub_running_max,
                ub_cross_kernel_32[:, :block],
            )
            T.pipe_barrier("PIPE_V")
            T.vexp(ub_cross_kernel_32[:, :block], ub_cross_kernel_32[:, :block])
            T.pipe_barrier("PIPE_V")
            T.vbrc(value_zero, ub_invalid_scores)
            T.pipe_barrier("PIPE_V")
            T.vselect(
                ub_valid_scores,
                ub_cross_kernel_32[:heads_half, :block],
                ub_invalid_scores,
                ub_cross_kernel_32[:heads_half, :block],
            )
            T.pipe_barrier("PIPE_V")
            T.reduce(
                ub_cross_kernel_32[:heads_half, :block],
                ub_tile_sum[stats_offset : stats_offset + heads_half, :],
                dims=[1],
                reduce_mode="sum",
            )
            T.pipe_barrier("PIPE_V")
            T.vadd(
                ub_running_sum,
                ub_tile_sum[stats_offset : stats_offset + heads_half, :],
                ub_running_sum,
            )
            T.pipe_barrier("PIPE_V")
            # Empty rows use neutral exponent state, preserving the public M/L.
            T.vcmp(ub_running_sum, T.cast(0.0, "float"), ub_sum_nonzero, "gt")
            T.vbrc(T.cast(0.0, "float"), ub_one)
            T.pipe_barrier("PIPE_V")
            T.vselect(ub_sum_nonzero, ub_running_max, ub_one, ub_n)
            T.pipe_barrier("PIPE_V")
            T.vmul(ub_n, -1.4426950408889634, ub_n)
            T.pipe_barrier("PIPE_V")
            T.vcast(ub_n, ub_delta_i32, round_mode="rint")
            T.pipe_barrier("PIPE_V")
            T.vcast(ub_delta_i32, ub_n)
            T.pipe_barrier("PIPE_V")
            T.vsub(ub_n, ub_prev_n, ub_delta)
            T.vselect(ub_sum_nonzero, ub_running_max, ub_one, ub_shat)
            T.pipe_barrier("PIPE_V")
            T.vadd(ub_n, 0.0, ub_prev_n)
            T.vmul(ub_shat, 1.4426950408889634, ub_shat)
            T.pipe_barrier("PIPE_V")
            T.vadd(ub_shat, ub_n, ub_shat)
            T.pipe_barrier("PIPE_V")
            T.vmul(ub_shat, 0.6931471805599453, ub_shat)
            T.pipe_barrier("PIPE_V")
            T.vexp(ub_shat, ub_shat)
            T.pipe_barrier("PIPE_V")
            T.vcast(ub_shat, ub_shat16, round_mode="rint")
            T.pipe_barrier("PIPE_V")
            T.vcast(ub_shat16, ub_tile_scale)
            T.pipe_barrier("PIPE_V")
            T.vdiv(ub_shat, ub_tile_scale, ub_cof)
            T.vmul(
                ub_cross_kernel_32[:, :block],
                ub_tile_scale,
                ub_cross_kernel_32[:, :block],
            )
            T.pipe_barrier("PIPE_V")
            T.vdiv(ub_prev_cof, ub_cof, ub_eps)
            T.vmul(ub_running_sum, ub_tile_scale, ub_safe_sum)
            T.pipe_barrier("PIPE_V")
            T.vadd(ub_cof, 0.0, ub_prev_cof)
            T.vadd(ub_eps, -1.0, ub_eps)
            T.pipe_barrier("PIPE_V")
            T.vmul(ub_eps, 1.5, ub_eps)
            T.vmax(ub_delta, -30.0, ub_delta)
            T.pipe_barrier("PIPE_V")
            T.vadd(ub_eps, 0.000001, ub_eps)
            T.pipe_barrier("PIPE_V")
            T.vadd(ub_delta, ub_eps, ub_delta)
            T.pipe_barrier("PIPE_V")
            T.vmul(ub_delta, 8388608.0, ub_delta)
            T.pipe_barrier("PIPE_V")
            T.vcast(ub_delta, ub_delta_i32, round_mode="rint")
            T.pipe_barrier("PIPE_V")
            T.vcast(
                ub_cross_kernel_32[:, :block],
                ub_cross_kernel_16[:, :block],
                round_mode="rint",
            )
            T.pipe_barrier("PIPE_V")
            T.set_flag("PIPE_MTE3", vector_event_v1_store_ready)
            T.set_flag("PIPE_MTE2", vector_event_v1_input_free)

    @T.macro
    def V1S(
        workspace_p,
        workspace_o_i32,
        workspace_o,
        ub_bias_chunk,
        ub_cross_kernel_16,
        ub_update,
        ub_update_chunk,
        ub_delta_i32,
        kernel_id,
        vid,
        task_id,
    ):
        row = vid * heads_half
        with T.rs("PIPE_MTE3"):
            T.wait_flag("PIPE_V", vector_event_v1_store_ready)
            T.sync_block_wait(flag_base_C2L_V1S_p_free + task_id % multi_ws_p)
            T.copy(
                ub_cross_kernel_16[:, :block],
                workspace_p[kernel_id, task_id % multi_ws_p, row : row + heads_half, :],
            )
            T.sync_block_set(flag_base_V1S_C2L_p + task_id % multi_ws_p)
            T.set_flag("PIPE_V", vector_event_v1_store_free)
        with T.rs("PIPE_V"):
            # P and integer updates reuse outputBuff1; the FP32 guard reuses tmp1.
            T.wait_flag("PIPE_MTE3", vector_event_v1_store_free)
            if task_id % num_top_k_blocks == 0:
                T.vbrc(T.cast(394264576, "int32"), ub_update)
            else:
                # Renew AscendC's 2^-80 guard after C2 cancellation to exact zero.
                T.vbrc(T.cast(8.271806125530277e-25, "float32"), ub_bias_chunk)
            T.pipe_barrier("PIPE_V")
            T.set_flag("PIPE_MTE3", vector_event_v1_store_ready)
        with T.rs("PIPE_MTE3"):
            T.wait_flag("PIPE_V", vector_event_v1_store_ready)
            T.sync_block_wait(8)
            if task_id % num_top_k_blocks == 0:
                T.copy(
                    ub_update,
                    workspace_o_i32[
                        kernel_id * 2 * heads + row : kernel_id * 2 * heads
                        + row
                        + heads_half,
                        :,
                    ],
                )
            else:
                T.pipe_barrier("PIPE_MTE3")
                T.set_atomic_add("float32")
                for chunk in T.serial(dim // 128):
                    T.copy(
                        ub_bias_chunk,
                        workspace_o[
                            kernel_id * 2 * heads + row : kernel_id * 2 * heads
                            + row
                            + heads_half,
                            chunk * 128 : (chunk + 1) * 128,
                        ],
                    )
                T.pipe_barrier("PIPE_MTE3")
                T.set_atomic_none("float32")
            T.set_flag("PIPE_V", vector_event_v1_store_free)
        with T.rs("PIPE_V"):
            T.wait_flag("PIPE_MTE3", vector_event_v1_store_free)
            if task_id % num_top_k_blocks != 0:
                T.vbrc(ub_delta_i32, ub_update_chunk)
            T.pipe_barrier("PIPE_V")
            T.set_flag("PIPE_MTE3", vector_event_v1_store_ready)
        with T.rs("PIPE_MTE3"):
            T.wait_flag("PIPE_V", vector_event_v1_store_ready)
            if task_id % num_top_k_blocks != 0:
                T.pipe_barrier("PIPE_MTE3")
                T.set_atomic_add("int32")
                for chunk in T.serial(dim // 128):
                    T.copy(
                        ub_update_chunk,
                        workspace_o_i32[
                            kernel_id * 2 * heads + row : kernel_id * 2 * heads
                            + row
                            + heads_half,
                            chunk * 128 : (chunk + 1) * 128,
                        ],
                    )
                T.pipe_barrier("PIPE_MTE3")
                T.set_atomic_none("int32")
            T.sync_block_set(6)
            T.set_flag("PIPE_V", vector_event_v1_store_free)

    @T.macro
    def FinalOutput(
        workspace_o,
        Output,
        SoftmaxMax,
        SoftmaxSum,
        ub_cross_kernel_32,
        ub_cross_kernel_16,
        ub_invalid_scores,
        ub_valid_scores,
        ub_running_max,
        ub_running_sum,
        ub_safe_sum,
        ub_one,
        ub_sum_nonzero,
        kernel_id,
        vid,
        task_id,
    ):
        token = task_id // num_top_k_blocks * num_kernels + kernel_id
        row = vid * heads_half
        with T.rs("PIPE_MTE2"):
            T.sync_block_wait(9)
            T.wait_flag("PIPE_V", vector_event_v1_input_free)
        with T.rs("PIPE_V"):
            T.wait_flag("PIPE_MTE3", vector_event_v1_store_free)
            T.wait_flag("PIPE_MTE3", vector_event_output_free)
            T.vcmp(ub_running_sum, T.cast(0.0, "float"), ub_sum_nonzero, "gt")
            T.vbrc(T.cast(1.0, "float"), ub_one)
            T.pipe_barrier("PIPE_V")
            T.vselect(ub_sum_nonzero, ub_safe_sum, ub_one, ub_safe_sum)
            T.vbrc(-T.infinity(accum_dtype), ub_one)
            T.pipe_barrier("PIPE_V")
            T.vselect(ub_sum_nonzero, ub_running_max, ub_one, ub_running_max)
            T.vbrc(T.cast(0.0, "float"), ub_invalid_scores)
            T.vcast(ub_sum_nonzero, ub_one)
            T.pipe_barrier("PIPE_V")
            T.set_flag("PIPE_MTE2", vector_event_merge_input_free)
        for chunk in T.serial(dim // 128):
            with T.rs("PIPE_MTE2"):
                T.wait_flag("PIPE_V", vector_event_merge_input_free)
                T.copy(
                    workspace_o[
                        kernel_id * 2 * heads + row : kernel_id * 2 * heads
                        + row
                        + heads_half,
                        chunk * 128 : (chunk + 1) * 128,
                    ],
                    ub_cross_kernel_32,
                )
                T.set_flag("PIPE_V", vector_event_merge_loaded)
            with T.rs("PIPE_V"):
                T.wait_flag("PIPE_MTE2", vector_event_merge_loaded)
                T.vabs(ub_cross_kernel_32, ub_invalid_scores)
                T.pipe_barrier("PIPE_V")
                T.vcmp(
                    ub_invalid_scores, T.cast(1.0e10, "float"), ub_valid_scores, "lt"
                )
                T.vbrc(T.cast(0.0, "float"), ub_invalid_scores)
                T.pipe_barrier("PIPE_V")
                T.vselect(
                    ub_valid_scores,
                    ub_cross_kernel_32,
                    ub_invalid_scores,
                    ub_cross_kernel_32,
                )
                T.pipe_barrier("PIPE_V")
                T.vdiv(ub_cross_kernel_32, ub_safe_sum, ub_cross_kernel_32)
                T.pipe_barrier("PIPE_V")
                T.vmul(ub_cross_kernel_32, ub_one, ub_cross_kernel_32)
                T.pipe_barrier("PIPE_V")
                T.vcast(
                    ub_cross_kernel_32, ub_cross_kernel_16[:, :128], round_mode="rint"
                )
                T.pipe_barrier("PIPE_V")
                T.set_flag("PIPE_MTE2", vector_event_merge_input_free)
                T.set_flag("PIPE_MTE3", vector_event_output_ready)
            with T.rs("PIPE_MTE3"):
                T.wait_flag("PIPE_V", vector_event_output_ready)
                T.copy(
                    ub_cross_kernel_16[:, :128],
                    Output[
                        token, row : row + heads_half, chunk * 128 : (chunk + 1) * 128
                    ],
                )
                T.set_flag("PIPE_V", vector_event_output_free)
            with T.rs("PIPE_V"):
                T.wait_flag("PIPE_MTE3", vector_event_output_free)
        with T.rs("PIPE_MTE2"):
            T.wait_flag("PIPE_V", vector_event_merge_input_free)
        with T.rs("PIPE_V"):
            T.set_flag("PIPE_MTE2", vector_event_v1_input_free)
            T.set_flag("PIPE_MTE3", vector_event_output_ready)
        with T.rs("PIPE_MTE3"):
            T.wait_flag("PIPE_V", vector_event_output_ready)
            T.copy(ub_running_max[:, 0], SoftmaxMax[0, token, row : row + heads_half])
            T.copy(ub_running_sum[:, 0], SoftmaxSum[0, token, row : row + heads_half])
            T.set_flag("PIPE_V", vector_event_output_free)
            T.set_flag("PIPE_V", vector_event_v1_store_free)

    # Define the main sparse attention kernel using TileLang
    @T.prim_func
    def SparseMlaAmlaAtomicV37(
        QNope: T.Tensor([total_tokens, heads, dim], dtype),
        QRope: T.Tensor([total_tokens, heads, tail_dim], dtype),
        KVNope: T.Tensor([num_pages, page_size, 1, dim], dtype),
        KRope: T.Tensor([num_pages, page_size, 1, tail_dim], dtype),
        BlockTable: T.Tensor([batch, max_blocks], indices_dtype),
        ActualKVLengths: T.Tensor([batch], indices_dtype),
        Indices: T.Tensor([total_tokens, 1, top_k], indices_dtype),
        Output: T.Tensor([total_tokens, heads, dim], dtype),
        SoftmaxMax: T.Tensor([1, total_tokens, heads], accum_dtype),
        SoftmaxSum: T.Tensor([1, total_tokens, heads], accum_dtype),
        workspace_nope: T.Tensor([num_kernels, multi_ws_kv, block, dim], dtype),
        workspace_rope: T.Tensor([num_kernels, multi_ws_kv, block, tail_dim], dtype),
        workspace_s: T.Tensor([num_kernels, multi_ws_s, heads, block], accum_dtype),
        workspace_p: T.Tensor([num_kernels, multi_ws_p, heads, block], dtype),
        workspace_o_i32: T.Tensor([num_kernels * multi_ws_o * heads, dim], "int32"),
        workspace_o: T.Tensor([num_kernels * multi_ws_o * heads, dim], accum_dtype),
        workspace_max: T.Tensor([num_kernels, multi_ws_o, heads], accum_dtype),
        workspace_sum: T.Tensor([num_kernels, multi_ws_o, heads], accum_dtype),
    ):
        # Launch NPU kernel with specified number of parallel kernels
        with T.Kernel(num_kernels, is_npu=True) as (kernel_id, vid):
            # Cube computation section (matrix operations)
            with T.Scope("Cube"):
                # AscendC: four Q/P + three KV slots, 72 KiB each (504 KiB).
                # Keep 72 KiB per Q/P slot, with the actual M=32 NZ row stride.
                l1_qp = T.alloc_L1([4, 32, 1152], dtype)
                l1_kv = T.alloc_L1([3, 128, 288], dtype)
                # Two independent 64 KiB accumulators.
                # Keep 64 KiB per C slot; MMAD/Fixpipe use packed M=32.
                l0_c = T.alloc_L0C([2, 32, 512], accum_dtype)

                num_local_logic_kernels = T.ceildiv(
                    num_logic_kernels - kernel_id, num_kernels
                )
                num_tasks = num_local_logic_kernels * num_top_k_blocks

                init_cube_flags()
                for stream_id in T.serial(num_tasks + 1):
                    if stream_id < num_tasks:
                        task_id = stream_id
                        C1Tile(
                            QNope,
                            QRope,
                            workspace_nope,
                            workspace_rope,
                            workspace_s,
                            l1_qp,
                            l1_kv,
                            l0_c,
                            kernel_id,
                            task_id,
                        )
                    if stream_id > 0:
                        task_id = stream_id - 1
                        C2Tile(
                            workspace_p,
                            workspace_nope,
                            workspace_o,
                            l1_qp,
                            l1_kv,
                            l0_c,
                            kernel_id,
                            task_id,
                        )
                clear_cube_flags()

            # Vector computation section (softmax and normalization)
            with T.Scope("Vector"):
                value_zero = 0
                value_one = 1
                value_min = -T.infinity(accum_dtype)
                # Allocate unified buffers for vector operations
                ub_update = T.alloc_ub([heads_half, dim], "int32")
                ub_update_chunk = T.alloc_ub([heads_half, 128], "int32")
                ub_delta_i32 = T.alloc_ub([heads_half, 1], "int32")
                ub_shat16 = T.alloc_ub([heads_half, 1], dtype)
                ub_n = T.alloc_ub([heads_half, 1], accum_dtype)
                ub_prev_n = T.alloc_ub([heads_half, 1], accum_dtype)
                ub_cof = T.alloc_ub([heads_half, 1], accum_dtype)
                ub_prev_cof = T.alloc_ub([heads_half, 1], accum_dtype)
                ub_eps = T.alloc_ub([heads_half, 1], accum_dtype)
                ub_shat = T.alloc_ub([heads_half, 1], accum_dtype)
                ub_delta = T.alloc_ub([heads_half, 1], accum_dtype)

                # Typed views are placed in AscendC-sized UB pools by the
                # isolated V28 adapter; see ub_pool_adapter.py for offsets.
                ub_nope_even = T.alloc_ub([gather_rows, dim], dtype)
                ub_nope_odd = T.alloc_ub([gather_rows, dim], dtype)
                ub_rope_even = T.alloc_ub([gather_rows, tail_dim], dtype)
                ub_rope_odd = T.alloc_ub([gather_rows, tail_dim], dtype)
                ub_cross_kernel_16 = T.alloc_ub([heads_half, block_dim_share], dtype)
                ub_cross_kernel_32 = T.alloc_ub([heads_half, block], accum_dtype)

                # ub only used in V1P
                ub_tile_max = T.alloc_ub([2 * heads_half, 1], accum_dtype)
                ub_epilogue_tile_max = T.alloc_ub([heads_half, 1], accum_dtype)
                ub_running_max = T.alloc_ub([heads_half, 1], accum_dtype)
                ub_previous_max = T.alloc_ub([heads_half, 1], accum_dtype)
                ub_valid_nonnegative = T.alloc_ub([1, block], "bool")
                ub_valid_below_length = T.alloc_ub([1, block], "bool")
                ub_valid_columns = T.alloc_ub([1, block], "bool")
                ub_valid_columns_fp32 = T.alloc_ub([1, block], accum_dtype)
                ub_valid_scores_fp32 = T.alloc_ub([heads_half, block], accum_dtype)
                ub_valid_scores = T.alloc_ub([heads_half, block], "bool")
                ub_invalid_scores = ub_valid_scores_fp32
                ub_final_input = T.alloc_ub([heads_half, 128], accum_dtype)
                ub_final_invalid = T.alloc_ub([heads_half, 128], accum_dtype)
                ub_final_valid = T.alloc_ub([heads_half, 128], "bool")

                # V2 running state is independent of the V1 softmax buffers.
                ub_tile_sum = T.alloc_ub([2 * heads_half, 1], accum_dtype)
                ub_epilogue_tile_sum = T.alloc_ub([heads_half, 1], accum_dtype)
                ub_running_sum = T.alloc_ub([heads_half, 1], accum_dtype)
                ub_scale = T.alloc_ub([heads_half, 1], accum_dtype)
                ub_tile_scale = T.alloc_ub([heads_half, 1], accum_dtype)
                ub_safe_sum = T.alloc_ub([heads_half, 1], accum_dtype)
                ub_sum_nonzero = T.alloc_ub([heads_half, 1], "bool")
                ub_one = T.alloc_ub([heads_half, 1], accum_dtype)

                # inner cross
                ub_indices = T.alloc_ub([multi_ub_inner_cross, 1, block], indices_dtype)
                ub_actual_kv = T.alloc_ub([multi_ub_inner_cross, 1], indices_dtype)
                ub_v1_indices = T.alloc_ub([1, block], indices_dtype)
                ub_v1_actual_kv = T.alloc_ub([1, 1], indices_dtype)
                ub_v1_indices_fp32 = T.alloc_ub([1, block], accum_dtype)
                ub_v1_actual_kv_fp32 = T.alloc_ub([1, 1], accum_dtype)
                ub_v1_lengths_fp32 = T.alloc_ub([1, block], accum_dtype)
                ub_block_table_even = T.alloc_ub([max_blocks], indices_dtype)
                ub_block_table_odd = T.alloc_ub([max_blocks], indices_dtype)

                num_local_logic_kernels = T.ceildiv(
                    num_logic_kernels - kernel_id, num_kernels
                )
                num_tasks = num_local_logic_kernels * num_top_k_blocks

                init_vector_flags()
                for stream_id in T.serial(num_tasks + 2):
                    if stream_id < num_tasks:
                        task_id = stream_id
                        MergeKV(
                            KVNope,
                            KRope,
                            BlockTable,
                            ActualKVLengths,
                            Indices,
                            workspace_nope,
                            workspace_rope,
                            ub_indices,
                            ub_block_table_even,
                            ub_actual_kv,
                            ub_nope_even,
                            ub_nope_odd,
                            ub_rope_even,
                            ub_rope_odd,
                            kernel_id,
                            vid,
                            task_id,
                        )

                    if stream_id > 1:
                        task_id = stream_id - 2
                        if task_id % num_top_k_blocks == num_top_k_blocks - 1:
                            FinalOutput(
                                workspace_o,
                                Output,
                                SoftmaxMax,
                                SoftmaxSum,
                                ub_final_input,
                                ub_cross_kernel_16,
                                ub_final_invalid,
                                ub_final_valid,
                                ub_running_max,
                                ub_running_sum,
                                ub_safe_sum,
                                ub_one,
                                ub_sum_nonzero,
                                kernel_id,
                                vid,
                                task_id,
                            )

                    if stream_id > 0 and stream_id - 1 < num_tasks:
                        task_id = stream_id - 1
                        V1L(
                            ActualKVLengths,
                            Indices,
                            workspace_s,
                            ub_v1_indices,
                            ub_v1_actual_kv,
                            ub_cross_kernel_32,
                            kernel_id,
                            vid,
                            task_id,
                        )
                        V1P(
                            ub_running_max,
                            ub_running_sum,
                            ub_previous_max,
                            ub_scale,
                            ub_tile_scale,
                            ub_n,
                            ub_prev_n,
                            ub_cof,
                            ub_prev_cof,
                            ub_eps,
                            ub_shat,
                            ub_shat16,
                            ub_delta,
                            ub_delta_i32,
                            ub_update,
                            ub_safe_sum,
                            ub_one,
                            ub_sum_nonzero,
                            ub_tile_max,
                            ub_tile_sum,
                            ub_v1_indices,
                            ub_v1_actual_kv,
                            ub_v1_indices_fp32,
                            ub_v1_actual_kv_fp32,
                            ub_v1_lengths_fp32,
                            ub_cross_kernel_16,
                            ub_cross_kernel_32,
                            ub_valid_nonnegative,
                            ub_valid_below_length,
                            ub_valid_columns,
                            ub_valid_columns_fp32,
                            ub_valid_scores_fp32,
                            ub_valid_scores,
                            ub_invalid_scores,
                            kernel_id,
                            vid,
                            task_id,
                        )
                        V1S(
                            workspace_p,
                            workspace_o_i32,
                            workspace_o,
                            ub_final_invalid,
                            ub_cross_kernel_16,
                            ub_update,
                            ub_update_chunk,
                            ub_delta_i32,
                            kernel_id,
                            vid,
                            task_id,
                        )
                clear_vector_flags()

    return SparseMlaAmlaAtomicV37


_SPARSE_MLA_WORKSPACE_CACHE = {}


def validate_sparse_mla_metadata(
    actual_q_lengths, actual_kv_lengths, block_table, num_pages, total_tokens=None
):
    """Optional preflight outside the hot path; copies metadata to CPU once.

    Uniform B=4 requires cumulative query lengths [S, 2*S, 3*S, 4*S].
    Pass total_tokens=query.shape[0] to also check the query tensor length.
    Every page covering an actual KV length must reference a physical cache page.
    Call this before warmup/capture when request metadata changes.
    """
    q_lengths = actual_q_lengths.detach().cpu().tolist()
    kv_lengths = actual_kv_lengths.detach().cpu().tolist()
    pages = block_table.detach().cpu().tolist()
    if total_tokens is None:
        total_tokens = q_lengths[-1] if q_lengths else 0
    if (
        total_tokens <= 0
        or total_tokens % 4
        or len(q_lengths) != 4
        or q_lengths != [total_tokens // 4 * i for i in range(1, 5)]
        or len(kv_lengths) != 4
        or len(pages) != 4
    ):
        raise ValueError("uniform B=4 requires cumulative Q lengths [S,2*S,3*S,4*S]")
    for length, row in zip(kv_lengths, pages):  # noqa: B905 - preserve measured API compatibility
        if length < 0 or length > len(row) * 128:
            raise ValueError("actual KV length exceeds the block table capacity")
        if any(page < 0 or page >= num_pages for page in row[: (length + 127) // 128]):
            raise ValueError("an active logical page has an invalid physical page")


def _check_tensor(tensor, name, shape, dtype, device):
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(
            f"{name} must have shape {tuple(shape)}, got {tuple(tensor.shape)}"
        )
    if tensor.dtype != dtype or tensor.device != device or not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous {dtype} on {device}")


def warmup_sparse_mla_fwd_pa_bsnd_highperf(
    num_pages,
    max_blocks,
    sm_scale,
    batch=4,
    total_tokens=16,
    heads=32,
    dim=512,
    tail_dim=64,
    topk=2048,
    page_size=128,
    block=None,
    sparse_mode=3,
):
    # Match the launch wrapper when callers precompile with default arguments.
    if block is None:
        block = (
            512
            if (heads, dim, tail_dim) == (32, 512, 64) and topk > 0 and topk % 512 == 0
            else 64
        )
    return sparse_mla_fwd_pa_bsnd_kernel(
        batch,
        total_tokens,
        num_pages,
        max_blocks,
        heads,
        dim,
        tail_dim,
        topk,
        page_size=page_size,
        sm_scale=sm_scale,
        block=block,
        sparse_mode=sparse_mode,
        multi_ws_kv=4,
        multi_ws_s=2,
        multi_ws_p=2,
        multi_ws_o=2,
    )


def clear_sparse_mla_workspace_cache():
    num_entries = len(_SPARSE_MLA_WORKSPACE_CACHE)
    _SPARSE_MLA_WORKSPACE_CACHE.clear()
    return num_entries


def _get_sparse_mla_workspaces(
    q_nope,
    q_rope,
    total_tokens,
    heads,
    dim,
    tail_dim,
    topk,
    block,
    num_kernels,
):
    stream_id = int(torch.npu.current_stream(q_nope.device).npu_stream)
    cache_key = (
        str(q_nope.device),
        stream_id,
        q_nope.dtype,
        q_rope.dtype,
        total_tokens,
        heads,
        dim,
        tail_dim,
        topk,
        block,
        num_kernels,
    )
    workspaces = _SPARSE_MLA_WORKSPACE_CACHE.get(cache_key)
    if workspaces is None:
        workspaces = (
            torch.empty(
                num_kernels,
                4,
                block,
                dim,
                dtype=q_nope.dtype,
                device=q_nope.device,
            ),
            torch.empty(
                num_kernels,
                4,
                block,
                tail_dim,
                dtype=q_rope.dtype,
                device=q_rope.device,
            ),
            torch.empty(
                num_kernels,
                2,
                heads,
                block,
                dtype=torch.float32,
                device=q_nope.device,
            ),
            torch.empty(
                num_kernels,
                2,
                heads,
                block,
                dtype=q_nope.dtype,
                device=q_nope.device,
            ),
            torch.empty(
                num_kernels,
                2,
                heads,
                dim,
                dtype=torch.float32,
                device=q_nope.device,
            ),
            torch.empty(
                num_kernels,
                2,
                heads,
                dtype=torch.float32,
                device=q_nope.device,
            ),
            torch.empty(
                num_kernels,
                2,
                heads,
                dtype=torch.float32,
                device=q_nope.device,
            ),
        )
        _SPARSE_MLA_WORKSPACE_CACHE[cache_key] = workspaces
    return workspaces


def sparse_mla_fwd_pa_bsnd_highperf(
    q_nope,
    kv_nope,
    q_rope,
    k_rope,
    sparse_indices,
    actual_q_lengths,
    actual_kv_lengths,
    block_table,
    sm_scale,
    block=None,
    workspaces=None,
    sparse_mode=3,
):
    """Launch the uniform B=4, S=T/4 specialization on the current NPU stream.

    See validate_sparse_mla_metadata for the value-level input preconditions.
    Cache entries belong to one device/stream; explicit workspaces must also be
    nonoverlapping and exclusive to that stream until its launch finishes.
    This V37 candidate requires H32/D512/Rope64, block512, and topk divisible
    by 512. Other dimension families are not supported by this specialization.
    No host sync is inserted.
    """
    if q_nope.ndim != 3 or q_rope.ndim != 3 or kv_nope.ndim != 4:
        raise ValueError("query must be TND and KV must be PA_BSND")
    if sparse_indices.ndim != 3 or block_table.ndim != 2:
        raise ValueError(
            "indices must be [T, 1, topk] and block_table must be [B, pages]"
        )
    total_tokens, heads, dim = q_nope.shape
    batch = actual_q_lengths.numel()
    tail_dim = q_rope.shape[-1]
    topk = sparse_indices.shape[-1]
    page_size = kv_nope.shape[1]
    num_kernels = min(NPUUtils.get().get_aicore_num(), total_tokens)

    if q_rope.shape[:2] != q_nope.shape[:2]:
        raise ValueError("q_nope and q_rope must have matching TND axes")
    if sparse_indices.shape[:2] != (total_tokens, 1):
        raise ValueError("sparse_indices must have shape [T, 1, topk]")
    if actual_kv_lengths.numel() != batch or block_table.shape[0] != batch:
        raise ValueError("query/KV lengths and block_table batch dimensions differ")
    if batch != 4 or total_tokens <= 0 or total_tokens % 4:
        raise ValueError(
            "uniform profile requires batch=4 and positive T divisible by 4"
        )
    if block is None:
        # Only the block512 dimension family passes this candidate's kernel guards.
        block = (
            512
            if (heads, dim, tail_dim) == (32, 512, 64) and topk > 0 and topk % 512 == 0
            else 64
        )
    if (
        block <= 0
        or block % 8 != 0
        or block > min(dim, 512)
        or topk <= 0
        or topk % block != 0
    ):
        raise ValueError("topk must be divisible by the Expert block size")
    if sparse_mode not in (0, 3):
        raise ValueError("the uniform profile supports sparse_mode 0 and 3")
    if (
        page_size != 128
        or kv_nope.shape[0] <= 0
        or not 0 < block_table.shape[1] <= 1565
    ):
        raise ValueError("expected page size 128 and block-table width in [1, 1565]")
    if (
        not 0 < heads <= 32
        or heads % 16
        or not 0 < dim <= 512
        or dim & (dim - 1)
        or not 0 < tail_dim <= 64
        or tail_dim % 16
    ):
        raise ValueError("unsupported head count or nope/rope dimensions")
    if num_kernels <= 0:
        raise ValueError("at least one AIC is required")
    if not math.isfinite(sm_scale) or sm_scale <= 0:
        raise ValueError("sm_scale must be finite and positive")
    if q_nope.device.type != "npu":
        raise ValueError("this launcher requires NPU tensors")
    device = q_nope.device
    tensor_specs = (
        (q_nope, "q_nope", (total_tokens, heads, dim), torch.bfloat16),
        (q_rope, "q_rope", (total_tokens, heads, tail_dim), torch.bfloat16),
        (kv_nope, "kv_nope", (kv_nope.shape[0], 128, 1, dim), torch.bfloat16),
        (k_rope, "k_rope", (kv_nope.shape[0], 128, 1, tail_dim), torch.bfloat16),
        (sparse_indices, "sparse_indices", (total_tokens, 1, topk), torch.int32),
        (actual_q_lengths, "actual_q_lengths", (4,), torch.int32),
        (actual_kv_lengths, "actual_kv_lengths", (4,), torch.int32),
        (block_table, "block_table", (4, block_table.shape[1]), torch.int32),
    )
    for tensor, name, shape, dtype in tensor_specs:
        _check_tensor(tensor, name, shape, dtype, device)

    output = torch.empty_like(q_nope)
    softmax_max = torch.empty(
        1, total_tokens, heads, dtype=torch.float32, device=q_nope.device
    )
    softmax_sum = torch.empty_like(softmax_max)
    if workspaces is None:
        workspaces = _get_sparse_mla_workspaces(
            q_nope,
            q_rope,
            total_tokens,
            heads,
            dim,
            tail_dim,
            topk,
            block,
            num_kernels,
        )
    if len(workspaces) != 7:
        raise ValueError("workspaces must contain seven tensors")
    workspace_specs = (
        ((num_kernels, 4, block, dim), torch.bfloat16),
        ((num_kernels, 4, block, tail_dim), torch.bfloat16),
        ((num_kernels, 2, heads, block), torch.float32),
        ((num_kernels, 2, heads, block), torch.bfloat16),
        ((num_kernels, 2, heads, dim), torch.float32),
        ((num_kernels, 2, heads), torch.float32),
        ((num_kernels, 2, heads), torch.float32),
    )
    for index, (tensor, (shape, dtype)) in enumerate(zip(workspaces, workspace_specs)):  # noqa: B905 - preserve measured API compatibility
        _check_tensor(tensor, f"workspaces[{index}]", shape, dtype, device)

    kernel = warmup_sparse_mla_fwd_pa_bsnd_highperf(
        num_pages=kv_nope.shape[0],
        max_blocks=block_table.shape[1],
        sm_scale=sm_scale,
        batch=batch,
        total_tokens=total_tokens,
        heads=heads,
        dim=dim,
        tail_dim=tail_dim,
        topk=topk,
        page_size=page_size,
        block=block,
        sparse_mode=sparse_mode,
    )
    kernel(
        q_nope,
        q_rope,
        kv_nope,
        k_rope,
        block_table,
        actual_kv_lengths,
        sparse_indices,
        output,
        softmax_max,
        softmax_sum,
        *workspaces[:4],
        workspaces[4].view(torch.int32).view(num_kernels * 2 * heads, dim),
        workspaces[4].view(num_kernels * 2 * heads, dim),
        *workspaces[5:],
    )
    return output, softmax_max, softmax_sum


def npu_sparse_flash_attention_tilelang(
    query,
    key,
    value,
    query_rope,
    key_rope,
    sparse_indices,
    scale_value,
    actual_seq_lengths_query,
    actual_seq_lengths_kv,
    block_table,
    sparse_block_size=1,
    layout_query="TND",
    layout_kv="PA_BSND",
    sparse_mode=3,
    attention_mode=2,
    return_softmax_lse=False,
    kernel_block=None,
):
    if query.dtype != torch.bfloat16 or key.dtype != torch.bfloat16:
        raise NotImplementedError("only BF16 query and KV are supported")
    if (
        value.data_ptr() != key.data_ptr()
        or value.shape != key.shape
        or value.stride() != key.stride()
        or value.dtype != key.dtype
        or value.device != key.device
    ):
        raise NotImplementedError("key and value must alias the same latent cache")
    if query.shape[1:] != (32, 512):
        raise NotImplementedError("query must have shape [T, 32, 512]")
    if query_rope.shape != (query.shape[0], 32, 64):
        raise NotImplementedError("query_rope must have shape [T, 32, 64]")
    if key.ndim != 4 or key.shape[1:] != (128, 1, 512):
        raise NotImplementedError("key/value must use PA_BSND [P, 128, 1, 512]")
    if key_rope.shape != (key.shape[0], 128, 1, 64):
        raise NotImplementedError("key_rope must have shape [P, 128, 1, 64]")
    if (
        sparse_indices.ndim != 3
        or sparse_indices.shape[:2] != (query.shape[0], 1)
        or sparse_indices.shape[2] <= 0
        or sparse_indices.shape[2] % 64
    ):
        raise NotImplementedError(
            "sparse_indices must have shape [T, 1, topk] with positive topk divisible by 64"
        )
    if (
        query.shape[0] <= 0
        or query.shape[0] % 4
        or actual_seq_lengths_query.numel() != 4
    ):
        raise NotImplementedError("requires batch=4 with uniform S and T=4*S")
    if actual_seq_lengths_kv.numel() != 4 or block_table.shape[0] != 4:
        raise NotImplementedError("KV lengths and block_table must have batch=4")
    if sparse_block_size != 1:
        raise NotImplementedError("only sparse_block_size=1 is supported")
    if layout_query != "TND" or layout_kv != "PA_BSND":
        raise NotImplementedError("only TND query with PA_BSND KV is supported")
    if sparse_mode not in (0, 3):
        raise NotImplementedError("only sparse_mode=0 and sparse_mode=3 are supported")
    if attention_mode != 2:
        raise NotImplementedError("only attention_mode=2 is supported")

    results = sparse_mla_fwd_pa_bsnd_highperf(
        query,
        key,
        query_rope,
        key_rope,
        sparse_indices,
        actual_seq_lengths_query,
        actual_seq_lengths_kv,
        block_table,
        scale_value,
        block=kernel_block,
        sparse_mode=sparse_mode,
    )
    if return_softmax_lse:
        return results
    return results[0]
