#!/usr/bin/env python3
"""Variable-T captured-shape runner; adapted from profile_highperf_manual_migrated.py."""

from __future__ import annotations

import argparse
import hashlib
import json
import importlib.util
import math
import statistics
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401 -- registers the NPU backend


OP_PATH = (
    Path(__file__).resolve().parents[2]
    / "example_sparse_mla_fwd_pa_bsnd_kernel_highperf_manual_sync.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=("ascendc", "tilelang"), required=True)
    parser.add_argument("--op", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument(
        "--kernel-block", type=int, choices=(0, 32, 64, 128, 512), default=0
    )
    parser.add_argument(
        "--expected-block", type=int, choices=(32, 64, 128, 512), required=True
    )
    parser.add_argument("--compile-failure-log", type=Path, required=True)
    parser.add_argument("--device", type=int, default=13)
    parser.add_argument("--actual-kv-len", type=int, default=131072)
    parser.add_argument("--sparse-mode", type=int, choices=(0, 3), default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--zero-input", action="store_true")
    parser.add_argument("--event", action="store_true")
    parser.add_argument("--discover-kernels", action="store_true")
    parser.add_argument(
        "--trace-path",
        type=Path,
        default=Path("/tmp/sfa_kernel_discovery.json"),
    )
    parser.add_argument(
        "--fluentllm-root",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    if args.tokens <= 0 or args.tokens % 4:
        parser.error("tokens must be positive and divisible by four")
    if args.topk <= 0 or args.topk % 64:
        parser.error("topk must be positive and divisible by the 64-key tile size")
    return args


def import_tilelang_operator():
    spec = importlib.util.spec_from_file_location("tilelang_sfa_under_test", OP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import TileLang operator from {OP_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def register_ascendc_operator(fluentllm_root: Path) -> None:
    if fluentllm_root is not None:
        sys.path.insert(0, str(fluentllm_root))
    import flash_ops  # noqa: F401


def make_inputs(args: argparse.Namespace) -> dict[str, torch.Tensor | float]:
    minimum_kv = max(args.topk, args.tokens // 4 + 2)
    if not minimum_kv <= args.actual_kv_len <= 1565 * 128:
        raise ValueError(
            f"actual-kv-len must be in [{minimum_kv}, 200320] for all-valid profiling"
        )

    torch.manual_seed(args.seed)
    torch.npu.manual_seed(args.seed)
    device = torch.device(f"npu:{args.device}")
    torch.npu.set_device(device)
    dtype = torch.bfloat16

    def make_tensor(shape: tuple[int, ...]) -> torch.Tensor:
        if args.zero_input:
            return torch.zeros(shape, dtype=dtype, device=device)
        return torch.randn(shape, dtype=dtype, device=device) * 0.1

    query = make_tensor((args.tokens, 32, 512))
    kv = make_tensor((5408, 128, 1, 512))
    query_rope = make_tensor((args.tokens, 32, 64))
    key_rope = make_tensor((5408, 128, 1, 64))
    actual_q = torch.tensor(
        [args.tokens // 4 * i for i in range(1, 5)], dtype=torch.int32, device=device
    )
    actual_kv = torch.full((4,), args.actual_kv_len, dtype=torch.int32, device=device)

    blocks_per_request = math.ceil(args.actual_kv_len / 128)
    if blocks_per_request * 4 > 5408:
        raise ValueError("four requests exceed the 5408 physical KV pages")
    block_table = torch.full((4, 1565), -1, dtype=torch.int32, device=device)
    for request_id in range(4):
        first_block = request_id * blocks_per_request
        block_table[request_id, :blocks_per_request] = torch.arange(
            first_block,
            first_block + blocks_per_request,
            dtype=torch.int32,
            device=device,
        )

    selected = torch.linspace(
        0,
        args.actual_kv_len - max(5, args.tokens // 4 + 1),
        args.topk,
        dtype=torch.float64,
        device=device,
    ).to(torch.int32)
    sparse_indices = (
        selected.view(1, 1, args.topk).expand(args.tokens, 1, args.topk).contiguous()
    )
    mscale = 0.1 * math.log(40.0) + 1.0
    scale_value = (192.0**-0.5) * mscale * mscale
    return {
        "query": query,
        "key": kv,
        "value": kv,
        "query_rope": query_rope,
        "key_rope": key_rope,
        "sparse_indices": sparse_indices,
        "scale_value": scale_value,
        "actual_seq_lengths_query": actual_q,
        "actual_seq_lengths_kv": actual_kv,
        "block_table": block_table,
    }


def call_ascendc(inputs: dict, sparse_mode: int):
    return torch.ops.custom.npu_sparse_flash_attention(
        **inputs,
        sparse_block_size=1,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=sparse_mode,
        attention_mode=2,
        return_softmax_lse=True,
    )


def call_tilelang(module, inputs: dict, sparse_mode: int, kernel_block=0):
    extra = {} if kernel_block == 0 else {"kernel_block": kernel_block}
    return module.npu_sparse_flash_attention_tilelang(
        **inputs,
        **extra,
        sparse_block_size=1,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=sparse_mode,
        attention_mode=2,
        return_softmax_lse=True,
    )


def assert_shapes(outputs) -> None:
    expected = (
        (PROFILE_TOKENS, 32, 512),
        (1, PROFILE_TOKENS, 32),
        (1, PROFILE_TOKENS, 32),
    )
    actual = tuple(tuple(tensor.shape) for tensor in outputs)
    if actual != expected:
        raise RuntimeError(f"unexpected output shapes: {actual}, expected {expected}")


def compare_outputs(ascendc_outputs, tilelang_outputs) -> None:
    names = ("output", "softmax_max", "softmax_sum")
    tolerances = ((2e-2, 2e-2), (2e-3, 2e-3), (2e-2, 2e-2))
    comparisons = []
    for name, ascendc, tilelang, (rtol, atol) in zip(  # noqa: B905 - Python 3.8 compatibility
        names, ascendc_outputs, tilelang_outputs, tolerances
    ):
        ascendc_f32 = ascendc.float()
        tilelang_f32 = tilelang.float()
        difference = torch.where(
            ascendc_f32 == tilelang_f32,
            torch.zeros_like(ascendc_f32),
            (ascendc_f32 - tilelang_f32).abs(),
        )
        max_abs_error = difference.max().item()
        print(f"[PRECISION] {name} max_abs_error={max_abs_error:.8f}")
        comparisons.append((tilelang_f32, ascendc_f32, rtol, atol))
    for tilelang_f32, ascendc_f32, rtol, atol in comparisons:
        torch.testing.assert_close(tilelang_f32, ascendc_f32, rtol=rtol, atol=atol)
    print("[PRECISION_PASS] TileLang matches AscendC")


def event_benchmark(call, warmup: int, iterations: int) -> None:
    for _ in range(warmup):
        outputs = call()
        assert_shapes(outputs)
    torch.npu.synchronize()

    elapsed_us = []
    for _ in range(iterations):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        outputs = call()
        end.record()
        end.synchronize()
        assert_shapes(outputs)
        elapsed_us.append(start.elapsed_time(end) * 1000.0)
    print("[EVENT_SAMPLES]", json.dumps(elapsed_us), flush=True)
    print(
        "[EVENT] "
        f"iterations={iterations} median_us={statistics.median(elapsed_us):.3f} "
        f"min_us={min(elapsed_us):.3f} max_us={max(elapsed_us):.3f}"
    )


def discover_kernels(call, trace_path: Path) -> None:
    call()
    torch.npu.synchronize()
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        record_shapes=False,
    ) as profiler:
        outputs = call()
        assert_shapes(outputs)
        torch.npu.synchronize()
    profiler.export_chrome_trace(str(trace_path))
    print(f"[KERNEL_TRACE] {trace_path}")


def main() -> None:
    global OP_PATH, PROFILE_TOKENS
    args = parse_args()
    OP_PATH = args.op.resolve()
    PROFILE_TOKENS = args.tokens
    print(
        "[SOURCE]",
        OP_PATH,
        hashlib.sha256(OP_PATH.read_bytes()).hexdigest(),
        flush=True,
    )
    if args.impl == "ascendc":
        register_ascendc_operator(args.fluentllm_root)
    tilelang_module = import_tilelang_operator()
    inputs = make_inputs(args)
    print(
        "[CONFIG]",
        json.dumps(
            dict(
                tokens=args.tokens,
                topk=args.topk,
                kernel_block=args.kernel_block,
                expected_block=args.expected_block,
                sparse_mode=args.sparse_mode,
                actual_kv_len=args.actual_kv_len,
                seed=args.seed,
                indices_shape=list(inputs["sparse_indices"].shape),
                actual_q_lengths=[args.tokens // 4 * i for i in range(1, 5)],
            )
        ),
        flush=True,
    )

    # Compile once before measurement. Retry only the known CANN SIGSEGV.
    # Preserve every failed attempt; numerical/device errors are never retried.
    if args.impl != "ascendc":
        from regress_variable_t import is_compiler_crash

        failures = []
        for attempt in range(3):
            try:
                tilelang_outputs = call_tilelang(
                    tilelang_module, inputs, args.sparse_mode, args.kernel_block
                )
                torch.npu.synchronize()
                assert_shapes(tilelang_outputs)
                break
            except RuntimeError as error:
                if not is_compiler_crash(error):
                    raise
                failures.append(dict(attempt=attempt + 1, error=str(error)))
                args.compile_failure_log.write_text(
                    json.dumps(failures, indent=2) + "\n"
                )
                if attempt == 2:
                    raise
                print("RETRY_CANN_COMPILER_CRASH", attempt + 1, flush=True)
        args.compile_failure_log.write_text(json.dumps(failures, indent=2) + "\n")
    if args.impl != "ascendc":
        pools = list(tilelang_module._SPARSE_MLA_WORKSPACE_CACHE.values())
        assert len(pools) == 1
        effective_block = pools[0][2].shape[-1]
        assert effective_block == args.expected_block, (
            effective_block,
            args.expected_block,
        )
        print(
            "[DISPATCH]",
            json.dumps(
                dict(requested_block=args.kernel_block, effective_block=effective_block)
            ),
            flush=True,
        )
    if args.impl == "compare":
        ascendc_outputs = call_ascendc(inputs, args.sparse_mode)
        tilelang_outputs = call_tilelang(
            tilelang_module, inputs, args.sparse_mode, args.kernel_block
        )
        torch.npu.synchronize()
        assert_shapes(ascendc_outputs)
        assert_shapes(tilelang_outputs)
        compare_outputs(ascendc_outputs, tilelang_outputs)
        return

    if args.impl == "ascendc":

        def call():
            return call_ascendc(inputs, args.sparse_mode)
    else:

        def call():
            return call_tilelang(
                tilelang_module, inputs, args.sparse_mode, args.kernel_block
            )

    if args.discover_kernels:
        discover_kernels(call, args.trace_path)
        return
    if args.event:
        event_benchmark(call, args.warmup, args.iterations)
        return
    for _ in range(args.warmup + args.iterations):
        assert_shapes(call())
    torch.npu.synchronize()
    print("[PROFILE_LAUNCHES_COMPLETE]", args.tokens, args.sparse_mode, flush=True)


if __name__ == "__main__":
    main()
