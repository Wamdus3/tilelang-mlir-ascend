#!/usr/bin/env python3
"""Prepare variable-T SFA fixtures, or test an explicitly selected NPU candidate.

The reference backend validates fixtures and a mathematical scheduling model.
It does not execute TileLang source and is not kernel precision acceptance.
Only --backend npu imports torch/torch_npu/the candidate module.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


@dataclass(frozen=True)
class Case:
    q_lengths: tuple[int, ...]
    blocks: int
    mode: int
    pattern: str
    small: bool = False
    kernel_block: int = 64

    @property
    def tokens(self):
        return sum(self.q_lengths)

    @property
    def name(self):
        layout = (
            "s" + str(self.q_lengths[0])
            if len(set(self.q_lengths)) == 1
            else "q" + "-".join(map(str, self.q_lengths))
        )
        suffix = "" if self.kernel_block == 64 else f"_block{self.kernel_block}"
        return f"t{self.tokens}_{layout}_tiles{self.blocks}_m{self.mode}_{self.pattern}{suffix}"

    def validate(self):
        if len(self.q_lengths) != 4 or min(self.q_lengths) < 0 or self.tokens <= 0:
            raise ValueError("expected B=4, nonnegative query lengths and T>0")
        if self.blocks <= 0 or self.mode not in (0, 3):
            raise ValueError("positive tile count and sparse_mode 0/3 are required")
        if self.pattern not in ("random", "zero", "wide"):
            raise ValueError("unknown input pattern")
        if self.kernel_block not in (0, 32, 64, 128, 512) or (
            self.kernel_block and self.blocks * 64 % self.kernel_block
        ):
            raise ValueError("topk (blocks*64) must be divisible by kernel_block")


def token_schedule(tokens, cores):
    """Expected strided ownership, not an assertion about a future implementation."""
    if tokens <= 0 or cores <= 0:
        raise ValueError("tokens/cores must be positive")
    active = min(tokens, cores)
    return [list(range(core, tokens, active)) for core in range(active)]


def manifest(case, cores):
    schedule = token_schedule(case.tokens, cores)
    ends, value = [], 0
    for length in case.q_lengths:
        value += length
        ends.append(value)
    return dict(
        **asdict(case),
        name=case.name,
        total_tokens=case.tokens,
        actual_q_lengths=ends,
        topk=64 * case.blocks,
        dimensions=[16, 64, 16] if case.small else [32, 512, 64],
        expected_strided_schedule=schedule,
        min_tokens_per_core=min(map(len, schedule)),
        max_tokens_per_core=max(map(len, schedule)),
        extension_only=len(set(case.q_lengths)) != 1,
    )


def bf16(x):
    import numpy as np

    x = np.asarray(x, dtype=np.float32)
    bits = x.view(np.uint32)
    return (
        (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    ).view(np.float32)


def make_case(case, epoch=0):
    import numpy as np

    case.validate()
    # Stable across processes; epoch changes data without changing tensor shapes.
    seed = 20260908 + case.tokens * 1009 + case.blocks * 31 + epoch * 7919
    rng = np.random.default_rng(seed)
    h, d, r = (16, 64, 16) if case.small else (32, 512, 64)
    t, topk, page = case.tokens, case.blocks * 64, 128
    max_s = max(case.q_lengths)
    width = (topk + max_s + 3 + 127) // page + 1
    num_pages = 4 * width + 1
    q = bf16(rng.normal(0, 0.2, (t, h, d)))
    qr = bf16(rng.normal(0, 0.2, (t, h, r)))
    if case.pattern == "zero":
        q.fill(0)
        qr.fill(0)
    if case.pattern == "wide":
        # BF16 exact power-of-two scaling: typical score std grows to ~5.12.
        # Consecutive tile maxima differ substantially, testing both rescale paths.
        q *= 128
        qr *= 128
    kv = bf16(rng.normal(0, 0.2, (num_pages, page, 1, d)))
    kr = bf16(rng.normal(0, 0.2, (num_pages, page, 1, r)))
    # The extra physical page must never be referenced by the page table.
    kv[-1].fill(np.nan)
    kr[-1].fill(np.nan)
    table = rng.permutation(num_pages - 1).reshape(4, width).astype(np.int32)
    lengths = np.array(
        [
            0,
            topk + max_s + 3 - epoch % 2,
            topk // 2 + max_s + 1 - epoch % 2,
            max(1, case.q_lengths[3] // 2),
        ],
        np.int32,
    )
    for request, length in enumerate(lengths):
        table[request, (int(length) + page - 1) // page :] = -1
    ends = np.cumsum(case.q_lengths, dtype=np.int32)
    indices = rng.integers(-9, width * page + 7, (t, 1, topk), dtype=np.int32)
    start = 0
    for request, count in enumerate(case.q_lengths):
        length = int(lengths[request])
        for local in range(count):
            row = indices[start + local, 0]
            if request == 1:
                row[:] = (np.arange(topk) + local * 7 + epoch) % length
                if local % 3 == 0:
                    row[:64] = -1
                elif local % 3 == 1:
                    row[-64:] = -1
                else:
                    row[:] = np.where(np.arange(topk) // 64 % 2, -1, 0)
            elif request == 3:
                # Includes L<S, early empty queries and exact causal counts.
                row[:] = np.arange(topk) % length
            if request != 3:
                # Include duplicates and both sides of each causal boundary.
                upper = length - count + local + 1
                row[:8] = [0, 0, -1, -7, length, length + 1, upper - 1, upper]
        start += count
    data = dict(
        QNope=q,
        QRope=qr,
        KVNope=kv,
        KRope=kr,
        BlockTable=table,
        ActualQLengths=ends,
        ActualKVLengths=lengths,
        Indices=indices,
    )
    validate_data(data)
    return data, (d + r) ** -0.5


def validate_data(data):
    import numpy as np

    ends = data["ActualQLengths"]
    if (
        ends.shape != (4,)
        or np.any(np.diff(np.r_[0, ends]) < 0)
        or ends[-1] != len(data["QNope"])
    ):
        raise ValueError("invalid cumulative query lengths")
    table = data["BlockTable"]
    for length, row in zip(data["ActualKVLengths"], table):  # noqa: B905 - Python 3.8 compatibility
        if length < 0 or length > len(row) * 128:
            raise ValueError("KV length exceeds page table")
        active = row[: (int(length) + 127) // 128]
        if np.any(active < 0) or np.any(active >= len(data["KVNope"])):
            raise ValueError("invalid active physical page")


def golden(data, scale, mode):
    """Dense independent FP32 golden; cumulative ends drive request mapping."""
    import numpy as np

    q, qr = data["QNope"], data["QRope"]
    t, h, _ = q.shape
    out = np.zeros_like(q)
    maximum = np.full((1, t, h), -np.inf, np.float32)
    denominator = np.zeros((1, t, h), np.float32)
    for token in range(t):
        request = int(np.searchsorted(data["ActualQLengths"], token, side="right"))
        upper = int(data["ActualKVLengths"][request])
        if mode == 3:
            upper += token - int(data["ActualQLengths"][request]) + 1
        idx = data["Indices"][token, 0]
        idx = idx[(idx >= 0) & (idx < upper)]
        if not len(idx):
            continue
        pages = data["BlockTable"][request, idx // 128]
        keys = data["KVNope"][pages, idx % 128, 0]
        rope = data["KRope"][pages, idx % 128, 0]
        scores = (q[token] @ keys.T + qr[token] @ rope.T) * scale
        m = scores.max(axis=1)
        p = np.exp(scores - m[:, None])
        l = p.sum(axis=1)
        out[token] = p @ keys / l[:, None]
        maximum[0, token], denominator[0, token] = m, l
    return out, maximum, denominator


def scheduled_reference(data, scale, mode, cores):
    """Per-token-reset tile math under strided core ownership, no TileLang import."""
    import numpy as np

    q, qr = data["QNope"], data["QRope"]
    t, h, d = q.shape
    result = (
        np.zeros_like(q),
        np.full((1, t, h), -np.inf, np.float32),
        np.zeros((1, t, h), np.float32),
    )
    counts = np.diff(np.r_[0, data["ActualQLengths"]])
    mapping = [
        (request, local, int(count))
        for request, count in enumerate(counts)
        for local in range(int(count))
    ]
    for owned_tokens in token_schedule(t, cores):
        for token in owned_tokens:
            # This reset and finishing the output before the next token are
            # required even when physical workspace is reused by the same core.
            running_m = np.full((h, 1), -1e30, np.float32)
            running_l = np.zeros((h, 1), np.float32)
            running_o = np.zeros((h, d), np.float32)
            request, local, count = mapping[token]
            upper = int(data["ActualKVLengths"][request])
            if mode == 3:
                upper = upper - count + local + 1
            for offset in range(0, data["Indices"].shape[-1], 64):
                idx = data["Indices"][token, 0, offset : offset + 64]
                valid = (idx >= 0) & (idx < upper)
                keys, rope = (
                    np.zeros((64, d), np.float32),
                    np.zeros((64, qr.shape[-1]), np.float32),
                )
                chosen = idx[valid]
                pages = data["BlockTable"][request, chosen // 128]
                keys[valid] = data["KVNope"][pages, chosen % 128, 0]
                rope[valid] = data["KRope"][pages, chosen % 128, 0]
                scores = (q[token] @ keys.T + qr[token] @ rope.T) * scale
                scores[:, ~valid] = -1e30
                m = scores.max(axis=1, keepdims=True)
                p = np.exp(scores - m)
                p[:, ~valid] = 0
                l, o = p.sum(axis=1, keepdims=True), bf16(p) @ keys
                new_m = np.maximum(running_m, m)
                a, b = np.exp(running_m - new_m), np.exp(m - new_m)
                running_o = a * running_o + b * o
                running_l = a * running_l + b * l
                running_m = new_m
            result[0][token] = bf16(running_o / np.where(running_l > 0, running_l, 1))
            result[1][0, token] = np.where(
                running_l[:, 0] > 0, running_m[:, 0], -np.inf
            )
            result[2][0, token] = running_l[:, 0]
    return result


def check(actual, expected):
    import numpy as np

    if len(actual) != 3:
        raise AssertionError("candidate must return Output, SoftmaxMax, SoftmaxSum")
    errors = {}
    for name, a, b, (rtol, atol) in zip(  # noqa: B905 - Python 3.8 compatibility
        ("Output", "Max", "Sum"),
        actual,
        expected,
        ((0.02, 0.002), (0.0002, 0.00002), (0.0002, 0.001)),
    ):
        if a.shape != b.shape:
            raise AssertionError(f"{name} shape {a.shape} != {b.shape}")
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=name)
        if np.isnan(a).any():
            raise AssertionError(f"{name} contains NaN")
        finite = np.isfinite(b)
        errors[name] = (
            float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.0
        )
    empty = expected[2][0] == 0
    np.testing.assert_array_equal(actual[0][empty], 0)
    np.testing.assert_array_equal(actual[1][0][empty], -np.inf)
    np.testing.assert_array_equal(actual[2][0][empty], 0)
    return errors


def check_zero_counts(data, output, mode):
    """Analytic count for request 3, including KV shorter than the query count."""
    import numpy as np

    start, end = map(int, data["ActualQLengths"][2:4])
    length = int(data["ActualKVLengths"][3])
    topk = data["Indices"].shape[-1]
    quotient, remainder = divmod(topk, length)
    for token in range(start, end):
        limit = length if mode == 0 else max(0, length + token - end + 1)
        expected = quotient * limit + min(remainder, limit)
        np.testing.assert_array_equal(output[2][0, token], expected)
        np.testing.assert_array_equal(output[1][0, token], 0 if expected else -np.inf)


def load_candidate(path, device):
    os.environ.setdefault("TILELANG_ASCEND_MODE", "Expert")
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(device)
    spec = importlib.util.spec_from_file_location("sfa_variable_t_candidate", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    if not callable(getattr(module, "sparse_mla_fwd_pa_bsnd_highperf", None)):
        raise ValueError("candidate must expose sparse_mla_fwd_pa_bsnd_highperf")
    return module


def is_compiler_crash(error):
    message = str(error)
    return (
        isinstance(error, RuntimeError)
        and getattr(error.__cause__, "returncode", None) == -11
        and "NPU IR opt failed" in message
        and "PLEASE submit a bug report" in message
        and "bishengir-compile" in message
    )


def run_npu(case, module, device, repeats):
    import numpy as np
    import torch

    # Alternate fixtures at the same device addresses: catches stale metadata.
    fixtures = [make_case(case, epoch) for epoch in (0, 1)]
    expected = [golden(data, scale, case.mode) for data, scale in fixtures]
    first = fixtures[0][0]
    dtype = {np.dtype("float32"): torch.bfloat16, np.dtype("int32"): torch.int32}
    inputs = {
        key: torch.tensor(value, dtype=dtype[value.dtype], device=device)
        for key, value in first.items()
    }
    maximum_errors = dict(Output=0.0, Max=0.0, Sum=0.0)
    poison_counts = []
    for iteration in range(repeats):
        epoch = iteration % 2
        data, scale = fixtures[epoch]
        for key, value in data.items():
            inputs[key].copy_(torch.from_numpy(value))
        preflight = getattr(module, "validate_sparse_mla_metadata", None)
        if callable(preflight):
            preflight(
                inputs["ActualQLengths"],
                inputs["ActualKVLengths"],
                inputs["BlockTable"],
                len(inputs["KVNope"]),
            )
        poisoned = 0
        cache = getattr(module, "_SPARSE_MLA_WORKSPACE_CACHE", None)
        if not isinstance(cache, dict):
            raise ValueError(
                "candidate must expose its workspace cache for reuse poisoning"
            )
        for buffers in cache.values():
            for tensor in buffers:
                if tensor.is_floating_point():
                    tensor.fill_(float("nan"))
                    poisoned += 1
        poison_counts.append(poisoned)
        outputs = module.sparse_mla_fwd_pa_bsnd_highperf(
            inputs["QNope"],
            inputs["KVNope"],
            inputs["QRope"],
            inputs["KRope"],
            inputs["Indices"],
            inputs["ActualQLengths"],
            inputs["ActualKVLengths"],
            inputs["BlockTable"],
            scale,
            block=case.kernel_block or None,
            sparse_mode=case.mode,
        )
        torch.npu.synchronize()
        if len(outputs) != 3:
            raise AssertionError("three outputs required")
        if tuple(out.dtype for out in outputs) != (
            torch.bfloat16,
            torch.float32,
            torch.float32,
        ):
            raise AssertionError("output dtypes must be BF16/FP32/FP32")
        if any(out.device != inputs["QNope"].device for out in outputs):
            raise AssertionError("all outputs must remain on the selected NPU")
        actual = tuple(out.float().cpu().numpy() for out in outputs)
        errors = check(actual, expected[epoch])
        if case.pattern == "zero":
            check_zero_counts(data, actual, case.mode)
        maximum_errors = {key: max(maximum_errors[key], errors[key]) for key in errors}
        if iteration > 0 and poisoned == 0:
            raise AssertionError(
                "repeated launch did not expose reusable workspace to poison"
            )
    return dict(
        max_abs=maximum_errors,
        launches=repeats,
        poisoned_tensors_per_launch=poison_counts,
        metadata_epochs=[i % 2 for i in range(repeats)],
    )


def self_test():
    for t in (4, 8, 16, 20, 24, 28, 32, 48, 64, 96):
        for cores in (16, 24):
            schedule = token_schedule(t, cores)
            assert sorted(token for group in schedule for token in group) == list(
                range(t)
            )
            assert max(map(len, schedule)) - min(map(len, schedule)) <= 1
    for counts in ((1, 1, 1, 1), (7, 7, 7, 7), (0, 2, 0, 3)):
        for mode in (0, 3):
            case = Case(counts, 3, mode, "zero", True)
            data, scale = make_case(case)
            reference = golden(data, scale, mode)
            check_zero_counts(data, reference, mode)
            check(scheduled_reference(data, scale, mode, 24), reference)
            broken = tuple(value.copy() for value in reference)
            broken[0][-1].fill(1)
            try:
                check(broken, reference)
            except AssertionError:
                pass
            else:
                raise AssertionError("output corruption was not detected")
            invalid = dict(data, ActualQLengths=data["ActualQLengths"].copy())
            invalid["ActualQLengths"][-1] += 1
            try:
                validate_data(invalid)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid query ends were not detected")
    # Model the actual multi-token hazard: core 16 later handles token 40;
    # writing by core_id again would replace token 16 with token 40's result.
    data, scale = make_case(Case((16,) * 4, 3, 3, "random", True))
    expected = golden(data, scale, 3)
    overwritten = tuple(value.copy() for value in expected)
    overwritten[0][16] = expected[0][40]
    try:
        check(overwritten, expected)
    except AssertionError:
        pass
    else:
        raise AssertionError("cross-token workspace overwrite was not detected")
    print(
        "REFERENCE_SELF_TEST_PASS: scheduling, analytic counts, corruption and metadata controls",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=("list", "reference", "npu"), default="list"
    )
    parser.add_argument(
        "--suite",
        choices=("smoke", "core-boundary", "pipeline", "transition", "ragged"),
        default="smoke",
    )
    parser.add_argument("--tokens", type=int, nargs="+")
    parser.add_argument("--blocks", type=int, nargs="+")
    parser.add_argument(
        "--kernel-block",
        type=int,
        choices=(0, 32, 64, 128, 512),
        default=64,
        help="kernel tile width; 0 requests candidate auto dispatch; --blocks retains units of 64 keys",
    )
    parser.add_argument("--modes", type=int, choices=(0, 3), nargs="+", default=[0, 3])
    parser.add_argument(
        "--patterns",
        choices=("random", "zero", "wide"),
        nargs="+",
        default=["random", "zero"],
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=24,
        help="reference scheduler core count; does not set NPU launch dims",
    )
    parser.add_argument(
        "--small", action="store_true", help="reference-only H=16/D=64/R=16 fixtures"
    )
    parser.add_argument(
        "--op",
        type=Path,
        help="explicit future candidate path; no fallback to fixed-T v2",
    )
    parser.add_argument("--device", default="npu:13")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--compile-retries",
        type=int,
        choices=(0, 1, 2),
        default=0,
        help="bounded retries only for a CANN compiler SIGSEGV; never retries precision failures",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="continue independent cases after compiler crashes; final exit remains nonzero",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="retain previous results and failures for the exact same source hash and matrix",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.cores <= 0 or args.repeats < 2:
        parser.error(
            "cores must be positive and repeats >= 2 for metadata changes/reuse"
        )
    if args.backend == "npu" and (
        args.op is None or not args.op.is_file() or args.small
    ):
        parser.error(
            "NPU requires --op EXISTING_CANDIDATE and target dimensions (no --small)"
        )
    defaults = {
        "smoke": [16, 28, 48],
        "core-boundary": [4, 8, 16, 20, 24, 28, 32, 48, 64, 96],
        "pipeline": [4, 28, 48, 64],
        "transition": [16, 48, 4, 64, 24, 28, 16],
    }
    if args.suite == "ragged":
        if args.tokens:
            parser.error("ragged extension defines its own query lengths")
        counts = [(1, 2, 3, 4), (0, 8, 0, 20), (1, 1, 1, 22)]
    else:
        tokens = args.tokens or defaults[args.suite]
        if any(t <= 0 or t % 4 for t in tokens):
            parser.error("uniform B=4 requires T>0 and T divisible by 4")
        counts = [(t // 4,) * 4 for t in tokens]
    blocks = args.blocks or ([1, 2, 3, 32, 33] if args.suite == "pipeline" else [32])
    cases = [
        Case(q, b, m, p, args.small, args.kernel_block)
        for q in counts
        for b in blocks
        for m in args.modes
        for p in args.patterns
    ]
    for case in cases:
        case.validate()
    result = dict(
        backend=args.backend,
        suite=args.suite,
        candidate=str(args.op.resolve()) if args.op else None,
        candidate_sha256=hashlib.sha256(args.op.read_bytes()).hexdigest()
        if args.op
        else None,
        reference_only=args.backend != "npu",
        cases=[manifest(c, args.cores) for c in cases],
        compile_failures=[],
        results=[],
    )
    if args.resume:
        if args.output is None or not args.output.is_file():
            parser.error("--resume requires an existing --output JSON")
        previous = json.loads(args.output.read_text())
        if (
            previous["backend"] != result["backend"]
            or previous["candidate_sha256"] != result["candidate_sha256"]
            or json.dumps(previous["cases"], sort_keys=True)
            != json.dumps(result["cases"], sort_keys=True)
        ):
            parser.error("resume source hash/backend/matrix mismatch")
        result["results"] = previous["results"]
        result["compile_failures"] = previous.get("compile_failures", [])
        result["resume_count"] = previous.get("resume_count", 0) + 1
    completed = {row["index"] for row in result["results"]}

    def compile_group(case):
        return case.q_lengths, case.blocks, case.mode, case.small, case.kernel_block

    failed_indexes = {
        row["index"] for row in result["results"] if row["status"] == "FAIL"
    }
    blocked_groups = {
        compile_group(cases[row["index"]])
        for row in result["compile_failures"]
        if row["index"] in failed_indexes
    }

    def save():
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    if args.backend == "list":
        print(json.dumps(result, indent=2))
        return
    if args.self_test:
        self_test()
    candidate = (
        load_candidate(args.op.resolve(), args.device)
        if args.backend == "npu"
        else None
    )
    for index, case in enumerate(cases):
        if index in completed:
            continue
        if compile_group(case) in blocked_groups:
            result["results"].append(
                dict(
                    index=index,
                    name=case.name,
                    status="BLOCKED_BY_COMPILE",
                    error="same specialization already exhausted compiler retries",
                )
            )
            print("BLOCKED_BY_COMPILE", case.name, flush=True)
            save()
            continue
        started = time.monotonic()
        try:
            if candidate:
                for attempt in range(args.compile_retries + 1):
                    try:
                        metrics = run_npu(case, candidate, args.device, args.repeats)
                        break
                    except RuntimeError as error:
                        # Exact failure classification: no device error, numeric
                        # mismatch, timeout or other compilation error is retried.
                        message = str(error)
                        if not is_compiler_crash(error):
                            raise
                        result["compile_failures"].append(
                            dict(
                                index=index,
                                name=case.name,
                                attempt=attempt + 1,
                                error=message,
                            )
                        )
                        save()
                        if attempt == args.compile_retries:
                            raise
                        print(
                            "RETRY_CANN_COMPILER_CRASH",
                            case.name,
                            attempt + 1,
                            flush=True,
                        )
            else:
                data, scale = make_case(case)
                expected = golden(data, scale, case.mode)
                actual = scheduled_reference(data, scale, case.mode, args.cores)
                metrics = dict(max_abs=check(actual, expected))
                if case.pattern == "zero":
                    check_zero_counts(data, actual, case.mode)
            status = "NPU_PRECISION_PASS" if candidate else "REFERENCE_FIXTURE_PASS"
            result["results"].append(
                dict(index=index, name=case.name, status=status, **metrics)
            )
            print(status, case.name, json.dumps(metrics), flush=True)
        except Exception as error:
            result["results"].append(
                dict(
                    index=index,
                    name=case.name,
                    status="FAIL",
                    failure_kind="cann_compiler_sigsegv"
                    if is_compiler_crash(error)
                    else "other",
                    error=f"{type(error).__name__}: {error}",
                )
            )
            save()
            if args.keep_going and is_compiler_crash(error):
                blocked_groups.add(compile_group(case))
                print("COMPILER_FAILURE_RETAINED", case.name, flush=True)
                continue
            raise
        result["results"][-1]["wall_seconds"] = time.monotonic() - started
        save()
    if any(
        row["status"] in ("FAIL", "BLOCKED_BY_COMPILE") for row in result["results"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
