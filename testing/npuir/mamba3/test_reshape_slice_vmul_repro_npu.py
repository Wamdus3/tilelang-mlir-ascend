import sys
from pathlib import Path

import pytest
import torch
import torch_npu  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.mamba3.reshape_slice_vmul_repro_npu import (  # noqa: E402
    reshape_slice_vmul_repro,
)

pytestmark = [
    pytest.mark.op("reshape_slice_vmul_repro"),
    pytest.mark.mode("Developer"),
]


def _summary(name, actual, expected, atol=5e-2, rtol=5e-2):
    actual_cpu = actual.cpu()
    expected_cpu = expected.cpu()
    diff = (actual_cpu - expected_cpu).abs()
    close = torch.isclose(actual_cpu, expected_cpu, atol=atol, rtol=rtol)
    mismatched = close.numel() - close.sum().item()
    flat_idx = torch.argmax(diff).item()
    shape = actual_cpu.shape
    idx = []
    remain = flat_idx
    for extent in reversed(shape):
        idx.append(remain % extent)
        remain //= extent
    idx = tuple(reversed(idx))
    return (
        mismatched == 0,
        f"{name}: mismatched={mismatched}/{close.numel()} "
        f"max_abs={diff.max().item():.8g} at={idx} "
        f"actual={actual_cpu[idx].item():.8g} "
        f"expected={expected_cpu[idx].item():.8g}",
    )


@pytest.mark.parametrize(
    "N, M, K",
    [
        (4, 4, 8),
        (8, 2, 8),
        (8, 4, 4),
        (8, 4, 8),
        (8, 8, 8),
        (8, 4, 12),
        (8, 4, 16),
    ],
)
def test_reshape_slice_vmul_repro(N, M, K):
    L = 4 * K
    torch.manual_seed(32000 + N * 100 + M * 10 + K)
    a = torch.randn((N * M, L), dtype=torch.float32, device="npu")
    b = torch.randn((N, K), dtype=torch.float32, device="npu")
    out_slice_vmul = torch.empty(
        (N, M, K),
        dtype=torch.float32,
        device="npu",
    )
    out_direct_slice_vmul = torch.empty_like(out_slice_vmul)
    out_scalar_vmul = torch.empty_like(out_slice_vmul)

    kernel = reshape_slice_vmul_repro(
        N=N,
        M=M,
        K=K,
    )
    kernel(a, b, out_direct_slice_vmul, out_slice_vmul, out_scalar_vmul)

    ref = a.view(N, M, L)[:, :, :K] * b[:, None, :]
    checks = [
        _summary("out_direct_slice_vmul", out_direct_slice_vmul, ref),
        _summary("out_slice_vmul", out_slice_vmul, ref),
        _summary("out_scalar_vmul", out_scalar_vmul, ref),
    ]
    print("\n".join(summary for _, summary in checks), flush=True)
    failed = [summary for passed, summary in checks if not passed]
    if failed:
        pytest.fail("\n".join(failed))
