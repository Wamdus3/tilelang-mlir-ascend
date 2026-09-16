"""MultiHeadAttentionFwdOp (spec-conformant interface).

Adaptation from GPU (TileOPs) to NPU:

- O3: ``tune`` parameter removed from ``__init__``; the kernel
  constructor takes no ``tune`` argument.
- O6: op-specific flow preserved unchanged -- the GPU routes MHA through
  ``GroupedQueryAttentionFwdOp`` (the ``heads_kv == heads``
  specialization), which resolves ``sm_scale = dim ** -0.5`` and
  ``softcap = 0.0`` and instantiates the dense-path GQA prefill kernel at
  construction time, and ``forward`` returns the historical MHA contract
  ``(output, lse)``.  The NPU port inlines that dense-path dispatch (the
  GQA op wrapper itself is not migrated) while keeping the same kernel
  key (``gqa_prefill_fwd_kernel``) and return contract.
- K5: the GPU kernel selection (``_select_gqa_prefill_fwd_kernel_cls`` --
  Hopper warp-specialized variant -- plus the H200 square dense fast
  path) collapses to the architecture-generic ``GQAPrefillFwdKernel``.
- ``eval_roofline`` is ported from the GPU manifest's
  ``tileops.perf.formulas.mha_fwd_roofline`` (the NPU project has no
  tileops.perf module; roofline arithmetic is device-agnostic).
"""

from typing import Dict, Optional

import torch

from tileops.kernels.attention.multi_head_attention import GQAPrefillFwdKernel
from tileops.kernels.kernel_base import Kernel
from tileops.ops.op_base import Op

__all__ = ["MultiHeadAttentionFwdOp"]


class MultiHeadAttentionFwdOp(Op):
    """Multi-head attention forward. Layout: BSHD.

    MHA is the ``heads_kv == heads`` specialization of GQA, so the
    maintained forward path runs the GQA prefill kernel while keeping the
    historical MHA return contract ``(output, lse)``.

    Args:
        batch: Batch size.
        heads: Number of heads.
        seq_len: Sequence length (query and kv; ``s_q != s_kv`` is not
            supported yet).
        dim: Head dimension.
        is_causal: Whether to apply a causal mask.
        dtype: Data type (float16 or bfloat16).
        kernel_map: Optional override for kernel dispatch.
    """

    def __init__(
        self,
        batch: int,
        heads: int,
        seq_len: int,
        dim: int,
        is_causal: bool = True,
        dtype: torch.dtype = torch.float16,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ) -> None:
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len  # TODO: support s_q != s_kv
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype

        self.dispatch_kernel(kernel_map)
        # GPU: GroupedQueryAttentionFwdOp(batch, heads, heads_kv=heads, ...)
        # resolves sm_scale = dim ** -0.5 and softcap = 0.0, then
        # instantiates the dense-path kernel at construction time (MHA's
        # torch.compile smoke expects forward to call an already-built
        # custom op).  O3: no ``tune`` argument.
        self.kernel: Kernel = self.kernel_map["gqa_prefill_fwd_kernel"](
            batch,
            heads,
            heads,
            seq_len,
            seq_len,
            dim,
            is_causal,
            dtype,
            sm_scale=dim**-0.5,
            softcap=0.0,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        # K5: the GPU selects GQAPrefillFwdKernel (generic),
        # GQAPrefillFwdWsPersistentCausalKernel (Hopper) or the H200
        # square GQAFwdWsPersistentCausalKernel via
        # _select_gqa_prefill_fwd_kernel_cls / the square dense fast path;
        # NPU uses the architecture-generic dense prefill kernel.
        return {"gqa_prefill_fwd_kernel": GQAPrefillFwdKernel}

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run MHA forward on BSHD-layout q/k/v; returns ``(output, lse)``."""
        return self.kernel(q, k, v)

    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)`` for the configured workload shape.

        Ported verbatim (device-agnostic arithmetic) from the GPU repo's
        ``tileops.perf.formulas.mha_fwd_roofline``: 2 GEMMs per
        ``(S x S x D)`` score/output pair = ``4 * B * H * S * S * D``
        FLOPs (halved when causal); q/k/v read + o written.
        """
        elem_bytes = self.dtype.itemsize
        flops = 4 * self.batch * self.heads * self.seq_len * self.seq_len * self.dim
        if self.is_causal:
            flops //= 2
        q_elems = self.batch * self.seq_len * self.heads * self.dim
        kv_elems = q_elems
        return int(flops), int(2 * (q_elems + kv_elems) * elem_bytes)
