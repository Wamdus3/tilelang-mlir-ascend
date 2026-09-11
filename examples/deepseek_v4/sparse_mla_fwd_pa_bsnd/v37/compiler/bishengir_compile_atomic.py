#!/usr/bin/env python3
"""Apply UB layout and CANN compatibility, then invoke the configured compilers.

Atomic mode changes are emitted by TileLang directly from explicit kernel APIs.
This driver does not insert, infer, hoist or remove atomic operations.
"""

import hashlib
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

from sfa_ub_layout import UB_LAYOUT
from ub_pool_adapter import place


def prepare_ir(source):
    """Preserve explicit kernel atomics; apply only CANN compatibility and UB layout."""
    source = source.replace(" syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>", "")
    return place(source, **UB_LAYOUT)


def main(args=None):
    args = sys.argv[1:] if args is None else args
    real = os.environ.get(
        "SFA_CANN_COMPILER", "/usr/local/Ascend/cann-9.0.0/bin/bishengir-compile"
    )
    front = os.environ.get(
        "SFA_ATOMIC_OPT", "/workdir/yja-bishengir-atomic/build/bin/bishengir-opt"
    )
    if not args or not Path(args[0]).is_file():
        os.execv(real, [real, *args])
    source = Path(args[0]).read_text()
    root = Path(os.environ["SFA_ATOMIC_IR_DIR"]) / (
        hashlib.sha256(source.encode()).hexdigest()[:16] + "-" + uuid.uuid4().hex[:8]
    )
    root.mkdir(parents=True)
    (root / "input.mlir").write_text(source)
    placed, layout = prepare_ir(source)
    (root / "ub-layout.json").write_text(json.dumps(layout, indent=2))
    (root / "explicit.mlir").write_text(placed)
    cmd = [
        front,
        str(root / "explicit.mlir"),
        "--canonicalize",
        "-o",
        str(root / "validated.mlir"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    (root / "frontend.log").write_text(result.stdout + result.stderr)
    (root / "receipt.json").write_text(
        json.dumps(
            {"args": args, "front": cmd, "atomic_source": "kernel_explicit"}, indent=2
        )
    )
    if result.returncode:
        sys.stderr.write(result.stderr)
        return result.returncode
    cmd = [
        real,
        str(root / "validated.mlir"),
        *args[1:],
        "--mlir-print-ir-after-failure",
        "--mlir-disable-threading",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    (root / "backend.log").write_text(result.stdout + result.stderr)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    print("[ATOMIC_FRONTEND]", root, "atomic_source", "kernel_explicit", flush=True)
    if result.returncode < 0:
        os.kill(os.getpid(), -result.returncode)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
