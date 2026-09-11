"""Lower the SFA jump example to IR, without invoking bishengir-compile.

Run from an NPUIR-enabled TileLang build:
    python examples/deepseek_v4/sfa_copy_jump_ir.py --output-dir /tmp/sfa-jump-ir

The output directory must be new so a failed run cannot leave stale success IR.
No device tensor is allocated and no kernel is launched. The TileLang import
currently still requires its normal Python and native-library dependencies.
"""

import argparse
import json
import os
from pathlib import Path
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, choices=(64, 512), default=512)
    parser.add_argument(
        "--jump",
        type=int,
        default=None,
        help="Use a constant jump; by default load it from UB",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    status = {
        "completed": [],
        "stage": "import",
        "width": args.width,
        "jump": args.jump,
        "invokes_bishengir_compile": False,
    }
    os.environ["TILELANG_ASCEND_MODE"] = "Expert"
    os.environ["TILELANG_ASCEND_DEVICE_NAME"] = "Ascend910B"
    os.environ["TILELANG_ENABLE_SIMT"] = "0"
    try:
        import tilelang
        from tilelang import tvm
        import tilelang.language as T
        from tilelang.engine.phase import LowerAndLegalize, OptimizeForTarget
        from tilelang.engine.lower import device_codegen
        from tilelang.tladapter import transforms
        from tilelang.tladapter.utils import Pipeline

        status["tilelang_version"] = tilelang.__version__
        status["stage"] = "frontend"
        width, constant_jump = args.width, args.jump

        @T.prim_func
        def sfa_copy_jump(
            A: T.Tensor((8, 4, 1, width), "float16"),
            Params: T.Tensor((4,), "int64"),
            Out: T.Tensor((2, width), "float16"),
        ):
            with T.Kernel(1, is_npu=True):
                params = T.alloc_ub((4,), "int64")
                ub = T.alloc_ub((2, width), "float16")
                T.copy(Params, params)
                page = params[0]
                lane = params[1]
                # 用于 SFA 单指令双搬运；默认在设备端从 UB 读取 jump。
                if constant_jump is None:
                    T.copy(A[page, lane, 0, 0], ub, jump=params[2])
                else:
                    T.copy(A[page, lane, 0, 0], ub, jump=constant_jump)
                T.copy(ub, Out)

        status["stage"] = "frontend"
        mod = tvm.IRModule({"sfa_copy_jump": sfa_copy_jump})
        (args.output_dir / "00_frontend.tir").write_text(mod.script())
        status["completed"].append("frontend")
        target = tvm.target.Target("npuir", host="stackvm")
        with tvm.transform.PassContext(opt_level=3):
            status["stage"] = "tir_passes"
            mod = LowerAndLegalize(mod, target)
            mod = OptimizeForTarget(mod, target)
            (args.output_dir / "01_optimized.tir").write_text(mod.script())
            status["completed"].append("tir_passes")
            status["stage"] = "npuir_codegen"
            raw = device_codegen(mod, target).get_source()
            (args.output_dir / "02_codegen.mlir").write_text(raw)
            status["completed"].append("npuir_codegen")
            status["stage"] = "pre_bishengir_pipeline"
            # Match engine.lower's non-A5 pipeline. These are in-process IR
            # passes, not the external bishengir-compile executable.
            pipeline = Pipeline()
            pipeline.add(transforms.mlir.canonicalize, top_down=True)
            pipeline.add(transforms.bishengir.adapt_triton_kernel)
            final = pipeline.run(raw)
            (args.output_dir / "03_pre_bishengir.mlir").write_text(final)
            status["completed"].append("pre_bishengir_pipeline")
        status["stage"] = "complete"
        print(args.output_dir / "03_pre_bishengir.mlir")
    except Exception:
        status["error"] = traceback.format_exc()
        (args.output_dir / "error.log").write_text(status["error"])
        raise
    finally:
        (args.output_dir / "status.json").write_text(
            json.dumps(status, indent=2, ensure_ascii=False) + "\n"
        )


if __name__ == "__main__":
    main()
