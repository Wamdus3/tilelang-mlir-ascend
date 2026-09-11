// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.
#pragma once

#include "../op/ascend.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/IR/Builders.h"

namespace tvm {
namespace codegen {

inline void EmitSetAtomic(mlir::OpBuilder &builder, const tir::CallNode *op,
                          tl::BufferMap vmap) {
  tl::NpuirSetAtomic atomic(op->args, vmap);
#ifdef TILELANG_HAS_HIVM_SET_ATOMIC
  auto kind = mlir::hivm::AtomicKind::NONE;
  if (atomic.kind == "add")
    kind = mlir::hivm::AtomicKind::ADD;
  if (atomic.kind == "max")
    kind = mlir::hivm::AtomicKind::MAX;
  if (atomic.kind == "min")
    kind = mlir::hivm::AtomicKind::MIN;
  mlir::Type type;
  if (atomic.dtype == "float16")
    type = builder.getF16Type();
  else if (atomic.dtype == "float32")
    type = builder.getF32Type();
  else if (atomic.dtype == "bfloat16")
    type = builder.getBF16Type();
  else if (atomic.dtype == "int8")
    type = builder.getIntegerType(8);
  else if (atomic.dtype == "int16")
    type = builder.getIntegerType(16);
  else
    type = builder.getIntegerType(32);
  builder.create<mlir::hivm::SetAtomicOp>(
      builder.getUnknownLoc(),
      mlir::hivm::AtomicKindAttr::get(builder.getContext(), kind),
      mlir::TypeAttr::get(type));
#else
  LOG(FATAL) << "T.set_atomic requires an AscendNPU-IR build with SetAtomicOp; "
                "rebuild TileLang against that build's headers and libraries";
#endif
}

} // namespace codegen
} // namespace tvm
