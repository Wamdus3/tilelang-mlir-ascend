# Tilelang.language.set_atomic

`T.set_atomic(kind, dtype="float32")` explicitly sets the atomic mode for
subsequent GM writes on the current core. Supported kinds are `add`, `max`,
`min` and `none`; supported types are `float16`, `float32`, `bfloat16`, `int8`,
`int16` and `int32`. `T.set_atomic_add(dtype)` and `T.set_atomic_none(dtype)`
are convenience calls for enabling addition and resetting the mode.

The call lowers directly to `hivm.hir.set_atomic`. Build TileLang against an
AscendNPU-IR library containing `SetAtomicOp`, including matching generated
headers. With older IR dependencies, existing kernels remain usable, but
lowering an explicit atomic call reports the missing capability.

Atomic mode is stateful. These APIs do not insert synchronization. The caller
must synchronize outstanding writes before enabling or disabling it, match the
configured element type, and reset the mode before ordinary writes and before
returning from the kernel. For UB-to-GM copies use `PIPE_MTE3`; for L0C-to-GM
Fixpipe stores use `PIPE_FIX`. Cross-pipeline/core dependencies still require
their normal flags or barriers.

```python
with T.rs("PIPE_MTE3"):
    T.pipe_barrier("PIPE_MTE3")
    T.set_atomic_add("float32")
    for chunk in T.serial(num_chunks):
        T.copy(src_ub, output[chunk * width : (chunk + 1) * width])
    T.pipe_barrier("PIPE_MTE3")
    T.set_atomic_none("float32")
```

Use ordinary `T.copy` or `T.store_fixpipe` inside an explicit region. Do not mix
it with `T.atomic_add`, whose per-store atomic attribute may independently
configure and reset the atomic state. The compiler preserves the explicit
calls as opaque side effects; it does not infer balanced regions or prove
that all control-flow paths restore the state.
