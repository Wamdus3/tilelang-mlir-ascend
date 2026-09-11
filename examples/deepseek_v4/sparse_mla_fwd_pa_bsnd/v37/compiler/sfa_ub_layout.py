"""Sparse MLA UB layout data; offsets are relative to compiler-allocated pools.

Aliased ranges follow the kernel's manual synchronization and lifetime handoffs.
Entries of the same typed shape follow allocation order.
"""

POOLS = {
    "input1": 32768,
    "input2": 16384,
    "output1": 32768,
    "output2": 4096,
    "tmp1": 32768,
    "v0meta": 8192,
    "stats": 14336,
    "input1_hi": 32768,
}

# (shape, dtype) -> (pool, byte offset) for each allocation occurrence.
# None denotes the checked, variable-length block-table dimension.
LAYOUT = {
    ((32, 512), "bf16"): (("input1", 0), ("input1_hi", 0)),
    ((32, 64), "bf16"): (("input2", 0), ("input2", 4096)),
    ((2, 1, 512), "i32"): (("input2", 8192),),
    ((16, 512), "i32"): (("output1", 0),),
    ((16, 128), "i32"): (("output1", 0),),
    ((16, 512), "bf16"): (("output1", 0),),
    ((16, 1), "i32"): (("output2", 0),),
    ((16, 512), "f32"): (("input1", 0), ("tmp1", 0)),
    ((16, 128), "f32"): (("input1", 0), ("tmp1", 0)),
    ((16, 512), "i1"): (("input1_hi", 0),),
    ((16, 128), "i1"): (("input1_hi", 0),),
    ((2, 1), "i32"): (("v0meta", 0),),
    ((None,), "i32"): (("v0meta", 32),),
    ((16, 1), "bf16"): (("stats", 0),),
    ((16, 1), "f32"): tuple(
        ("stats", offset)
        for offset in (
            32,
            96,
            160,
            224,
            288,
            352,
            416,
            608,
            672,
            4448,
            4512,
            4576,
            4640,
            4736,
        )
    ),
    ((32, 1), "f32"): (("stats", 480), ("stats", 4320)),
    ((1, 512), "i1"): (("stats", 736), ("stats", 1248), ("stats", 1760)),
    ((1, 512), "f32"): (("stats", 2272), ("stats", 6880), ("stats", 8960)),
    ((16, 1), "i1"): (("stats", 4704),),
    ((1, 512), "i32"): (("stats", 4800),),
    ((1, 1), "i32"): (("stats", 6848),),
    ((1, 1), "f32"): (("stats", 8928),),
}

# A single key tile has no inter-tile exponent update, so its update scratch
# allocation is eliminated. All other entries require every listed placement.
OPTIONAL_ALLOCATIONS = {((16, 128), "i32")}

# Bounds and optional entries are declarative parts of this kernel's contract.
UB_LAYOUT = {
    "pools": POOLS,
    "layout": LAYOUT,
    "optional_allocations": OPTIONAL_ALLOCATIONS,
    "dimension_limits": {((None,), "i32"): (1565,)},
    "alignment": 32,
}
