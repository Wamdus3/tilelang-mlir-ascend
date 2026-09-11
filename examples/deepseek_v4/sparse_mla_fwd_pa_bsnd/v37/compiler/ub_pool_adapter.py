"""Apply a caller-supplied UB layout to static, contiguous MLIR allocations.

This module contains no operator layout or kernel-name dispatch. Callers supply
pool capacities, typed-shape placements, optional entries and dimension bounds.
Overlapping ranges are permitted: lifetime/synchronization safety is the
caller's responsibility. One call describes one UB allocation domain.
"""

import math
import re
from collections import Counter

ELEMENT_BYTES = {"bf16": 2, "f32": 4, "i32": 4, "i1": 1}
ALLOCATION = re.compile(
    r"^(\s*)(%[\w]+) = memref.alloc\(\) : "
    r"(memref<([0-9x]+)(bf16|f32|i32|i1), "
    r"strided<\[([0-9, ]+)\]>, #hivm.address_space<ub>>)\s*$"
)


def place(
    source,
    *,
    pools,
    layout,
    optional_allocations=(),
    dimension_limits=None,
    alignment=32,
):
    """Return rewritten IR and a placement receipt.

    ``layout[(shape, dtype)]`` lists ``(pool, byte_offset)`` per occurrence.
    Shape dimensions may be None to match any positive static extent; optional
    ``dimension_limits[key]`` specifies per-axis upper bounds (None = unbounded).
    Exact shape keys take precedence over wildcard keys; ambiguous wildcard
    matches are rejected. Optional keys accept zero through their listed count.
    All allocations must be contiguous, supported scalar types in UB memory.
    """
    dimension_limits = {} if dimension_limits is None else dimension_limits
    if alignment <= 0:
        raise ValueError("Alignment must be positive")
    if (
        set(optional_allocations) - layout.keys()
        or dimension_limits.keys() - layout.keys()
    ):
        raise ValueError("Constraints reference an unknown layout entry")
    for pool, capacity in pools.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", pool) or capacity <= 0:
            raise ValueError(f"Invalid UB pool: {pool}, {capacity}")
    for key, entries in layout.items():
        shape, dtype = key
        if (
            not shape
            or dtype not in ELEMENT_BYTES
            or any(dim is not None and dim <= 0 for dim in shape)
        ):
            raise ValueError(f"Invalid layout key: {key}")
        limits = dimension_limits.get(key, (None,) * len(shape))
        if len(limits) != len(shape) or any(
            limit is not None and limit <= 0 for limit in limits
        ):
            raise ValueError(f"Invalid dimension limits: {key}")
        for pool, offset in entries:
            if pool not in pools or offset < 0 or offset % alignment:
                raise ValueError(f"Invalid UB placement: {key}, {pool}, {offset}")
    seen = Counter()
    placements = []
    lines = []
    for line in source.splitlines():
        match = ALLOCATION.match(line)
        if not match:
            if "memref.alloc" in line and "address_space<ub>" in line:
                raise ValueError("Unrecognized UB allocation: " + line)
            lines.append(line)
            continue
        indent, name, typ, dims, dtype, strides = match.groups()
        shape = tuple(int(x) for x in dims.rstrip("x").split("x"))
        key = (shape, dtype)
        if key not in layout:
            matches = [
                candidate
                for candidate in layout
                if candidate[1] == dtype
                and len(candidate[0]) == len(shape)
                and all(
                    want is None or want == actual
                    for want, actual in zip(candidate[0], shape, strict=True)
                )
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Unlisted or ambiguous UB allocation: {shape}, {dtype}"
                )
            key = matches[0]
        limits = dimension_limits.get(key, (None,) * len(shape))
        if any(
            dim <= 0 or (limit is not None and dim > limit)
            for dim, limit in zip(shape, limits, strict=True)
        ):
            raise ValueError(f"UB dimension out of bounds: {shape}, {limits}")
        expected_strides = tuple(math.prod(shape[i + 1 :]) for i in range(len(shape)))
        if tuple(int(x) for x in strides.split(",")) != expected_strides:
            raise ValueError(f"UB allocation must be contiguous: {name}")
        occurrence = seen[key]
        if occurrence >= len(layout[key]):
            raise ValueError(f"Too many UB allocations: {key}")
        seen[key] += 1
        pool, offset = layout[key][occurrence]
        size = math.prod(shape) * ELEMENT_BYTES[dtype]
        if offset < 0 or offset % alignment or offset + size > pools[pool]:
            raise ValueError(f"Invalid UB placement: {name}, {pool}, {offset}, {size}")
        if not placements:
            for pool_name, capacity in pools.items():
                lines.append(
                    f"{indent}%ubpool_{pool_name} = memref.alloc() : "
                    f"memref<{capacity}xi8, #hivm.address_space<ub>>"
                )
        index = len(placements)
        identity = f"memref<{dims}{dtype}, #hivm.address_space<ub>>"
        raw = f"memref<{pools[pool]}xi8, #hivm.address_space<ub>>"
        lines.extend(
            [
                f"{indent}%uboffset_{index} = arith.constant {offset} : index",
                f"{indent}%ubview_{index} = memref.view %ubpool_{pool}[%uboffset_{index}][] : {raw} to {identity}",
                f"{indent}{name} = memref.cast %ubview_{index} : {identity} to {typ}",
            ]
        )
        placements.append(
            dict(
                ssa=name,
                shape=shape,
                dtype=dtype,
                pool=pool,
                offset=offset,
                bytes=size,
                occurrence=occurrence,
            )
        )
    for key, entries in layout.items():
        minimum = 0 if key in optional_allocations else len(entries)
        if not minimum <= seen[key] <= len(entries):
            raise ValueError(
                f"UB allocation count: {key}, expected {minimum}..{len(entries)}, actual {seen[key]}"
            )
    return "\n".join(lines) + "\n", {
        "pools": dict(pools),
        "physical_bytes": sum(pools.values()),
        "placements": placements,
    }
