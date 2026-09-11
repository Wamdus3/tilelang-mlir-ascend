# Tilelang.language.copy

## 1. OP概述

简介：`tilelang.language.copy` 该算子用于不同内存区域之间（ub-ub， gm-ub）执行数据复制操作。

```python
T.copy(src[0:size], dst[0:size] ) [Developer Op]
T.copy(src[0:size], dst[0:size] ) [Expert Op]
或
T.copy(src, dst, size) [Developer Op]
T.copy(src, dst, size) [Expert Op]
```

## 2. OP规格

### 2.1 参数说明

| 参数名    | 类型         | 说明       |
| ----------- | -------------- | ------------ |
| `src` | `tensor` | 输入tensor |
| `dst` | `tensor` | 输出tensor |
| `size` | `list(int)` | 拷贝数据的size |

### 2.2 支持规格

#### 2.2.1 DataType支持

|        | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| -------- | ------- | ------ | -------- | ------- | -------- | ------- | -------- | ------- | ------ | ------ | ------ | ----------- |
| Ascend | √    | √   | √     | √    | √     | √    | √     | √    | √   |√   | √   | ×        |

#### 2.2.2 Shape支持

结论：输入（input）与输出（output）的shape要一致。

### 2.3 特殊限制说明

无

### 2.4 使用方法

示例1：实现了Expert Mode中将一个二维张量（Tensor）copy到A_ub中 (gm -> ub)

```python
@tilelang.jit(target="npuir")
def atomic_add_2d(M, N, block_M, block_N, dtype="float32"):
    m_blocks = M // block_M
    n_blocks = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_blocks * n_blocks, is_npu=True) as (cid, _):
            bx = (cid // n_blocks) * block_M
            by = (cid % n_blocks) * block_N
            A_ub = T.alloc_ub((block_M, block_N), dtype)
            tile_M = T.min(block_M, M - bx)
            tile_N = T.min(block_N, N - by)
            T.copy(
                A[bx : bx + tile_M, by : by + tile_N],
                A_ub[0:tile_M, 0:tile_N],
            )
            T.npuir_atomic_add(B[bx, by], A_ub, [tile_M, tile_N])

    return main
```

示例2：实现了Developer Mode中将一个二维张量（Tensor） 的copy到A_shared中 (gm -> ub)

```python
@tilelang.jit(target="npuir")
def atomic_add_2d_dev(M, N, block_M, block_N, dtype="float32"):
    m_blocks = M // block_M
    n_blocks = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_blocks * n_blocks, is_npu=True) as (cid, _):
            bx = (cid // n_blocks) * block_M
            by = (cid % n_blocks) * block_N
            A_shared = T.alloc_shared((block_M, block_N), dtype)
            tile_M = T.min(block_M, M - bx)
            tile_N = T.min(block_N, N - by)
            T.copy(
                A[bx : bx + tile_M, by : by + tile_N],
                A_shared[0:tile_M, 0:tile_N],
            )
            T.npuir_atomic_add(B[bx, by], A_shared, [tile_M, tile_N])

    return main
```

## 3. Tilelang Op到Ascend NPU IR Op的转换

**1. 在expert 模式下：**

当src和dst的shape一致时：**tilelang::copyOp**将被转换为memref::CopyOp

否则：**tilelang::copyOp**将被转换为memref::ExtractStridedMetadataOp、memref::DimOp（动态shape）、arith::ConstantIndexOp（非动态shape）、memref::ReinterpretCastOp、memref::CopyOp

**2. 在developer 模式下：**

**GM -> UB:  ​tilelang::copyOp**将被转换为 memref::SubViewOp、memref::AllocOp、bufferization::ToTensorOp、（tensor::DimOp、tensor::EmptyOp、hivm::VCastOp）[for type cast]、tensor::InsertSliceOp

**UB​​​ -> UB​:  tilelang::copyOp**将被转换为 tensor::ExtractSliceOp、（tensor::DimOp、tensor::EmptyOp、hivm::VCastOp）[for type cast]、tensor::InsertSliceOp

**UB -> GM:  tilelang::copyOp**将被转换为 tensor::ExtractSliceOp、（tensor::DimOp、tensor::EmptyOp、hivm::VCastOp）[for type cast]、bufferization::MaterializeInDestinationOp

## 4. jump：SFA 单指令双搬运（首版仅非 A5 Expert）

```python
T.copy(A[p, 0], ub[slot:slot + 2, :W], jump=pitch)
T.copy(A[p, 0], ub[slot, 0], size=[2, W], jump=pitch)
```

`jump` 是有符号、以元素计的两个源块起点间距，不是字节 gap。
展平源起点为 `base`，另一起点为 `base + pitch`。目标第 0 行取两个起点中地址较低的一段，
第 1 行取地址较高的一段，各复制 W 个元素。负 jump 因此交换调用者给定的两行；
该顺序规则对快速路径与所有回退路径均成立。不转换 dtype，不筛除无效 token。
两个完整源块及目标切片必须位于有效分配内，地址运算必须在 int64 范围内。
源必须是连续 GM 参数 Buffer 的标量起点；目标为连续二维 UB Buffer/切片。
W 为正静态整数，行长度须 32B 对齐且不大于 2097120B；目标起点及行距须可证明 32B 对齐。
不支持 `T.Parallel`、与 `coalesced_width` 混用、动态 W、动态行数、隐式 dtype 转换。
无 jump 的原有调用行为保持不变；其他 target、Developer、A5 明确报错。

GM 两个源区间可以来自两个独立随机 offset，设备端以 `offset1 - offset0` 计算 jump。
正序或逆序、不重叠且绝对间距位于 DMA gap 范围时，只要目标两行物理连续（UB 行距等于 W），
即可生成二维 strided memref.copy，供后端选择一次两块 GM→UB DMA；GM 两段本身不必相邻。
逆序时从 `base + jump` 开始，以 `-jump` 为正行距生成一次双块搬运；不增加 UB 临时区或行重排。
重复、重叠、超过保守 gap 上限或目标行间有空隙时也使用两次单行 copy。
带间隔 UB 的回退规避当前 CANN 9 对带列偏移、非连续二维目标行距的处理问题。
KV 与 RoPE 必须使用同一对 token indices，且各自 GM 布局对这两个 indices 的地址顺序一致。
例如 KV/RoPE 都按 token 递增存储时，可分别用 `(index1-index0)*512` 和 `(index1-index0)*64`。
两次 copy 的 jump 符号相同，因此输出行一一对应；即使其中一个回退，顺序也不变。
KV/RoPE 配对时，先从 indices UB 读取一次标量并共用：

```python
index0 = indices_ub[0]
index1 = indices_ub[1]
T.copy(KV[index0, 0], kv_ub, jump=(index1 - index0) * T.int64(512))
T.copy(Rope[index0, 0], rope_ub, jump=(index1 - index0) * T.int64(64))
```

这里要求 indices 为 int64，避免减法在较窄整数中溢出。当前 CANN 9 实测中，
在两次搬运表达式里分别重新读取 indices UB 的写法，在混合带间隔 UB 回退时出现过错误的第 0 行读取；
上述共享标量写法已通过设备验证。该后端/运行时问题尚未定位到具体 pass。
其他依赖 token 顺序的数据（如 V、mask、位置元数据）也须与该顺序保持对应；普通无 jump 的 copy 语义不变。
单条硬件指令属于快速路径优化目标，必须用最终指令/trace 验证，不能由一个 T.copy 推定。
