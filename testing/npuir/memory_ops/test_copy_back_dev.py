import pytest
import torch

import tilelang
import tilelang.language as T

import testcommon as tc


tilelang.cache.clear_cache()

pytestmark = pytest.mark.mode("Developer")


ROWS = 8
SRC_ROWS = 16
WIDTH = 64
DEPTH = 4
SRC2_ROWS = 5
HEIGHT = 8

EXPLICIT_ROWS = 1
EXPLICIT_SRC_ROWS = 1
EXPLICIT_SRC2_ROWS = 1
EXPLICIT_HEIGHT = 1
EXPLICIT_WIDTH = 16


def copy_back_1d_kernel(dtype="float16"):
    @T.prim_func
    def main(
        A: T.Tensor((SRC_ROWS * WIDTH,), dtype),
        Indices: T.Tensor((ROWS,), "int32"),
        Out: T.Tensor((ROWS * WIDTH,), dtype),
    ):
        with T.Kernel(ROWS, is_npu=True) as (bx, _):
            row = Indices[bx]
            T.copy_back(A[row * WIDTH], Out[bx * WIDTH], size=[WIDTH])

    return main


def copy_back_2d_kernel(dtype="float16"):
    @T.prim_func
    def main(
        A: T.Tensor((SRC_ROWS, WIDTH), dtype),
        Indices: T.Tensor((ROWS,), "int32"),
        Out: T.Tensor((ROWS, WIDTH), dtype),
    ):
        with T.Kernel(ROWS, is_npu=True) as (bx, _):
            row = Indices[bx]
            T.copy_back(A[row, 0], Out[bx, 0], size=[1, WIDTH])

    return main


def copy_back_3d_kernel(dtype="float16"):
    @T.prim_func
    def main(
        A: T.Tensor((SRC_ROWS, DEPTH, HEIGHT, WIDTH), dtype),
        Indices: T.Tensor((ROWS,), "int32"),
        Out: T.Tensor((ROWS, DEPTH, HEIGHT, WIDTH), dtype),
    ):
        with T.Kernel(ROWS, is_npu=True) as (bx, _):
            row = Indices[bx]
            T.copy_back(
                A[row, 0, 0, 0], Out[bx, 0, 0, 0], size=[1, DEPTH, HEIGHT, WIDTH]
            )

    return main


def copy_back_3d_src2_rows_kernel(dtype="float16"):
    @T.prim_func
    def main(
        A: T.Tensor((SRC_ROWS, SRC2_ROWS, HEIGHT, WIDTH), dtype),
        Indices: T.Tensor((ROWS,), "int32"),
        Out: T.Tensor((ROWS, SRC2_ROWS, HEIGHT, WIDTH), dtype),
    ):
        with T.Kernel(ROWS, is_npu=True) as (bx, _):
            row = Indices[bx]
            T.copy_back(
                A[row, 0, 0, 0],
                Out[bx, 0, 0, 0],
                size=[1, SRC2_ROWS, HEIGHT, WIDTH],
            )

    return main


def copy_back_2d_gather_src2_index_kernel(dtype="float16"):
    @T.prim_func
    def main(
        A: T.Tensor((SRC_ROWS, SRC2_ROWS, HEIGHT, WIDTH), dtype),
        Indices: T.Tensor((ROWS, SRC2_ROWS), "int32"),
        Out: T.Tensor((ROWS, SRC2_ROWS, HEIGHT, WIDTH), dtype),
    ):
        with T.Kernel(ROWS * SRC2_ROWS, is_npu=True) as (bid, _):
            bx = bid // SRC2_ROWS
            by = bid % SRC2_ROWS
            row = Indices[bx, by]
            T.copy_back(
                A[row, by, 0, 0],
                Out[bx, by, 0, 0],
                size=[1, 1, HEIGHT, WIDTH],
            )

    return main


def copy_back_2d_gather_row_col_index_kernel(dtype="float16"):
    @T.prim_func
    def main(
        A: T.Tensor((SRC_ROWS, SRC2_ROWS, HEIGHT, WIDTH), dtype),
        RowIndices: T.Tensor((ROWS, SRC2_ROWS), "int32"),
        ColIndices: T.Tensor((ROWS, SRC2_ROWS), "int32"),
        Out: T.Tensor((ROWS, SRC2_ROWS, HEIGHT, WIDTH), dtype),
    ):
        with T.Kernel(ROWS * SRC2_ROWS, is_npu=True) as (bid, _):
            row = bid // SRC2_ROWS
            col = bid % SRC2_ROWS
            row_src = RowIndices[row, col]
            col_src = ColIndices[row, col]
            T.copy_back(
                A[row_src, col_src, 0, 0],
                Out[row, col, 0, 0],
                size=[1, 1, HEIGHT, WIDTH],
            )

    return main


def copy_back_explicit_tmp_kernel(dtype="float16"):
    @T.prim_func
    def main(
        A: T.Tensor(
            (EXPLICIT_SRC_ROWS, EXPLICIT_SRC2_ROWS, EXPLICIT_HEIGHT, EXPLICIT_WIDTH),
            dtype,
        ),
        RowIndices: T.Tensor((EXPLICIT_ROWS, EXPLICIT_SRC2_ROWS), "int32"),
        ColIndices: T.Tensor((EXPLICIT_ROWS, EXPLICIT_SRC2_ROWS), "int32"),
        Out: T.Tensor(
            (EXPLICIT_ROWS, EXPLICIT_SRC2_ROWS, EXPLICIT_HEIGHT, EXPLICIT_WIDTH), dtype
        ),
    ):
        with T.Kernel(1, is_npu=True):
            tmp = T.alloc_buffer(
                (EXPLICIT_ROWS, EXPLICIT_SRC2_ROWS, EXPLICIT_HEIGHT, EXPLICIT_WIDTH),
                dtype,
                scope="shared",
            )
            for bid in T.serial(EXPLICIT_ROWS * EXPLICIT_SRC2_ROWS):
                row = bid // EXPLICIT_SRC2_ROWS
                col = bid % EXPLICIT_SRC2_ROWS
                row_src = RowIndices[row, col]
                col_src = ColIndices[row, col]
                T.copy_back(
                    A[row_src, col_src, 0, 0],
                    tmp[row, col, 0, 0],
                    size=[1, 1, EXPLICIT_HEIGHT, EXPLICIT_WIDTH],
                )
            T.copy_back(
                tmp,
                Out,
            )

    return main


@pytest.mark.op("copy_back_1d")
@pytest.mark.parametrize("dtype", ["float16"])
def test_copy_back_1d(dtype):
    torch_dtype = tc.DTYPE_MAP[dtype]
    indices = torch.tensor([3, 1, 7, 0, 5, 2, 6, 4], dtype=torch.int32).npu()
    A = torch.arange(SRC_ROWS * WIDTH, dtype=torch.float32).to(torch_dtype).npu()
    Out = torch.zeros((ROWS * WIDTH,), dtype=torch_dtype).npu()

    compiled = tilelang.compile(copy_back_1d_kernel(dtype), target="npuir")
    compiled(A, indices, Out)

    ref = A.reshape(SRC_ROWS, WIDTH)[indices.cpu().long()].reshape(ROWS * WIDTH)
    tc.assert_close(Out.cpu(), ref.cpu(), dtype=dtype)


@pytest.mark.op("copy_back_2d")
@pytest.mark.parametrize("dtype", ["float16"])
def test_copy_back_2d(dtype):
    torch_dtype = tc.DTYPE_MAP[dtype]
    indices = torch.tensor([3, 1, 7, 0, 5, 2, 6, 4], dtype=torch.int32).npu()
    A = torch.arange(SRC_ROWS * WIDTH, dtype=torch.float32).reshape(SRC_ROWS, WIDTH)
    A = A.to(torch_dtype).npu()
    Out = torch.zeros((ROWS, WIDTH), dtype=torch_dtype).npu()

    compiled = tilelang.compile(copy_back_2d_kernel(dtype), target="npuir")
    compiled(A, indices, Out)

    ref = A.cpu()[indices.cpu().long()]
    tc.assert_close(Out.cpu(), ref, dtype=dtype)


@pytest.mark.op("copy_back_3d_src2_rows")
@pytest.mark.parametrize("dtype", ["float16"])
def test_copy_back_3d_src2_rows(dtype):
    torch_dtype = tc.DTYPE_MAP[dtype]
    indices = torch.tensor([3, 1, 7, 0, 5, 2, 6, 4], dtype=torch.int32).npu()
    A = torch.arange(SRC_ROWS * SRC2_ROWS * HEIGHT * WIDTH, dtype=torch.float32)
    A = A.reshape(SRC_ROWS, SRC2_ROWS, HEIGHT, WIDTH).to(torch_dtype).npu()
    Out = torch.zeros((ROWS, SRC2_ROWS, HEIGHT, WIDTH), dtype=torch_dtype).npu()

    compiled = tilelang.compile(copy_back_3d_src2_rows_kernel(dtype), target="npuir")
    compiled(A, indices, Out)

    ref = A.cpu()[indices.cpu().long()]
    tc.assert_close(Out.cpu(), ref, dtype=dtype)


@pytest.mark.op("copy_back_2d_gather_src2_index")
@pytest.mark.parametrize("dtype", ["float16"])
def test_copy_back_2d_gather_src2_index(dtype):
    torch_dtype = tc.DTYPE_MAP[dtype]
    indices = torch.tensor(
        [
            [3, 1, 7, 0, 5],
            [2, 6, 4, 8, 10],
            [9, 11, 13, 15, 12],
            [14, 0, 2, 4, 6],
            [8, 10, 12, 14, 1],
            [3, 5, 7, 9, 11],
            [13, 15, 1, 3, 5],
            [7, 9, 11, 13, 15],
        ],
        dtype=torch.int32,
    ).npu()
    A = torch.arange(SRC_ROWS * SRC2_ROWS * HEIGHT * WIDTH, dtype=torch.float32)
    A = A.reshape(SRC_ROWS, SRC2_ROWS, HEIGHT, WIDTH).to(torch_dtype).npu()
    Out = torch.zeros((ROWS, SRC2_ROWS, HEIGHT, WIDTH), dtype=torch_dtype).npu()

    compiled = tilelang.compile(
        copy_back_2d_gather_src2_index_kernel(dtype), target="npuir"
    )
    compiled(A, indices, Out)

    indices_cpu = indices.cpu().long()
    src2 = torch.arange(SRC2_ROWS).reshape(1, SRC2_ROWS).expand(ROWS, SRC2_ROWS)
    ref = A.cpu()[indices_cpu, src2]
    tc.assert_close(Out.cpu(), ref, dtype=dtype)


@pytest.mark.op("copy_back_2d_gather_row_col_index")
@pytest.mark.parametrize("dtype", ["float16"])
def test_copy_back_2d_gather_row_col_index(dtype):
    torch_dtype = tc.DTYPE_MAP[dtype]
    row_indices = torch.tensor(
        [
            [3, 1, 7, 0, 5],
            [2, 6, 4, 8, 10],
            [9, 11, 13, 15, 12],
            [14, 0, 2, 4, 6],
            [8, 10, 12, 14, 1],
            [3, 5, 7, 9, 11],
            [13, 15, 1, 3, 5],
            [7, 9, 11, 13, 15],
        ],
        dtype=torch.int32,
    ).npu()
    col_indices = torch.tensor(
        [
            [4, 0, 3, 1, 2],
            [2, 4, 1, 3, 0],
            [1, 3, 0, 2, 4],
            [0, 2, 4, 1, 3],
            [3, 1, 2, 4, 0],
            [4, 2, 0, 3, 1],
            [1, 0, 4, 2, 3],
            [2, 3, 1, 0, 4],
        ],
        dtype=torch.int32,
    ).npu()
    A = torch.arange(SRC_ROWS * SRC2_ROWS * HEIGHT * WIDTH, dtype=torch.float32)
    A = A.reshape(SRC_ROWS, SRC2_ROWS, HEIGHT, WIDTH).to(torch_dtype).npu()
    Out = torch.zeros((ROWS, SRC2_ROWS, HEIGHT, WIDTH), dtype=torch_dtype).npu()

    compiled = tilelang.compile(
        copy_back_2d_gather_row_col_index_kernel(dtype), target="npuir"
    )
    compiled(A, row_indices, col_indices, Out)

    ref = A.cpu()[row_indices.cpu().long(), col_indices.cpu().long()]
    tc.assert_close(Out.cpu(), ref, dtype=dtype)


@pytest.mark.op("copy_back_explicit_tmp")
@pytest.mark.parametrize("dtype", ["float16"])
def test_copy_back_explicit_tmp(dtype):
    torch_dtype = tc.DTYPE_MAP[dtype]
    row_indices = torch.tensor(
        [
            [0],
        ],
        dtype=torch.int32,
    ).npu()
    col_indices = torch.tensor(
        [
            [0],
        ],
        dtype=torch.int32,
    ).npu()
    A = torch.arange(
        EXPLICIT_SRC_ROWS * EXPLICIT_SRC2_ROWS * EXPLICIT_HEIGHT * EXPLICIT_WIDTH,
        dtype=torch.float32,
    )
    A = A.reshape(
        EXPLICIT_SRC_ROWS, EXPLICIT_SRC2_ROWS, EXPLICIT_HEIGHT, EXPLICIT_WIDTH
    )
    A = A.to(torch_dtype).npu()
    Out = torch.zeros(
        (EXPLICIT_ROWS, EXPLICIT_SRC2_ROWS, EXPLICIT_HEIGHT, EXPLICIT_WIDTH),
        dtype=torch_dtype,
    ).npu()

    compiled = tilelang.compile(copy_back_explicit_tmp_kernel(dtype), target="npuir")
    compiled(A, row_indices, col_indices, Out)

    ref = A.cpu()[row_indices.cpu().long(), col_indices.cpu().long()]
    tc.assert_close(Out.cpu(), ref, dtype=dtype)


@pytest.mark.op("copy_back_3d")
@pytest.mark.parametrize("dtype", ["float16"])
def test_copy_back_3d(dtype):
    torch_dtype = tc.DTYPE_MAP[dtype]
    indices = torch.tensor([3, 1, 7, 0, 5, 2, 6, 4], dtype=torch.int32).npu()
    A = torch.arange(SRC_ROWS * DEPTH * HEIGHT * WIDTH, dtype=torch.float32)
    A = A.reshape(SRC_ROWS, DEPTH, HEIGHT, WIDTH).to(torch_dtype).npu()
    Out = torch.zeros((ROWS, DEPTH, HEIGHT, WIDTH), dtype=torch_dtype).npu()

    compiled = tilelang.compile(copy_back_3d_kernel(dtype), target="npuir")
    compiled(A, indices, Out)

    ref = A.cpu()[indices.cpu().long()]
    tc.assert_close(Out.cpu(), ref, dtype=dtype)
