import sys
from typing import Callable
import math

import torch
import torch.nn.functional as F
from torch import Tensor

class TestCase:
    idx_ = 1
    def __init__(
        self,
        input_dims: tuple[int, int, int, int], input_values: list[float],
        kernel_dims: tuple[int, int, int, int], kernel_values: list[float],
        stride: tuple[int, int], padding: tuple[int, int], dilation: tuple[int, int]
    ):
        if (len(input_values) != input_dims[0] * input_dims[1] * input_dims[2] * input_dims[3]):
            raise ValueError("Input values size does not match input shape")
        if (len(kernel_values) != kernel_dims[0] * kernel_dims[1] * kernel_dims[2] * kernel_dims[3]):
            raise ValueError("Kernel values size does not match kernel shape")
        self.idx = TestCase.idx_
        TestCase.idx_ += 1
        self.input_dims = input_dims
        self.input_values = input_values
        self.kernel_dims = kernel_dims
        self.kernel_values = kernel_values
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

def product(shape: tuple[int, int, int, int]) -> int:
    result = 1
    for dim in shape:
        result *= dim
    return result

def make_test_case (
    input_shape: tuple[int, int, int, int],
    input_seed: float,
    kernel_shape: tuple[int, int, int, int],
    kernel_seed: float,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int]
) -> TestCase:
    input_values: list[float] = [math.sin(input_seed + i) * 10.0 for i in range(product(input_shape))]
    kernel_values: list[float] = [math.cos(kernel_seed + i) * 10.0 for i in range(product(kernel_shape))]
    return TestCase(
        input_shape, input_values,
        kernel_shape, kernel_values,
        stride, padding, dilation
    )

def conv_to_string(
    input_dims: tuple[int, int, int, int], kernel_dims: tuple[int, int, int, int],
    stride: tuple[int, int], padding: tuple[int, int], dilation: tuple[int, int]
) -> str:
    out_string = "Input: "
    for dim in input_dims:
        out_string += f"{dim} "
    out_string += "\nKernel: "
    for dim in kernel_dims:
        out_string += f"{dim} "
    out_string += f"\nStride: {stride[0]}x{stride[1]}"
    out_string += f"\nPadding: {padding[0]}x{padding[1]}"
    out_string += f"\nDilation: {dilation[0]}x{dilation[1]}"
    return out_string


def tensor_to_string(tensor: Tensor) -> str:
    result: str = "Tensor(\n\tshape=["
    for i in range(len(tensor.shape)):
        result += f"{tensor.shape[i]}"
        if i < len(tensor.shape) - 1:
            result += ", "
    result += "],\n\tvalues="
    # Compute strides for row-major indexing
    N: int = tensor.dim()
    strides = []
    if (N > 0):
        strides.append(1)
        for i in range(tensor.dim() - 2, -1, -1):
            strides.insert(0, strides[0] * tensor.shape[i + 1])
    # Recursive formatter: at dimension `dim`, starting from `offset`.
    def fmt(dim, offset) -> str:
        s: str = "\n"
        for _ in range(dim):
            s += "\t"
        s += "["
        if (dim + 1 == N):
            for i in range(tensor.shape[dim]):
                if (i):
                    s += ","
                s += f"{tensor.flatten()[offset + i].item():.6f}"
        else:
            for i in range(tensor.shape[dim]):
                if (i):
                    s += ","
                s += fmt(dim + 1, offset + i * strides[dim])
        if (dim + 1 != N):
            s += "\n"
            for i in range (dim):
                s += "\t"
        s += "]"
        return s
    if (tensor.numel() == 0):
        result += "[]"
    else:
        result += fmt(0, 0)
    result += "\n)"
    return result

def run_test_case(test_case: TestCase) -> str:
    raw_inp: Tensor = Tensor(test_case.input_values).reshape(test_case.input_dims)
    inp = raw_inp.permute(0, 3, 1, 2).contiguous() # NHWC -> NCHW
    raw_ker: Tensor = Tensor(test_case.kernel_values).reshape(test_case.kernel_dims)
    ker = raw_ker.permute(3, 0, 1, 2).contiguous() # CRSK -> KCRS
    raw_out = F.conv2d(inp, ker, stride=test_case.stride, padding=test_case.padding, dilation=test_case.dilation)
    out = raw_out.permute(0, 2, 3, 1).contiguous() # NKXY -> NXYK
    return "\n".join(
        [
            f"Test {test_case.idx}",
            conv_to_string(test_case.input_dims, test_case.kernel_dims, test_case.stride, test_case.padding, test_case.dilation),
            f"Output:\n{tensor_to_string(out)}",
            f"Input:\n{tensor_to_string(raw_inp)}",
            f"Kernel:\n{tensor_to_string(raw_ker)}",
        ]
    )

CASES: list[
    tuple[
        tuple[int, int, int, int], float,
        tuple[int, int, int, int], float,
        tuple[int, int], tuple[int, int], tuple[int, int]
    ]
] = [
    ((1, 4, 4, 1), 0.0, (1, 3, 3, 2), 0.0, (1, 1), (0, 0), (1, 1)),
    ((1, 8, 8, 1), 1.0, (1, 3, 3, 2), 3.0, (1, 1), (0, 0), (1, 1)),
    ((4, 8, 8, 1), 2.0, (1, 3, 3, 2), 6.0, (2, 2), (1, 1), (1, 1)),
    ((1, 8, 8, 2), 2.0, (2, 3, 3, 2), 6.0, (1, 1), (1, 1), (1, 1)),
    ((4, 16, 16, 3), 3.0, (3, 3, 3, 2), 9.0, (1, 2), (1, 2), (1, 1)),
    ((4, 16, 16, 4), 3.0, (4, 3, 3, 8), 9.0, (1, 1), (1, 1), (2, 2)),
    ((4, 16, 16, 4), 4.0, (4, 3, 3, 8), 10.0, (1, 1), (1, 1), (2, 1)),
    ((4, 16, 16, 4), 5.0, (4, 1, 2, 8), 11.0, (1, 1), (2, 1), (1, 2)),
    ((4, 16, 16, 4), 5.0, (4, 2, 3, 8), 11.0, (1, 1), (2, 1), (1, 2)),
    ((1, 5, 7, 4), 6.0, (4, 1, 1, 6), 12.0, (1, 1), (0, 0), (1, 1)),
    ((2, 10, 6, 2), 7.0, (2, 5, 1, 3), 13.0, (1, 1), (2, 0), (1, 1)),
    ((2, 6, 10, 2), 8.0, (2, 1, 5, 3), 14.0, (1, 1), (0, 2), (1, 1)),
    ((1, 15, 15, 3), 9.0, (3, 3, 3, 4), 15.0, (3, 3), (1, 1), (1, 1)),
    ((1, 12, 12, 2), 10.0, (2, 3, 3, 4), 16.0, (1, 1), (0, 0), (3, 3)),
    ((2, 9, 9, 3), 11.0, (3, 5, 5, 4), 17.0, (1, 1), (2, 2), (1, 1)),
    # Dilated 5x5: tiny-dnn's AVX backend dispatches on kernel size alone and its
    # 5x5 kernel ignores dilation, so Conv2dTinydnn pins this shape to the scalar
    # engine. Keep a case here so that guard stays covered.
    ((2, 12, 12, 3), 11.0, (3, 5, 5, 4), 17.0, (1, 1), (2, 2), (2, 2)),
    ((2, 14, 10, 3), 12.0, (3, 3, 2, 5), 18.0, (2, 3), (1, 0), (2, 1)),
    ((1, 8, 8, 8), 13.0, (8, 3, 3, 8), 19.0, (1, 1), (1, 1), (1, 1)),
    ((2, 7, 7, 5), 14.0, (5, 2, 2, 3), 20.0, (2, 2), (0, 0), (1, 1)),
    ((1, 3, 3, 2), 15.0, (2, 3, 3, 4), 21.0, (1, 1), (2, 2), (1, 1)),
    ((1, 1, 1, 3), 16.0, (3, 1, 1, 4), 22.0, (1, 1), (0, 0), (1, 1)),
    ((1, 20, 20, 3), 17.0, (3, 7, 7, 4), 23.0, (2, 2), (3, 3), (2, 2)),
    ((8, 6, 6, 2), 18.0, (2, 3, 3, 4), 24.0, (1, 1), (1, 1), (1, 1))
]

def main() -> None:
    if len(sys.argv) != 2:
        raise ValueError("Invalid number of arguments")
    output_file = sys.argv[1]
    with open(output_file, "w", encoding="utf-8") as file:
        for test_case in CASES:
            file.write(run_test_case(make_test_case(*test_case)))
            file.write("\n")


if __name__ == "__main__":
    main()
