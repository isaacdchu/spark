#ifndef IM2COL_HPP
#define IM2COL_HPP

#include <array>
#include <algorithm>
#include <numeric>
#include <functional>
#include <memory>
#include <type_traits>
#include <vector>
#include <cstddef>

#include <taskflow.hpp>
#include <algorithm/for_each.hpp>
#include <cblas.h>

#include "tensor.hpp"
#include "concepts.hpp"
#include "util.hpp"

namespace sc {

template <FloatLike T>
void im2col2d(
    const Tensor<T, 4>& input,
    const Tensor<T, 4>& kernel,
    const std::array<std::size_t, 2>& stride,
    const std::array<std::size_t, 2>& padding,
    const std::array<std::size_t, 2>& dilation,
    std::size_t n,
    Tensor<T, 2>& out
) {
    // Assumes out is Tensor<T, 2>(output_shape[1] * output_shape[2], kernel.size() / kernel.shape()[3]);
    // NHWC/CRSK format
    std::array<std::size_t, 4> output_shape = calculate_output_shape(
        input.shape(), kernel.shape(),
        stride, padding, dilation
    );
    // NHWC -> NPQ x CRS
    // Isolate sample n
    const std::size_t H = input.shape()[1];
    const std::size_t W = input.shape()[2];
    const std::size_t C = input.shape()[3];
    const std::size_t K = kernel.shape()[3];
    const std::size_t P = output_shape[1];
    const std::size_t Q = output_shape[2];
    const std::size_t R = kernel.shape()[1];
    const std::size_t S = kernel.shape()[2];
    const std::size_t U = stride[0];
    const std::size_t V = stride[1];
    const std::size_t PADH = padding[0];
    const std::size_t PADW = padding[1];
    const std::size_t DILH = dilation[0];
    const std::size_t DILW = dilation[1];
    const T* input_data = input.values().data() + n * input.stride(0);
    // Fill in the matrix
    T* out_ptr = out.values().data();
    for (std::size_t p = 0; p < P; p++) {
        for (std::size_t q = 0; q < Q; q++) {
            for (std::size_t c = 0; c < C; c++) {
                for (std::size_t r = 0; r < R; r++) {
                    for (std::size_t s = 0; s < S; s++) {
                        const std::ptrdiff_t h = static_cast<std::ptrdiff_t>(p * U + r * DILH)
                            - static_cast<std::ptrdiff_t>(PADH);
                        const std::ptrdiff_t w = static_cast<std::ptrdiff_t>(q * V + s * DILW)
                            - static_cast<std::ptrdiff_t>(PADW);
                        if (h >= 0 && w >= 0 && h < static_cast<std::ptrdiff_t>(H) && w < static_cast<std::ptrdiff_t>(W)) {
                            *out_ptr = input_data[
                                static_cast<std::size_t>(h) * W * C + static_cast<std::size_t>(w) * C + c
                            ];
                        } else {
                            *out_ptr = 0;
                        }
                        ++out_ptr;
                    }
                }
            }
        }
    }
}

template <FloatLike T>
void im2col2d(const Tensor<T, 4>& kernel, Tensor<T, 2>& out) {
    // Assumes out is Tensor<T, 2>(kernel.size() / kernel.shape()[3], kernel.shape()[3]);
    // CRSK -> CRS x K
    out.values() = kernel.values();
}

template <FloatLike T>
void col2im2d(
    const Tensor<T, 2>& output_matrix,
    const std::array<std::size_t, 4>& output_shape,
    std::size_t n,
    Tensor<T, 4>& out
) {
    // Assumes out is Tensor<T, 4>(output_shape);
    // Fill in slice n of output tensor
    auto output_data_it = out.values().begin() + n * out.stride(0);
    std::copy(output_matrix.values().begin(), output_matrix.values().end(), output_data_it);
}

} // namespace sc

#endif // IM2COL_HPP