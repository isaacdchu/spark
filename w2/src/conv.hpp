#ifndef CONV_HPP
#define CONV_HPP

#include <array>
#include <vector>
#include <stdexcept>
#include <chrono>
#include <cstddef>

#include <taskflow.hpp>
#include <cblas.h>

#include "tensor.hpp"
#include "concepts.hpp"
#include "im2col.hpp"
#include "util.hpp"

namespace sc {

enum class Padding {
    VALID,
    NONE,
    SAME
};

static std::array<std::size_t, 2> calculate_padding(
    const std::array<std::size_t, 4>& input_shape,
    const std::array<std::size_t, 4>& kernel_shape,
    const std::array<std::size_t, 2>& stride,
    const std::array<std::size_t, 2>& dilation
) {
    std::array<std::size_t, 2> padding;
    std::ptrdiff_t in_h = static_cast<std::ptrdiff_t>(input_shape[2]);
    std::ptrdiff_t in_w = static_cast<std::ptrdiff_t>(input_shape[3]);
    std::ptrdiff_t s_h = static_cast<std::ptrdiff_t>(stride[0]);
    std::ptrdiff_t s_w = static_cast<std::ptrdiff_t>(stride[1]);
    std::ptrdiff_t k_h = static_cast<std::ptrdiff_t>(kernel_shape[2]);
    std::ptrdiff_t k_w = static_cast<std::ptrdiff_t>(kernel_shape[3]);
    std::ptrdiff_t d_h = static_cast<std::ptrdiff_t>(dilation[0]);
    std::ptrdiff_t d_w = static_cast<std::ptrdiff_t>(dilation[1]);

    std::ptrdiff_t pad_total_h = (in_h - 1) * s_h + d_h * (k_h - 1) + 1 - in_h;
    if (pad_total_h < 0) pad_total_h = 0;
    if (pad_total_h % 2 == 1) {
        throw std::invalid_argument("Padding height is odd");
    }
    padding[0] = static_cast<std::size_t>(pad_total_h / 2);

    std::ptrdiff_t pad_total_w = (in_w - 1) * s_w + d_w * (k_w - 1) + 1 - in_w;
    if (pad_total_w < 0) pad_total_w = 0;
    if (pad_total_w % 2 == 1) {
        throw std::invalid_argument("Padding width is odd");
    }
    padding[1] = static_cast<std::size_t>(pad_total_w / 2);
    return padding;
}

template <FloatLike T>
static Tensor<T, 2> matmul(
    const Tensor<T, 2>& a,
    const Tensor<T, 2>& b
) {
    // Implementation for matrix multiplication
    Tensor<T, 2> result(a.shape()[0], b.shape()[1]);

    auto safe_int = [](std::size_t v) -> int {
        if (v > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::overflow_error("matrix dimension too large for BLAS int");
        }
        return static_cast<int>(v);
    };

    int m = safe_int(a.shape()[0]);
    int n = safe_int(b.shape()[1]);
    int k = safe_int(a.shape()[1]);
    int lda = k; // row-major: leading dimension for A (M x K)
    int ldb = n; // row-major: leading dimension for B (K x N)
    int ldc = n; // leading dimension for result (M x N)

    if constexpr (std::is_same_v<T, float>) {
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            m, n, k,
            1.0f,
            a.values().data(), lda,
            b.values().data(), ldb,
            0.0f,
            result.values().data(), ldc
        );
    } else {
        cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            m, n, k,
            1.0,
            a.values().data(), lda,
            b.values().data(), ldb,
            0.0,
            result.values().data(), ldc
        );
    }
    return result;
}

// Symmetric hyperparameters
template <FloatLike T>
Tensor<T, 4> conv2d(
    const Tensor<T, 4>& input,
    const Tensor<T, 4>& kernel,
    std::size_t stride,
    std::size_t padding,
    std::size_t dilation = 1
) {
    return conv2d(input, kernel, {stride, stride}, {padding, padding}, {dilation, dilation});
}

// Default hyperparameters and enum padding
template <FloatLike T>
Tensor<T, 4> conv2d(
    const Tensor<T, 4>& input,
    const Tensor<T, 4>& kernel,
    Padding padding = Padding::VALID,
    const std::array<std::size_t, 2>& stride = {1, 1},
    const std::array<std::size_t, 2>& dilation = {1, 1}
) {
    switch (padding) {
        case Padding::VALID:
        case Padding::NONE:
            return conv2d(input, kernel, stride, {0, 0}, dilation);
        case Padding::SAME:
            return conv2d(
                input, kernel,
                stride,
                calculate_padding(input.shape(), kernel.shape(), stride, dilation),
                dilation
            );
        default:
            throw std::invalid_argument("Invalid padding type");
    }
}

// Symmetric hyperparameters with enum padding
template <FloatLike T>
Tensor<T, 4> conv2d(
    const Tensor<T, 4>& input,
    const Tensor<T, 4>& kernel,
    std::size_t stride,
    Padding padding,
    std::size_t dilation = 1
) {
    return conv2d(input, kernel, padding, {stride, stride}, {dilation, dilation});
}

// Nonsymmetric hyperparameters with enum padding
template <FloatLike T>
Tensor<T, 4> conv2d(
    const Tensor<T, 4>& input,
    const Tensor<T, 4>& kernel,
    const std::array<std::size_t, 2>& stride,
    Padding padding,
    const std::array<std::size_t, 2>& dilation
) {
    return conv2d(input, kernel, padding, stride, dilation);
}

// Nonsymmetric hyperparameters without enum padding
inline struct {
    std::chrono::microseconds im2col;
    std::chrono::microseconds matmul;
    std::chrono::microseconds col2im;
} total_times;
template <FloatLike T>
Tensor<T, 4> conv2d(
    const Tensor<T, 4>& input,
    const Tensor<T, 4>& kernel,
    const std::array<std::size_t, 2>& stride,
    const std::array<std::size_t, 2>& padding,
    const std::array<std::size_t, 2>& dilation
) {
    // Input and kernel must be 4D tensors
    // Implementation for 2D convolution on last 2 dimensions of the input tensor
    // Calculate the output shape based on the input shape, kernel size, stride, padding, and dilation
    std::array<std::size_t, 4> output_shape = calculate_output_shape(
        input.shape(),
        kernel.shape(),
        stride,
        padding,
        dilation
    );
    Tensor<T, 4> output(output_shape);
    // im2col
    auto start = std::chrono::high_resolution_clock::now();
    auto [input_matrix, kernel_matrix] = im2col2d(input, kernel, stride, padding, dilation);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    total_times.im2col += duration;
    // matrix multiplication
    start = std::chrono::high_resolution_clock::now();
    Tensor<T, 2> output_matrix = matmul(kernel_matrix, input_matrix);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    total_times.matmul += duration;
    // col2im
    start = std::chrono::high_resolution_clock::now();
    output = col2im2d(output_matrix, output_shape);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    total_times.col2im += duration;
    return output;
}

} // namespace sc

#endif // CONV_HPP