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

enum class Im2ColBackend {
    STANDARD,
    TASKFLOW,
    OPENMP
};

constexpr const char* im2col_backend_to_string(Im2ColBackend backend) {
    switch (backend) {
        case Im2ColBackend::STANDARD: return "STANDARD";
        case Im2ColBackend::TASKFLOW: return "TASKFLOW";
        case Im2ColBackend::OPENMP: return "OPENMP";
        default: return "UNKNOWN";
    }
}

Im2ColBackend IM2COL_BACKEND = Im2ColBackend::STANDARD;
inline tf::Executor executor;

template <FloatLike T>
std::array<Tensor<T, 2>, 2> im2col2d_standard(
    const Tensor<T, 4>& input,
    const Tensor<T, 4>& kernel,
    const std::array<std::size_t, 2>& stride,
    const std::array<std::size_t, 2>& padding,
    const std::array<std::size_t, 2>& dilation
) {
    const std::array<std::size_t, 4> output_shape = calculate_output_shape(
        input.shape(),
        kernel.shape(),
        stride,
        padding,
        dilation
    );
    const std::array<std::size_t, 2> input_matrix_shape = {
        input.shape()[1] * kernel.shape()[2] * kernel.shape()[3],
        input.shape()[0] * output_shape[2] * output_shape[3]
    };
    Tensor<T, 2> input_matrix(input_matrix_shape);
    // Standard im2col: rows = C_in * K_h * K_w, cols = N * out_h * out_w
    const std::size_t N = input.shape()[0];
    const std::size_t C = input.shape()[1];
    const std::size_t H = input.shape()[2];
    const std::size_t W = input.shape()[3];
    const std::size_t out_h = output_shape[2];
    const std::size_t out_w = output_shape[3];
    const std::size_t k_h = kernel.shape()[2];
    const std::size_t k_w = kernel.shape()[3];
    const std::size_t cols = N * out_h * out_w;
    const std::size_t rows = C * k_h * k_w;
    const std::ptrdiff_t stride_h = static_cast<std::ptrdiff_t>(stride[0]);
    const std::ptrdiff_t stride_w = static_cast<std::ptrdiff_t>(stride[1]);
    const std::ptrdiff_t pad_h = static_cast<std::ptrdiff_t>(padding[0]);
    const std::ptrdiff_t pad_w = static_cast<std::ptrdiff_t>(padding[1]);
    const std::ptrdiff_t dilation_h = static_cast<std::ptrdiff_t>(dilation[0]);
    const std::ptrdiff_t dilation_w = static_cast<std::ptrdiff_t>(dilation[1]);
    // Fill input_matrix (row-major: row * cols + col)
    for (std::size_t n = 0; n < N; n++) {
        for (std::size_t c = 0; c < C; c++) {
            for (std::size_t ki = 0; ki < k_h; ki++) {
                for (std::size_t kj = 0; kj < k_w; kj++) {
                    const std::size_t row = c * (k_h * k_w) + ki * k_w + kj;
                    for (std::size_t out_i = 0; out_i < out_h; out_i++) {
                        for (std::size_t out_j = 0; out_j < out_w; out_j++) {
                            const std::size_t col = n * (out_h * out_w) + out_i * out_w + out_j;
                            const std::ptrdiff_t in_i = (
                                static_cast<std::ptrdiff_t>(out_i * stride_h) - pad_h +
                                dilation_h * static_cast<std::ptrdiff_t>(ki)
                            );
                            const std::ptrdiff_t in_j = (
                                static_cast<std::ptrdiff_t>(out_j * stride_w) - pad_w +
                                dilation_w * static_cast<std::ptrdiff_t>(kj)
                            );
                            T val = static_cast<T>(0);
                            if (
                                in_i >= 0 && 
                                in_i < static_cast<std::ptrdiff_t>(H) &&
                                in_j >= 0 &&
                                in_j < static_cast<std::ptrdiff_t>(W)
                            ) {
                                const std::size_t idx = (
                                    n * C * H * W + c * H * W +
                                    static_cast<std::size_t>(in_i) * W + static_cast<std::size_t>(in_j)
                                );
                                val = input.values()[idx];
                            }
                            input_matrix.values()[row * cols + col] = val;
                        }
                    }
                }
            }
        }
    }
    std::array<std::size_t, 2> kernel_matrix_shape = {
        kernel.shape()[0],
        kernel.shape()[1] * kernel.shape()[2] * kernel.shape()[3]
    };
    Tensor<T, 2> kernel_matrix(kernel_matrix_shape);
    // Kernel matrix is a flattened version of the kernel
    const std::vector<T>& kernel_data = kernel.values();
    std::copy(kernel_data.begin(), kernel_data.end(), kernel_matrix.values().begin());
    // Return input matrix and kernel matrix
    return {input_matrix, kernel_matrix};
}

template <FloatLike T>
std::array<Tensor<T, 2>, 2> im2col2d_taskflow(
    const Tensor<T, 4>& input,
    const Tensor<T, 4>& kernel,
    const std::array<std::size_t, 2>& stride,
    const std::array<std::size_t, 2>& padding,
    const std::array<std::size_t, 2>& dilation
) {
    const std::array<std::size_t, 4> output_shape = calculate_output_shape(
        input.shape(),
        kernel.shape(),
        stride,
        padding,
        dilation
    );
    const std::array<std::size_t, 2> input_matrix_shape = {
        input.shape()[1] * kernel.shape()[2] * kernel.shape()[3],
        input.shape()[0] * output_shape[2] * output_shape[3]
    };
    Tensor<T, 2> input_matrix(input_matrix_shape);
    // Standard im2col: rows = C_in * K_h * K_w, cols = N * out_h * out_w
    const std::size_t N = input.shape()[0];
    const std::size_t C = input.shape()[1];
    const std::size_t H = input.shape()[2];
    const std::size_t W = input.shape()[3];
    const std::size_t out_h = output_shape[2];
    const std::size_t out_w = output_shape[3];
    const std::size_t k_h = kernel.shape()[2];
    const std::size_t k_w = kernel.shape()[3];
    const std::size_t cols = N * out_h * out_w;
    const std::size_t rows = C * k_h * k_w;
    const std::ptrdiff_t stride_h = static_cast<std::ptrdiff_t>(stride[0]);
    const std::ptrdiff_t stride_w = static_cast<std::ptrdiff_t>(stride[1]);
    const std::ptrdiff_t pad_h = static_cast<std::ptrdiff_t>(padding[0]);
    const std::ptrdiff_t pad_w = static_cast<std::ptrdiff_t>(padding[1]);
    const std::ptrdiff_t dilation_h = static_cast<std::ptrdiff_t>(dilation[0]);
    const std::ptrdiff_t dilation_w = static_cast<std::ptrdiff_t>(dilation[1]);
    // Fill input_matrix (row-major: row * cols + col)
    tf::Taskflow taskflow;
    tf::IndexRange<std::size_t> indices(0, N, 1);
    taskflow.for_each_by_index(
        indices,
        [&](const tf::IndexRange<std::size_t>& range) {
            for (std::size_t n = range.begin(); n < range.end(); n++) {
                for (std::size_t c = 0; c < C; c++) {
                    for (std::size_t ki = 0; ki < k_h; ki++) {
                        for (std::size_t kj = 0; kj < k_w; kj++) {
                            const std::size_t row = c * (k_h * k_w) + ki * k_w + kj;
                            for (std::size_t out_i = 0; out_i < out_h; out_i++) {
                                for (std::size_t out_j = 0; out_j < out_w; out_j++) {
                                    const std::size_t col = n * (out_h * out_w) + out_i * out_w + out_j;
                                    const std::ptrdiff_t in_i = (
                                        static_cast<std::ptrdiff_t>(out_i * stride_h) - pad_h +
                                        dilation_h * static_cast<std::ptrdiff_t>(ki)
                                    );
                                    const std::ptrdiff_t in_j = (
                                        static_cast<std::ptrdiff_t>(out_j * stride_w) - pad_w +
                                        dilation_w * static_cast<std::ptrdiff_t>(kj)
                                    );
                                    T val = static_cast<T>(0);
                                    if (
                                        in_i >= 0 && 
                                        in_i < static_cast<std::ptrdiff_t>(H) &&
                                        in_j >= 0 &&
                                        in_j < static_cast<std::ptrdiff_t>(W)
                                    ) {
                                        const std::size_t idx = (
                                            n * C * H * W + c * H * W +
                                            static_cast<std::size_t>(in_i) * W + static_cast<std::size_t>(in_j)
                                        );
                                        val = input.values()[idx];
                                    }
                                    input_matrix.values()[row * cols + col] = val;
                                }
                            }
                        }
                    }
                }
            }
        }
    );
    executor.run(taskflow).wait();
    std::array<std::size_t, 2> kernel_matrix_shape = {
        kernel.shape()[0],
        kernel.shape()[1] * kernel.shape()[2] * kernel.shape()[3]
    };
    Tensor<T, 2> kernel_matrix(kernel_matrix_shape);
    // Kernel matrix is a flattened version of the kernel
    const std::vector<T>& kernel_data = kernel.values();
    std::copy(kernel_data.begin(), kernel_data.end(), kernel_matrix.values().begin());
    // Return input matrix and kernel matrix
    return {input_matrix, kernel_matrix};
}

// template <FloatLike T>
// std::array<Tensor<T, 2>, 2> im2col2d_taskflow(
//     const Tensor<T, 4>& input,
//     const Tensor<T, 4>& kernel,
//     const std::array<std::size_t, 2>& stride,
//     const std::array<std::size_t, 2>& padding,
//     const std::array<std::size_t, 2>& dilation
// ) {
//     const std::array<std::size_t, 4> output_shape = calculate_output_shape(
//         input.shape(),
//         kernel.shape(),
//         stride,
//         padding,
//         dilation
//     );
//     const std::array<std::size_t, 2> input_matrix_shape = {
//         input.shape()[1] * kernel.shape()[2] * kernel.shape()[3],
//         input.shape()[0] * output_shape[2] * output_shape[3]
//     };
//     Tensor<T, 2> input_matrix(input_matrix_shape);
//     // Standard im2col: rows = C_in * K_h * K_w, cols = N * out_h * out_w
//     const std::size_t N = input.shape()[0];
//     const std::size_t C = input.shape()[1];
//     const std::size_t H = input.shape()[2];
//     const std::size_t W = input.shape()[3];
//     const std::size_t out_h = output_shape[2];
//     const std::size_t out_w = output_shape[3];
//     const std::size_t k_h = kernel.shape()[2];
//     const std::size_t k_w = kernel.shape()[3];
//     const std::size_t cols = N * out_h * out_w;
//     const std::size_t rows = C * k_h * k_w;
//     const std::ptrdiff_t stride_h = static_cast<std::ptrdiff_t>(stride[0]);
//     const std::ptrdiff_t stride_w = static_cast<std::ptrdiff_t>(stride[1]);
//     const std::ptrdiff_t pad_h = static_cast<std::ptrdiff_t>(padding[0]);
//     const std::ptrdiff_t pad_w = static_cast<std::ptrdiff_t>(padding[1]);
//     const std::ptrdiff_t dilation_h = static_cast<std::ptrdiff_t>(dilation[0]);
//     const std::ptrdiff_t dilation_w = static_cast<std::ptrdiff_t>(dilation[1]);
//     // Fill input_matrix (row-major: row * cols + col)
//     tf::Taskflow taskflow;
//     tf::IndexRange<std::size_t, 2> indices(
//         tf::IndexRange<std::size_t>(0, k_h, 1),
//         tf::IndexRange<std::size_t>(0, k_w, 1)
//     );
//     for (std::size_t n = 0; n < N; n ++) {
//         for (std::size_t c = 0; c < C; c++) {
//             taskflow.for_each_by_index(
//                 indices,
//                 [=, &input_matrix, &input](const tf::IndexRange<std::size_t, 2>& range) -> void {
//                     for (std::size_t ki = range.dim(0).begin(); ki < range.dim(0).end(); ki++) {
//                         for (std::size_t kj = range.dim(1).begin(); kj < range.dim(1).end(); kj++) {
//                             const std::size_t row = c * (k_h * k_w) + ki * k_w + kj;
//                             for (std::size_t out_i = 0; out_i < out_h; out_i++) {
//                                 for (std::size_t out_j = 0; out_j < out_w; out_j++) {
//                                     const std::size_t col = n * (out_h * out_w) + out_i * out_w + out_j;
//                                     const std::ptrdiff_t in_i = (
//                                         static_cast<std::ptrdiff_t>(out_i * stride_h) - pad_h +
//                                         dilation_h * static_cast<std::ptrdiff_t>(ki)
//                                     );
//                                     const std::ptrdiff_t in_j = (
//                                         static_cast<std::ptrdiff_t>(out_j * stride_w) - pad_w +
//                                         dilation_w * static_cast<std::ptrdiff_t>(kj)
//                                     );
//                                     T val = static_cast<T>(0);
//                                     if (
//                                         in_i >= 0 && 
//                                         in_i < static_cast<std::ptrdiff_t>(H) &&
//                                         in_j >= 0 &&
//                                         in_j < static_cast<std::ptrdiff_t>(W)
//                                     ) {
//                                         const std::size_t idx = (
//                                             n * C * H * W + c * H * W +
//                                             static_cast<std::size_t>(in_i) * W + static_cast<std::size_t>(in_j)
//                                         );
//                                         val = input.values()[idx];
//                                     }
//                                     input_matrix.values()[row * cols + col] = val;
//                                 }
//                             }
//                         }
//                     }
//                 }
//             );
//         }
//     }
//     executor.run(taskflow).wait();
//     std::array<std::size_t, 2> kernel_matrix_shape = {
//         kernel.shape()[0],
//         kernel.shape()[1] * kernel.shape()[2] * kernel.shape()[3]
//     };
//     Tensor<T, 2> kernel_matrix(kernel_matrix_shape);
//     // Kernel matrix is a flattened version of the kernel
//     const std::vector<T>& kernel_data = kernel.values();
//     std::copy(kernel_data.begin(), kernel_data.end(), kernel_matrix.values().begin());
//     // Return input matrix and kernel matrix
//     return {input_matrix, kernel_matrix};
// }

template <FloatLike T>
std::array<Tensor<T, 2>, 2> im2col2d_openmp(
    const Tensor<T, 4>& input,
    const Tensor<T, 4>& kernel,
    const std::array<std::size_t, 2>& stride,
    const std::array<std::size_t, 2>& padding,
    const std::array<std::size_t, 2>& dilation
) {
    const std::array<std::size_t, 4> output_shape = calculate_output_shape(
        input.shape(),
        kernel.shape(),
        stride,
        padding,
        dilation
    );
    const std::array<std::size_t, 2> input_matrix_shape = {
        input.shape()[1] * kernel.shape()[2] * kernel.shape()[3],
        input.shape()[0] * output_shape[2] * output_shape[3]
    };
    Tensor<T, 2> input_matrix(input_matrix_shape);
    // Standard im2col: rows = C_in * K_h * K_w, cols = N * out_h * out_w
    const std::size_t N = input.shape()[0];
    const std::size_t C = input.shape()[1];
    const std::size_t H = input.shape()[2];
    const std::size_t W = input.shape()[3];
    const std::size_t out_h = output_shape[2];
    const std::size_t out_w = output_shape[3];
    const std::size_t k_h = kernel.shape()[2];
    const std::size_t k_w = kernel.shape()[3];
    const std::size_t cols = N * out_h * out_w;
    const std::size_t rows = C * k_h * k_w;
    const std::ptrdiff_t stride_h = static_cast<std::ptrdiff_t>(stride[0]);
    const std::ptrdiff_t stride_w = static_cast<std::ptrdiff_t>(stride[1]);
    const std::ptrdiff_t pad_h = static_cast<std::ptrdiff_t>(padding[0]);
    const std::ptrdiff_t pad_w = static_cast<std::ptrdiff_t>(padding[1]);
    const std::ptrdiff_t dilation_h = static_cast<std::ptrdiff_t>(dilation[0]);
    const std::ptrdiff_t dilation_w = static_cast<std::ptrdiff_t>(dilation[1]);
    // Fill input_matrix (row-major: row * cols + col)
    #pragma omp parallel for
    for (std::size_t n = 0; n < N; n++) {
        for (std::size_t c = 0; c < C; c++) {
            for (std::size_t ki = 0; ki < k_h; ki++) {
                for (std::size_t kj = 0; kj < k_w; kj++) {
                    const std::size_t row = c * (k_h * k_w) + ki * k_w + kj;
                    for (std::size_t out_i = 0; out_i < out_h; out_i++) {
                        for (std::size_t out_j = 0; out_j < out_w; out_j++) {
                            const std::size_t col = n * (out_h * out_w) + out_i * out_w + out_j;
                            const std::ptrdiff_t in_i = (
                                static_cast<std::ptrdiff_t>(out_i * stride_h) - pad_h +
                                dilation_h * static_cast<std::ptrdiff_t>(ki)
                            );
                            const std::ptrdiff_t in_j = (
                                static_cast<std::ptrdiff_t>(out_j * stride_w) - pad_w +
                                dilation_w * static_cast<std::ptrdiff_t>(kj)
                            );
                            T val = static_cast<T>(0);
                            if (
                                in_i >= 0 && 
                                in_i < static_cast<std::ptrdiff_t>(H) &&
                                in_j >= 0 &&
                                in_j < static_cast<std::ptrdiff_t>(W)
                            ) {
                                const std::size_t idx = (
                                    n * C * H * W + c * H * W +
                                    static_cast<std::size_t>(in_i) * W + static_cast<std::size_t>(in_j)
                                );
                                val = input.values()[idx];
                            }
                            input_matrix.values()[row * cols + col] = val;
                        }
                    }
                }
            }
        }
    }
    std::array<std::size_t, 2> kernel_matrix_shape = {
        kernel.shape()[0],
        kernel.shape()[1] * kernel.shape()[2] * kernel.shape()[3]
    };
    Tensor<T, 2> kernel_matrix(kernel_matrix_shape);
    // Kernel matrix is a flattened version of the kernel
    const std::vector<T>& kernel_data = kernel.values();
    std::copy(kernel_data.begin(), kernel_data.end(), kernel_matrix.values().begin());
    // Return input matrix and kernel matrix
    return {input_matrix, kernel_matrix};
}

template <FloatLike T>
std::array<Tensor<T, 2>, 2> im2col2d(
    const Tensor<T, 4>& input,
    const Tensor<T, 4>& kernel,
    const std::array<std::size_t, 2>& stride,
    const std::array<std::size_t, 2>& padding,
    const std::array<std::size_t, 2>& dilation
) {
    auto im2col2d_function = im2col2d_standard<T>;
    switch (IM2COL_BACKEND) {
        case Im2ColBackend::STANDARD:
            break;
        case Im2ColBackend::TASKFLOW:
            im2col2d_function = im2col2d_taskflow<T>;
            break;
        case Im2ColBackend::OPENMP:
            im2col2d_function = im2col2d_openmp<T>;
            break;
        default:
            throw std::invalid_argument("[im2col.hpp][im2col2d] Invalid im2col backend");
    }
    return im2col2d_function(input, kernel, stride, padding, dilation);
}

template <FloatLike T>
Tensor<T, 4> col2im2d_standard(
    const Tensor<T, 2>& output_matrix,
    const std::array<std::size_t, 4>& output_shape
) {
    // Convert output_matrix (out_c x (N*out_h*out_w)) into output tensor shape [N, out_c, out_h, out_w]
    // This is the same as transposing N and out_c
    const std::size_t N = output_shape[0];
    const std::size_t out_c = output_shape[1];
    const std::size_t out_h = output_shape[2];
    const std::size_t out_w = output_shape[3];
    Tensor<T, 4> output(output_shape);
    const std::size_t cols = N * out_h * out_w;
    for (std::size_t n = 0; n < N; n++) {
        for (std::size_t oc = 0; oc < out_c; oc++) {
            for (std::size_t i = 0; i < out_h; i++) {
                for (std::size_t j = 0; j < out_w; j++) {
                    const std::size_t col = n * (out_h * out_w) + i * out_w + j;
                    const std::size_t out_idx = n * (out_c * out_h * out_w) + oc * (out_h * out_w) + i * out_w + j;
                    output.values()[out_idx] = output_matrix.values()[oc * cols + col];
                }
            }
        }
    }
    return output;
}

template <FloatLike T>
Tensor<T, 4> col2im2d_taskflow(
    const Tensor<T, 2>& output_matrix,
    const std::array<std::size_t, 4>& output_shape
) {
    // Convert output_matrix (out_c x (N*out_h*out_w)) into output tensor shape [N, out_c, out_h, out_w]
    // This is the same as transposing N and out_c
    const std::size_t N = output_shape[0];
    const std::size_t out_c = output_shape[1];
    const std::size_t out_h = output_shape[2];
    const std::size_t out_w = output_shape[3];
    Tensor<T, 4> output(output_shape);
    const std::size_t cols = N * out_h * out_w;
    tf::Taskflow taskflow;
    tf::IndexRange<std::size_t> indices(0, N, 1);
    taskflow.for_each_by_index(
        indices,
        [&](const tf::IndexRange<std::size_t>& range) {
            for (std::size_t n = range.begin(); n < range.end(); n++) {
                for (std::size_t oc = 0; oc < out_c; oc++) {
                    for (std::size_t i = 0; i < out_h; i++) {
                        for (std::size_t j = 0; j < out_w; j++) {
                            const std::size_t col = n * (out_h * out_w) + i * out_w + j;
                            const std::size_t out_idx = n * (out_c * out_h * out_w) + oc * (out_h * out_w) + i * out_w + j;
                            output.values()[out_idx] = output_matrix.values()[oc * cols + col];
                        }
                    }
                }
            }
        }
    );
    executor.run(taskflow).wait();
    return output;
}

template <FloatLike T>
Tensor<T, 4> col2im2d_openmp(
    const Tensor<T, 2>& output_matrix,
    const std::array<std::size_t, 4>& output_shape
) {
    // Convert output_matrix (out_c x (N*out_h*out_w)) into output tensor shape [N, out_c, out_h, out_w]
    // This is the same as transposing N and out_c
    const std::size_t N = output_shape[0];
    const std::size_t out_c = output_shape[1];
    const std::size_t out_h = output_shape[2];
    const std::size_t out_w = output_shape[3];
    Tensor<T, 4> output(output_shape);
    const std::size_t cols = N * out_h * out_w;
    #pragma omp parallel for
    for (std::size_t n = 0; n < N; n++) {
        for (std::size_t oc = 0; oc < out_c; oc++) {
            for (std::size_t i = 0; i < out_h; i++) {
                for (std::size_t j = 0; j < out_w; j++) {
                    const std::size_t col = n * (out_h * out_w) + i * out_w + j;
                    const std::size_t out_idx = n * (out_c * out_h * out_w) + oc * (out_h * out_w) + i * out_w + j;
                    output.values()[out_idx] = output_matrix.values()[oc * cols + col];
                }
            }
        }
    }
    return output;
}

template <FloatLike T>
Tensor<T, 4> col2im2d(
    const Tensor<T, 2>& output_matrix,
    const std::array<std::size_t, 4>& output_shape
) {
    auto col2im2d_function = col2im2d_standard<T>;
    switch (IM2COL_BACKEND) {
        case Im2ColBackend::STANDARD:
            break;
        case Im2ColBackend::TASKFLOW:
            col2im2d_function = col2im2d_taskflow<T>;
            break;
        case Im2ColBackend::OPENMP:
            col2im2d_function = col2im2d_openmp<T>;
            break;
        default:
            throw std::invalid_argument("[im2col.hpp][col2im2d] Invalid IM2COL_BACKEND");
    }
    return col2im2d_function(output_matrix, output_shape);
}

} // namespace sc

#endif // IM2COL_HPP