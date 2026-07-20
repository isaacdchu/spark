#ifndef CONV_HPP
#define CONV_HPP

#include <array>
#include <vector>
#include <stdexcept>
#include <chrono>
#include <atomic>
#include <utility>
#include <cstddef>

#include <taskflow.hpp>
#include <algorithm/pipeline.hpp>
#include <cblas.h>

#include "tensor.hpp"
#include "concepts.hpp"
#include "im2col.hpp"
#include "util.hpp"

namespace sc {

inline tf::Executor executor(8);

enum class Padding {
    VALID,
    NONE,
    SAME
};

template <FloatLike T>
void matmul(
    const Tensor<T, 2>& a,
    const Tensor<T, 2>& b,
    Tensor<T, 2>& out
) {
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
            out.values().data(), ldc
        );
    } else {
        cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            m, n, k,
            1.0,
            a.values().data(), lda,
            b.values().data(), ldb,
            0.0,
            out.values().data(), ldc
        );
    }
}

inline struct {
    std::atomic<long long> im2col_kernel_us{0};
    std::atomic<long long> im2col_kernel_cnt{0};
    std::atomic<long long> im2col_us{0};
    std::atomic<long long> im2col_cnt{0};
    std::atomic<long long> matmul_us{0};
    std::atomic<long long> matmul_cnt{0};
    std::atomic<long long> col2im_us{0};
    std::atomic<long long> col2im_cnt{0};
} conv2d_times;

struct padding_t {
    std::size_t PADH;
    std::size_t PADW;
    operator std::array<std::size_t, 2>() const {
        return {PADH, PADW};
    }
    padding_t(const std::array<std::size_t, 2>& pad) : PADH(pad[0]), PADW(pad[1]) {}
};

template <FloatLike T>
Tensor<T, 4> conv2d(
    const Tensor<T, 4> &input,
    const Tensor<T, 4> &kernel,
    const std::array<std::size_t, 2> &stride = {1, 1},
    std::variant<padding_t, Padding> padding = Padding::VALID,
    const std::array<std::size_t, 2> &dilation = {1, 1}
) {
    // Check that both input and kernel have matching channel dimensions
    if (input.shape()[3] != kernel.shape()[0]) {
        throw std::invalid_argument("[conv.hpp][conv2d] Input and kernel channel dimensions do not match");
    }
    // Reset per-call timing accumulators so totals reflect this conv2d invocation
    conv2d_times.im2col_us.store(0);
    conv2d_times.im2col_cnt.store(0);
    conv2d_times.matmul_us.store(0);
    conv2d_times.matmul_cnt.store(0);
    conv2d_times.col2im_us.store(0);
    conv2d_times.col2im_cnt.store(0);
    conv2d_times.im2col_kernel_us.store(0);
    conv2d_times.im2col_kernel_cnt.store(0);
    // Handle padding
    std::array<std::size_t, 2> pad = {0, 0};
    if (std::holds_alternative<Padding>(padding)) {
        switch (std::get<Padding>(padding)) {
            case Padding::SAME:
                pad = calculate_padding(input.shape(), kernel.shape(), stride, dilation);
                break;
            case Padding::VALID:
                break;
            default:
                throw std::invalid_argument("Unsupported padding type");
        }
    } else {
        pad = static_cast<std::array<std::size_t, 2>>(std::get<padding_t>(padding));
    }
    // Input and kernel must be 4D tensors in NHWC/CRSK format
    const std::size_t N = input.shape()[0];
    // Calculate the output shape based on the input shape, kernel size, stride, padding, and dilation
    const std::array<std::size_t, 4> output_shape = calculate_output_shape(
        input.shape(), kernel.shape(),
        stride, pad, dilation
    );
    Tensor<T, 4> output = Tensor<T, 4>(output_shape);
    struct PipeData {
        Tensor<T, 2> im2col_result;
        Tensor<T, 2> matmul_result;
    };
    constexpr std::size_t num_lines = 4;
    std::array<PipeData, num_lines> buffer;
    // Pre-allocate memory
    for (std::size_t i = 0; i < num_lines; ++i) {
        buffer[i].im2col_result = Tensor<T, 2>(
            output_shape[1] * output_shape[2], kernel.size() / kernel.shape()[3]
        );
        buffer[i].matmul_result = Tensor<T, 2>(
            output_shape[1] * output_shape[2], kernel.shape()[3]
        );
        buffer[i].im2col_result.values().resize(buffer[i].im2col_result.size());
        buffer[i].matmul_result.values().resize(buffer[i].matmul_result.size());
    }
    Tensor<T, 2> kernel_matrix(
        kernel.size() / kernel.shape()[3], kernel.shape()[3]
    );
    tf::Taskflow taskflow("conv2d");
    tf::Task kernel_task = taskflow.emplace(
        [&]() -> void {
            auto start = std::chrono::high_resolution_clock::now();
            im2col2d(kernel, kernel_matrix);
            auto end = std::chrono::high_resolution_clock::now();
            auto dt_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            conv2d_times.im2col_kernel_us.fetch_add(dt_us);
            conv2d_times.im2col_kernel_cnt.fetch_add(1);
        }
    ).name("kernel_task");
    tf::Pipeline pipeline(
        num_lines,
        tf::Pipe{
            tf::PipeType::SERIAL,
            [&](tf::Pipeflow& pf) -> void {
                // stop pipe after N samples
                if (pf.token() == N) {
                    pf.stop();
                    return;
                }
            }
        },
        tf::Pipe{
            tf::PipeType::PARALLEL,
            [&](tf::Pipeflow& pf) -> void {
                // im2col
                auto start = std::chrono::high_resolution_clock::now();
                im2col2d(
                    input, kernel, stride, pad, dilation, pf.token(), buffer[pf.line()].im2col_result
                );
                auto end = std::chrono::high_resolution_clock::now();
                auto dt_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
                conv2d_times.im2col_us.fetch_add(dt_us);
                conv2d_times.im2col_cnt.fetch_add(1);
            }
        },
        tf::Pipe{
            tf::PipeType::PARALLEL,
            [&](tf::Pipeflow& pf) -> void {
                // matmul
                auto start = std::chrono::high_resolution_clock::now();
                matmul<T>(
                    buffer[pf.line()].im2col_result, kernel_matrix, buffer[pf.line()].matmul_result
                );
                auto end = std::chrono::high_resolution_clock::now();
                auto dt_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
                conv2d_times.matmul_us.fetch_add(dt_us);
                conv2d_times.matmul_cnt.fetch_add(1);
            }
        },
        tf::Pipe{
            tf::PipeType::PARALLEL,
            [&](tf::Pipeflow& pf) -> void {
                // col2im
                auto start = std::chrono::high_resolution_clock::now();
                col2im2d(buffer[pf.line()].matmul_result, output_shape, pf.token(), output);
                auto end = std::chrono::high_resolution_clock::now();
                auto dt_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
                conv2d_times.col2im_us.fetch_add(dt_us);
                conv2d_times.col2im_cnt.fetch_add(1);
            }
        }
    );
    tf::Task pipeline_task = taskflow.composed_of(pipeline).name("pipeline_task");
    // taskflow.for_each_index(
    //     0, N, 1,
    //     [&](std::size_t n) -> void {
    //         const auto [input_matrix, kernel_matrix] = im2col2d(input, kernel, stride, pad, dilation, n);
    //         const Tensor<T, 2> output_matrix = matmul(kernel_matrix, input_matrix);
    //         col2im2d(output_matrix, output_shape, output, n);
    //     }
    // );
    kernel_task.precede(pipeline_task);
    executor.run(taskflow).wait();
    return output;
}

template <FloatLike T>
Tensor<T, 4> conv2d(
    const Tensor<T, 4> &input,
    const Tensor<T, 4> &kernel,
    Padding padding
) {
    return conv2d(input, kernel, {1, 1}, padding, {1, 1});
}

} // namespace sc

#endif // CONV_HPP