#ifndef CONV_HPP
#define CONV_HPP

#include <array>
#include <vector>
#include <stdexcept>
#include <chrono>
#include <atomic>
#include <utility>
#include <cstddef>
#include <print>

#include <taskflow.hpp>
#include <algorithm/pipeline.hpp>
#include <algorithm/for_each.hpp>
#include <cblas.h>
#include <xnnpack.h>
#include <pthreadpool.h>

#include "tensor.hpp"
#include "concepts.hpp"
#include "util.hpp"

template <FloatLike T>
void gemv(
    const CBLAS_ORDER order, const CBLAS_TRANSPOSE trans,
    const int M, const int N, const T alpha, const T* A, const int lda,
    const T* x, const int incx, const T beta,
    T* y, const int incy
) {
    if constexpr (std::is_same_v<T, float>) {
        cblas_sgemv(order, trans, M, N, alpha, A, lda, x, incx, beta, y, incy);
    } else {
        cblas_dgemv(order, trans, M, N, alpha, A, lda, x, incx, beta, y, incy);
    }
}

template <FloatLike T>
void gemm(
    const T* A, const T* B, T* C,
    int M, int N, int K,
    int lda, int ldb, int ldc,
    T alpha = static_cast<T>(1), T beta = static_cast<T>(0)
) {
    if constexpr (std::is_same_v<T, float>) {
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            M, N, K, alpha, A, lda, B, ldb, beta, C, ldc
        );
    } else if constexpr (std::is_same_v<T, double>) {
        cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            M, N, K, alpha, A, lda, B, ldb, beta, C, ldc
        );
    }
}

std::array<std::size_t, 2> handle_padding(
    const std::array<std::size_t, 4>& input_shape,
    const std::array<std::size_t, 4>& kernel_shape,
    const std::array<std::size_t, 2>& stride,
    std::variant<sc::padding_t, sc::Padding> padding,
    const std::array<std::size_t, 2>& dilation
) {
    std::array<std::size_t, 2> pad = {0, 0};
    if (std::holds_alternative<sc::Padding>(padding)) {
        switch (std::get<sc::Padding>(padding)) {
            case sc::Padding::SAME:
                pad = sc::calculate_padding(input_shape, kernel_shape, stride, dilation);
            case sc::Padding::NONE:
            case sc::Padding::VALID:
                break;
            default:
                throw std::invalid_argument("[conv.hpp][handle_padding] Unsupported padding type");
        }
    } else {
        pad = static_cast<std::array<std::size_t, 2>>(std::get<sc::padding_t>(padding));
    }
    return pad;
}

namespace sc {

inline tf::Executor executor(8);

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

template <FloatLike T>
Tensor<T, 4> crsk_to_krsc(const Tensor<T, 4>& tensor) {
    const auto [C, R, S, K] = tensor.shape();
    Tensor<T, 4> output(std::array<std::size_t, 4>{K, R, S, C});
    const T* input_ptr = tensor.values().data();
    T* output_ptr = output.values().data();
    for (std::size_t c = 0; c < C; c++) {
        for (std::size_t r = 0; r < R; r++) {
            for (std::size_t s = 0; s < S; s++) {
                for (std::size_t k = 0; k < K; k++) {
                    output_ptr[k * R * S * C + r * S * C + s * C + c] =
                        input_ptr[c * R * S * K + r * S * K + s * K + k];
                }
            }
        }
    }
    return output;
}

template <FloatLike T>
Tensor<T, 4> crsk_to_rsck(const Tensor<T, 4>& tensor) {
    const auto [C, R, S, K] = tensor.shape();
    Tensor<T, 4> output(std::array<std::size_t, 4>{R, S, C, K});
    const T* input_ptr = tensor.values().data();
    T* output_ptr = output.values().data();
    for (std::size_t c = 0; c < C; c++) {
        for (std::size_t r = 0; r < R; r++) {
            for (std::size_t s = 0; s < S; s++) {
                for (std::size_t k = 0; k < K; k++) {
                    output_ptr[r * S * C * K + s * C * K + c * K + k] =
                        input_ptr[c * R * S * K + r * S * K + s * K + k];
                }
            }
        }
    }
    return output;
}

template <FloatLike T>
Tensor<T, 4> conv2d_implicit_gemm(
    const Tensor<T, 4>& input,
    const Tensor<T, 4>& kernel,
    const std::array<std::size_t, 2>& stride = {1, 1},
    std::variant<padding_t, Padding> padding = Padding::VALID,
    const std::array<std::size_t, 2>& dilation = {1, 1}
) {
    // NHWC/CRSK
    // Check that both input and kernel have matching channel dimensions
    if (input.shape()[3] != kernel.shape()[0]) {
        throw std::invalid_argument("[conv.hpp][conv2d] Input and kernel channel dimensions do not match");
    }
    // Handle padding
    std::array<std::size_t, 2> pad = handle_padding(input.shape(), kernel.shape(), stride, padding, dilation);
    // Calculate the output shape based on the input shape, kernel size, stride, padding, and dilation
    const std::array<std::size_t, 4> output_shape = calculate_output_shape(
        input.shape(), kernel.shape(),
        stride, pad, dilation
    );
    Tensor<T, 4> output(output_shape);
    std::fill(output.values().begin(), output.values().end(), static_cast<T>(0));
    const auto [N, H, W, C] = input.shape();
    const auto [C_, R, S, K] = kernel.shape();
    const auto [N_, P, Q, K_] = output_shape;
    const auto [S_H, S_W] = stride;
    const auto [P_H, P_W] = pad;
    const auto [D_H, D_W] = dilation;
    T const* const input_ptr = input.values().data();
    T const* const kernel_ptr = kernel.values().data();
    T* output_ptr = output.values().data();
    const std::size_t M = P * Q;
    const std::size_t gemm_K = C * R * S;
    constexpr std::size_t TILE_M = 64;
    constexpr std::size_t TILE_N = 64;
    constexpr std::size_t TILE_K = 64;
    tf::Taskflow taskflow("conv2d");
    if (D_H != 1 || D_W != 1) {
        taskflow.for_each_index(
            static_cast<std::size_t>(0), static_cast<std::size_t>(N), static_cast<std::size_t>(1),
            [=](std::size_t n) -> void {
                T* out_n = output_ptr + n * M * K;
                const T* in_n = input_ptr + n * H * W * C;
                for (std::size_t m0 = 0; m0 < M; m0 += TILE_M) {
                    const std::size_t m_end = std::min(m0 + TILE_M, M);
                    for (std::size_t n0 = 0; n0 < K; n0 += TILE_N) {
                        const std::size_t n_end = std::min(n0 + TILE_N, K);
                        for (std::size_t k0 = 0; k0 < gemm_K; k0 += TILE_K) {
                            const std::size_t k_end = std::min(k0 + TILE_K, gemm_K);
                            for (std::size_t m = m0; m < m_end; m++) {
                                const std::size_t p = m / Q;
                                const std::size_t q = m % Q;
                                for (std::size_t kk = k0; kk < k_end; kk++) {
                                    const std::size_t c = kk / (R * S);
                                    const std::size_t rs = kk % (R * S);
                                    const std::size_t r = rs / S;
                                    const std::size_t s = rs % S;
                                    const std::size_t h_idx = p * S_H + r * D_H;
                                    if (h_idx < P_H) continue;
                                    const std::size_t h = h_idx - P_H;
                                    if (h >= H) continue;
                                    const std::size_t w_idx = q * S_W + s * D_W;
                                    if (w_idx < P_W) continue;
                                    const std::size_t w = w_idx - P_W;
                                    if (w >= W) continue;
                                    const T a_val = in_n[h * W * C + w * C + c];
                                    const T* k_row = kernel_ptr + c * R * S * K + r * S * K + s * K;
                                    T* o_row = out_n + m * K;
                                    for (std::size_t nn = n0; nn < n_end; nn++) {
                                        o_row[nn] += a_val * k_row[nn];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        );
    } else if (P_H == 0 && P_W == 0) {
        taskflow.for_each_index(
            static_cast<std::size_t>(0), static_cast<std::size_t>(N), static_cast<std::size_t>(1),
            [=](std::size_t n) -> void {
                T* out_n = output_ptr + n * M * K;
                const T* in_n = input_ptr + n * H * W * C;
                for (std::size_t p = 0; p < P; p++) {
                    T* out_row = out_n + p * Q * K;
                    for (std::size_t r = 0; r < R; r++) {
                        const std::size_t h = p * S_H + r;
                        for (std::size_t s = 0; s < S; s++) {
                            const T* in_ptr = in_n + h * W * C + s * C;
                            const T* k_ptr = kernel_ptr + r * S * K + s * K;
                            gemm<T>(
                                in_ptr, k_ptr, out_row,
                                static_cast<int>(Q), static_cast<int>(K), static_cast<int>(C),
                                static_cast<int>(S_W * C), static_cast<int>(R * S * K), static_cast<int>(K),
                                static_cast<T>(1), static_cast<T>(1)
                            );
                        }
                    }
                }
            }
        );
    } else {
        taskflow.for_each_index(
            static_cast<std::size_t>(0), static_cast<std::size_t>(N), static_cast<std::size_t>(1),
            [=](std::size_t n) -> void {
                T* out_n = output_ptr + n * M * K;
                const T* in_n = input_ptr + n * H * W * C;
                for (std::size_t p = 0; p < P; p++) {
                    for (std::size_t q = 0; q < Q; q++) {
                        T* out_pq = out_n + (p * Q + q) * K;
                        for (std::size_t r = 0; r < R; r++) {
                            const std::size_t h_idx = p * S_H + r;
                            if (h_idx < P_H || h_idx - P_H >= H) continue;
                            const std::size_t h = h_idx - P_H;
                            for (std::size_t s = 0; s < S; s++) {
                                const std::size_t w_idx = q * S_W + s;
                                if (w_idx < P_W || w_idx - P_W >= W) continue;
                                const std::size_t w = w_idx - P_W;
                                const T* in_ptr = in_n + h * W * C + w * C;
                                const T* k_ptr = kernel_ptr + r * S * K + s * K;
                                gemv<T>(
                                    CblasRowMajor, CblasTrans,
                                    static_cast<int>(C), static_cast<int>(K),
                                    static_cast<T>(1), k_ptr, static_cast<int>(R * S * K),
                                    in_ptr, 1, static_cast<T>(1),
                                    out_pq, 1
                                );
                            }
                        }
                    }
                }
            }
        );
    }
    executor.run(taskflow).wait();
    return output;
}

template <FloatLike T>
Tensor<T, 4> conv2d_krsc(
    const Tensor<T, 4>& input,
    const Tensor<T, 4>& kernel,
    const std::array<std::size_t, 2>& stride = {1, 1},
    std::variant<padding_t, Padding> padding = Padding::VALID,
    const std::array<std::size_t, 2>& dilation = {1, 1}
) {
    // NHWC/KRSC
    if (input.shape()[3] != kernel.shape()[3]) {
        throw std::invalid_argument("[conv.hpp][conv2d_krsc] Input and kernel channel dimensions do not match");
    }
    const std::array<std::size_t, 4> kernel_crsk = {
        kernel.shape()[3], kernel.shape()[1], kernel.shape()[2], kernel.shape()[0]
    };
    std::array<std::size_t, 2> pad = handle_padding(
        input.shape(), kernel_crsk,
        stride, padding, dilation);
    const std::array<std::size_t, 4> output_shape = calculate_output_shape(
        input.shape(), kernel_crsk,
        stride, pad, dilation
    );
    Tensor<T, 4> output(output_shape);
    std::fill(output.values().begin(), output.values().end(), static_cast<T>(0));
    const auto [N, H, W, C] = input.shape();
    const auto [K, R, S, C_] = kernel.shape();
    const auto [N_, P, Q, K_] = output_shape;
    const auto [S_H, S_W] = stride;
    const auto [P_H, P_W] = pad;
    const auto [D_H, D_W] = dilation;
    T const* const in_base = input.values().data();
    T const* const k_base = kernel.values().data();
    T* out_ptr = output.values().data();
    // Implicit GEMM: C[m,n] = sum_k A[m,k] * B[k,n]
    // m = n_batch * (P * Q) + p*Q + q      (row of the virtual im2col matrix)
    // n = k_out                            (output channel index)
    // k = r * (S * C) + s * C + c          (reduction index over kernel spatial/channel positions)
    // A[m,k] = input[n_batch, h, w, c]     (zero for out-of-bounds/padded positions)
    // B[k,n] = kernel[k_out, r, s, c]      (KRSC: offset = k_out*R*S*C + r*S*C + s*C + c)
    const std::size_t gemm_K = R * S * C;
    tf::Taskflow taskflow("conv2d_krsc");
    auto handle_sample = [=](std::size_t n_batch) -> void {
        T* out_n = out_ptr + n_batch * P * Q * K;
        const T* in_n = in_base + n_batch * H * W * C;
        for (std::size_t p = 0; p < P; p++) {
            for (std::size_t q = 0; q < Q; q++) {
                T* out_pq = out_n + (p * Q + q) * K;
                for (std::size_t n = 0; n < K; n++) {
                    T acc = static_cast<T>(0);
                    for (std::size_t k = 0; k < gemm_K; k++) {
                        const std::size_t r   = k / (S * C);
                        const std::size_t rsc = k % (S * C);
                        const std::size_t s   = rsc / C;
                        const std::size_t c   = rsc % C;
                        const std::size_t h_padded = p * S_H + r * D_H;
                        if (h_padded < P_H) continue;
                        const std::size_t h = h_padded - P_H;
                        if (h >= H) continue;
                        const std::size_t w_padded = q * S_W + s * D_W;
                        if (w_padded < P_W) continue;
                        const std::size_t w = w_padded - P_W;
                        if (w >= W) continue;
                        acc += in_n[h * W * C + w * C + c] * k_base[n * R * S * C + r * S * C + s * C + c];
                    }
                    out_pq[n] = acc;
                }
            }
        }
    };
    taskflow.for_each_index(std::size_t(0), N, std::size_t(1), handle_sample);
    executor.run(taskflow).wait();
    return output;
}

template <FloatLike T>
Tensor<T, 4> conv2d_rsck(
    const Tensor<T, 4>& input,
    const Tensor<T, 4>& kernel,
    const std::array<std::size_t, 2>& stride = {1, 1},
    std::variant<padding_t, Padding> padding = Padding::VALID,
    const std::array<std::size_t, 2>& dilation = {1, 1}
) {
    // NHWC/RSCK format
    // Check that both input and kernel have matching channel dimensions
    if (input.shape()[3] != kernel.shape()[2]) {
        throw std::invalid_argument("[conv.hpp][conv2d_rsck] Input and kernel channel dimensions do not match");
    }
    const std::array<std::size_t, 4> kernel_crsk = {
        kernel.shape()[2], kernel.shape()[0], kernel.shape()[1], kernel.shape()[3]
    };
    // Handle padding
    std::array<std::size_t, 2> pad = handle_padding(input.shape(), kernel_crsk, stride, padding, dilation);
    // Calculate the output shape based on the input shape, kernel size, stride, padding, and dilation
    const std::array<std::size_t, 4> output_shape = calculate_output_shape(
        input.shape(), kernel_crsk,
        stride, pad, dilation
    );
    Tensor<T, 4> output(output_shape);
    std::fill(output.values().begin(), output.values().end(), static_cast<T>(0));
    const auto [N, H, W, C] = input.shape();
    const auto [R, S, C_, K] = kernel.shape();
    const auto [N_, P, Q, K_] = output_shape;
    const auto [S_H, S_W] = stride;
    const auto [P_H, P_W] = pad;
    const auto [D_H, D_W] = dilation;
    T const* const in_base = input.values().data();
    T const* const k_base = kernel.values().data();
    T* output_ptr = output.values().data();
    tf::Taskflow taskflow("conv2d_rsck");
    if (P_H == 0 && P_W == 0 && D_H == 1 && D_W == 1) {
        taskflow.for_each_index(
            static_cast<std::size_t>(0), static_cast<std::size_t>(N), static_cast<std::size_t>(1),
            [=](std::size_t n) -> void {
                T* out_n = output_ptr + n * P * Q * K;
                for (std::size_t p = 0; p < P; p++) {
                    T* out_row = out_n + p * Q * K;
                    for (std::size_t r = 0; r < R; r++) {
                        const std::size_t h = p * S_H + r;
                        for (std::size_t s = 0; s < S; s++) {
                            const T* in_ptr = in_base + n * H * W * C + h * W * C + s * C;
                            const T* k_ptr = k_base + r * S * C * K + s * C * K;
                            gemm<T>(
                                in_ptr, k_ptr, out_row,
                                static_cast<int>(Q), static_cast<int>(K), static_cast<int>(C),
                                static_cast<int>(S_W * C), static_cast<int>(K), static_cast<int>(K),
                                static_cast<T>(1), static_cast<T>(1)
                            );
                        }
                    }
                }
            }
        );
    } else if (D_W == 1) {
        taskflow.for_each_index(
            static_cast<std::size_t>(0), static_cast<std::size_t>(N), static_cast<std::size_t>(1),
            [=](std::size_t n) -> void {
                T* out_n = output_ptr + n * P * Q * K;
                for (std::size_t p = 0; p < P; p++) {
                    for (std::size_t q = 0; q < Q; q++) {
                        T* out_pq = out_n + (p * Q + q) * K;
                        for (std::size_t r = 0; r < R; r++) {
                            const std::size_t h_pad = p * S_H + r * D_H;
                            if (h_pad < P_H) continue;
                            const std::size_t h = h_pad - P_H;
                            if (h >= H) break;
                            const std::size_t w0 = q * S_W;
                            const std::size_t s_start = w0 < P_W ? P_W - w0 : 0;
                            const std::size_t s_end = W + P_W > w0 ? std::min(S, W + P_W - w0) : 0;
                            if (s_start < s_end) {
                                const T* in_ptr = in_base + n * H * W * C + h * W * C + (w0 + s_start - P_W) * C;
                                const T* k_ptr = k_base + r * S * C * K + s_start * C * K;
                                gemv<T>(
                                    CblasRowMajor, CblasTrans,
                                    static_cast<int>((s_end - s_start) * C), static_cast<int>(K),
                                    static_cast<T>(1), k_ptr, static_cast<int>(K),
                                    in_ptr, 1, static_cast<T>(1),
                                    out_pq, 1
                                );
                            }
                        }
                    }
                }
            }
        );
    } else {
        taskflow.for_each_index(
            static_cast<std::size_t>(0), static_cast<std::size_t>(N), static_cast<std::size_t>(1),
            [=](std::size_t n) -> void {
                T* out_n = output_ptr + n * P * Q * K;
                for (std::size_t p = 0; p < P; p++) {
                    for (std::size_t q = 0; q < Q; q++) {
                        T* out_pq = out_n + (p * Q + q) * K;
                        for (std::size_t r = 0; r < R; r++) {
                            const std::size_t h_pad = p * S_H + r * D_H;
                            if (h_pad < P_H) continue;
                            const std::size_t h = h_pad - P_H;
                            if (h >= H) break;
                            for (std::size_t s = 0; s < S; s++) {
                                const std::size_t w_pad = q * S_W + s * D_W;
                                if (w_pad < P_W) continue;
                                const std::size_t w = w_pad - P_W;
                                if (w >= W) break;
                                const T* in_ptr = in_base + n * H * W * C + h * W * C + w * C;
                                const T* k_ptr = k_base + r * S * C * K + s * C * K;
                                gemv<T>(
                                    CblasRowMajor, CblasTrans,
                                    static_cast<int>(C), static_cast<int>(K),
                                    static_cast<T>(1), k_ptr, static_cast<int>(K),
                                    in_ptr, 1, static_cast<T>(1),
                                    out_pq, 1
                                );
                            }
                        }
                    }
                }
            }
        );
    }
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

template <FloatLike T>
Tensor<T, 4> conv2d_explicit_gemm(
    const Tensor<T, 4> &input,
    const Tensor<T, 4> &kernel,
    const std::array<std::size_t, 2> &stride = {1, 1},
    std::variant<padding_t, Padding> padding = Padding::VALID,
    const std::array<std::size_t, 2> &dilation = {1, 1}
) {
    // NHWC/CRSK format
    if (input.shape()[3] != kernel.shape()[0]) {
        throw std::invalid_argument("[conv.hpp][conv2d_explicit_gemm] Input and kernel channel dimensions do not match");
    }
    std::array<std::size_t, 2> pad = handle_padding(input.shape(), kernel.shape(), stride, padding, dilation);
    const std::array<std::size_t, 4> output_shape = calculate_output_shape(
        input.shape(), kernel.shape(),
        stride, pad, dilation
    );
    Tensor<T, 4> output(output_shape);
    auto kernel_matrix = Tensor<T, 2>(
        kernel.size() / kernel.shape()[3], kernel.shape()[3]
    );
    im2col2d(kernel, kernel_matrix);
    tf::Taskflow taskflow("conv2d_explicit_gemm");
    taskflow.for_each_index(
        static_cast<std::size_t>(0), input.shape()[0], static_cast<std::size_t>(1),
        [&](std::size_t n) -> void {
            auto im2col_result = Tensor<T, 2>(
                output_shape[1] * output_shape[2], kernel.size() / kernel.shape()[3]
            );
            im2col2d(input, kernel, stride, pad, dilation, n, im2col_result);
            auto matmul_result = Tensor<T, 2>(
                output_shape[1] * output_shape[2], kernel.shape()[3]
            );
            matmul<T>(im2col_result, kernel_matrix, matmul_result);
            col2im2d(matmul_result, output_shape, n, output);
        }
    );
    executor.run(taskflow).wait();
    return output;
}

template <FloatLike T>
Tensor<T, 4> conv2d_pipeline(
    const Tensor<T, 4> &input,
    const Tensor<T, 4> &kernel,
    const std::array<std::size_t, 2> &stride = {1, 1},
    std::variant<padding_t, Padding> padding = Padding::VALID,
    const std::array<std::size_t, 2> &dilation = {1, 1}
) {
    // NHWC/CRSK format
    if (input.shape()[3] != kernel.shape()[0]) {
        throw std::invalid_argument("[conv.hpp][conv2d_pipeline] Input and kernel channel dimensions do not match");
    }
    // Handle padding
    std::array<std::size_t, 2> pad = handle_padding(input.shape(), kernel.shape(), stride, padding, dilation);
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
            im2col2d(kernel, kernel_matrix);
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
                im2col2d(
                    input, kernel, stride, pad, dilation, pf.token(), buffer[pf.line()].im2col_result
                );
            }
        },
        tf::Pipe{
            tf::PipeType::PARALLEL,
            [&](tf::Pipeflow& pf) -> void {
                // matmul
                matmul<T>(
                    buffer[pf.line()].im2col_result, kernel_matrix, buffer[pf.line()].matmul_result
                );
            }
        },
        tf::Pipe{
            tf::PipeType::PARALLEL,
            [&](tf::Pipeflow& pf) -> void {
                // col2im
                col2im2d(buffer[pf.line()].matmul_result, output_shape, pf.token(), output);
            }
        }
    );
    tf::Task pipeline_task = taskflow.composed_of(pipeline).name("pipeline_task");
    kernel_task.precede(pipeline_task);
    executor.run(taskflow).wait();
    return output;
}

template <FloatLike T>
Tensor<T, 4> conv2d_xnnpack(
    const Tensor<T, 4>& input,
    const Tensor<T, 4>& kernel,
    const std::array<std::size_t, 2>& stride = {1, 1},
    std::variant<padding_t, Padding> padding = Padding::VALID,
    const std::array<std::size_t, 2>& dilation = {1, 1}
) {
    // NHWC/KRSC format
    static_assert(std::is_same_v<T, float>, "XNNPACK conv2d only supports float");

    if (input.shape()[3] != kernel.shape()[3]) {
        throw std::invalid_argument("[conv.hpp][conv2d_xnnpack] Input and kernel channel dimensions do not match");
    }

    const std::array<std::size_t, 4> kernel_crsk = {
        kernel.shape()[3], kernel.shape()[1], kernel.shape()[2], kernel.shape()[0]
    };

    std::array<std::size_t, 2> pad = handle_padding(
        input.shape(), kernel_crsk,
        stride, padding, dilation
    );

    const std::array<std::size_t, 4> output_shape = calculate_output_shape(
        input.shape(), kernel_crsk,
        stride, pad, dilation
    );

    Tensor<T, 4> output(output_shape);
    const auto [N, H, W, C] = input.shape();
    const auto [K, R, S, C_k] = kernel.shape();  // KRSC format

    static std::once_flag xnn_init_flag;
    std::call_once(xnn_init_flag, []() -> void {
        if (xnn_initialize(nullptr) != xnn_status_success) {
            throw std::runtime_error("[conv.hpp][conv2d_xnnpack] Failed to initialize XNNPACK");
        }
    });

    static pthreadpool_t threadpool = pthreadpool_create(8);

    xnn_operator_t conv_op = nullptr;
    xnn_status status = xnn_create_convolution2d_nhwc_f32(
        pad[0], pad[1], pad[0], pad[1],
        static_cast<uint32_t>(R), static_cast<uint32_t>(S),
        static_cast<uint32_t>(stride[0]), static_cast<uint32_t>(stride[1]),
        static_cast<uint32_t>(dilation[0]), static_cast<uint32_t>(dilation[1]),
        1,
        C, K,
        C, K,
        kernel.values().data(),
        nullptr,
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        0, nullptr,
        &conv_op
    );
    if (status != xnn_status_success) {
        throw std::runtime_error("[conv.hpp][conv2d_xnnpack] Failed to create XNNPACK conv2d operator");
    }

    std::size_t workspace_size = 0;
    std::size_t output_height = 0, output_width = 0;
    status = xnn_reshape_convolution2d_nhwc_f32(
        conv_op, N, H, W,
        &workspace_size,
        &output_height, &output_width,
        threadpool
    );
    if (status != xnn_status_success) {
        xnn_delete_operator(conv_op);
        throw std::runtime_error("[conv.hpp][conv2d_xnnpack] Failed to reshape XNNPACK conv2d operator");
    }

    void* workspace = nullptr;
    if (workspace_size > 0) {
        workspace = std::malloc(workspace_size);
    }

    status = xnn_setup_convolution2d_nhwc_f32(
        conv_op, workspace,
        input.values().data(), output.values().data()
    );
    if (status != xnn_status_success) {
        std::free(workspace);
        xnn_delete_operator(conv_op);
        throw std::runtime_error("[conv.hpp][conv2d_xnnpack] Failed to setup XNNPACK conv2d operator");
    }

    status = xnn_run_operator(conv_op, threadpool);
    if (status != xnn_status_success) {
        std::free(workspace);
        xnn_delete_operator(conv_op);
        throw std::runtime_error("[conv.hpp][conv2d_xnnpack] Failed to run XNNPACK conv2d operator");
    }

    std::free(workspace);
    xnn_delete_operator(conv_op);
    return output;
}

} // namespace sc

#endif // CONV_HPP