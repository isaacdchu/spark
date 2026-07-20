#ifndef CONV_HPP
#define CONV_HPP

#include <array>
#include <vector>
#include <stdexcept>
#include <chrono>
#include <atomic>
#include <utility>
#include <new>
#include <cstddef>
#include <cstring>
#include <optional>
#include <print>

#include <xsimd.hpp>
#include <taskflow.hpp>
#include <algorithm/pipeline.hpp>
#include <algorithm/for_each.hpp>
#include <cblas.h>
#include <xnnpack.h>

#include "tensor.hpp"
#include "im2col.hpp"
#include "concepts.hpp"
#include "util.hpp"

template <FloatLike T>
void matmul(
    const sc::Tensor<T, 2>& a,
    const sc::Tensor<T, 2>& b,
    sc::Tensor<T, 2>& out
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

namespace sc {

inline tf::Executor executor(8);

template <FloatLike T>
Tensor<T, 4> crsk_to_krsc(const Tensor<T, 4>& kernel) {
    const auto [C, R, S, K] = kernel.shape();
    Tensor<T, 4> converted(K, R, S, C);
    for (std::size_t c = 0; c < C; c++) {
        for (std::size_t r = 0; r < R; r++) {
            for (std::size_t s = 0; s < S; s++) {
                for (std::size_t k = 0; k < K; k++) {
                    converted.values()[k * R * S * C + r * S * C + s * C + c] =
                        kernel.values()[c * R * S * K + r * S * K + s * K + k];
                }
            }
        }
    }
    return converted;
}

template <FloatLike T>
Tensor<T, 4> crsk_to_rsck(const Tensor<T, 4>& kernel) {
    const auto [C, R, S, K] = kernel.shape();
    Tensor<T, 4> converted(R, S, C, K);
    for (std::size_t c = 0; c < C; c++) {
        for (std::size_t r = 0; r < R; r++) {
            for (std::size_t s = 0; s < S; s++) {
                for (std::size_t k = 0; k < K; k++) {
                    converted.values()[r * S * C * K + s * C * K + c * K + k] =
                        kernel.values()[c * R * S * K + r * S * K + s * K + k];
                }
            }
        }
    }
    return converted;
}

inline std::size_t round_up(std::size_t n, std::size_t multiple) {
    return (n + multiple - 1) / multiple * multiple;
}

// Zero-pad the last (contiguous) axis of a tensor up to a multiple of
// `multiple`. XNNPACK-style weight padding: the SIMD kernels can then run
// full-width vector loads over that axis, with the zero lanes contributing
// nothing, instead of falling back to a scalar tail loop.
template <FloatLike T>
Tensor<T, 4> pad_last_axis(const Tensor<T, 4>& tensor, std::size_t multiple) {
    const auto [D0, D1, D2, D3] = tensor.shape();
    const std::size_t D3_pad = round_up(D3, multiple);
    Tensor<T, 4> padded = Tensor<T, 4>::zeros(std::array<std::size_t, 4>{D0, D1, D2, D3_pad});
    const T* const src = tensor.values().data();
    T* const dst = padded.values().data();
    for (std::size_t row = 0; row < D0 * D1 * D2; row++) {
        std::memcpy(dst + row * D3_pad, src + row * D3, D3 * sizeof(T));
    }
    return padded;
}

// Load `count` elements (count <= lane width) into the low lanes of a vector,
// zero-filling the remaining lanes.
template <FloatLike T>
xsimd::batch<T> load_partial(const T* src, std::size_t count) {
    alignas(xsimd::batch<T>::arch_type::alignment()) T buf[xsimd::batch<T>::size] = {};
    std::memcpy(buf, src, count * sizeof(T));
    return xsimd::load_aligned(buf);
}

// Store the low `count` lanes (count <= lane width) of a vector.
template <FloatLike T>
void store_partial(const xsimd::batch<T>& v, T* dst, std::size_t count) {
    alignas(xsimd::batch<T>::arch_type::alignment()) T buf[xsimd::batch<T>::size];
    v.store_aligned(buf);
    std::memcpy(dst, buf, count * sizeof(T));
}

template <FloatLike T>
Tensor<T, 4> conv2d_block_krsc(
    const Tensor<T, 4>& input,
    const Tensor<T, 4>& kernel,
    const std::array<std::size_t, 2>& stride = {1, 1},
    const Padding& padding = Padding(),
    const std::array<std::size_t, 2>& dilation = {1, 1}
) {
    // NHWC/KRSC format
    if (input.shape()[3] != kernel.shape()[3]) {
        throw std::invalid_argument("[conv.hpp][conv2d_block_krsc] Input and kernel channel dimensions do not match");
    }
    const std::array<std::size_t, 4> kernel_shape_crsk = {
        kernel.shape()[3], kernel.shape()[1], kernel.shape()[2], kernel.shape()[0]
    };
    const std::array<std::size_t, 2> pad = handle_padding(
        input.shape(), kernel_shape_crsk, stride, padding, dilation
    );
    const std::array<std::size_t, 4> output_shape = calculate_output_shape(
        input.shape(), kernel_shape_crsk,
        stride, pad, dilation
    );
    Tensor<T, 4> output(output_shape);
    const auto [N, H, W, C] = input.shape();
    const auto [K, R, S, C_] = kernel.shape();
    const auto [N_, P, Q, K_] = output_shape;
    const auto [S_H, S_W] = stride;
    const auto [P_H, P_W] = pad;
    const auto [D_H, D_W] = dilation;
    T const* const input_ptr = input.values().data();
    T* const output_ptr = output.values().data();
    // Vectorize over the channel axis C (contiguous in both NHWC input and KRSC
    // weights): multiply a vector of input channels against a vector of weights
    // and accumulate. C is a reduction axis, so each accumulator is horizontally
    // summed (reduce_add) into a single output value at the end.
    //
    // Register tile: Q_BLOCK output pixels x K_BLOCK output channels. An input
    // vector is reused across the K_BLOCK channels and a weight vector is reused
    // across the Q_BLOCK pixels, giving Q_BLOCK*K_BLOCK independent FMA chains to
    // hide FMA latency while the r/s/c reduction is streamed.
    constexpr std::size_t inc = xsimd::batch<T>::size;
    constexpr std::size_t Q_BLOCK = 8;
    constexpr std::size_t K_BLOCK = 4;
    // Zero-pad the weights' contiguous C axis to a full vector multiple
    // (XNNPACK-style) so the reduction runs entirely in full-width FMAs: the
    // padded weight lanes are zero, and the matching input tail lanes are
    // zero-filled via load_partial (the input itself cannot be over-read).
    const std::size_t C_pad = round_up(C, inc);
    std::optional<Tensor<T, 4>> kernel_padded;
    if (C_pad != C) {
        kernel_padded = pad_last_axis(kernel, inc);
    }
    T const* const kernel_ptr = (kernel_padded ? *kernel_padded : kernel).values().data();
    const std::size_t c_vec = C - (C % inc); // channels covered by full input vectors
    auto simd_c = [&](std::size_t n) -> void {
        T* const output_ptr_n = output_ptr + n * P * Q * K;
        T const* const input_ptr_n = input_ptr + n * H * W * C;
        for (std::size_t p = 0; p < P; p++) {
            for (std::size_t q0 = 0; q0 < Q; q0 += Q_BLOCK) {
                const std::size_t qn = std::min(Q_BLOCK, Q - q0);
                for (std::size_t k0 = 0; k0 < K; k0 += K_BLOCK) {
                    const std::size_t kn = std::min(K_BLOCK, K - k0);
                    xsimd::batch<T> acc[Q_BLOCK][K_BLOCK];
                    for (std::size_t qi = 0; qi < qn; qi++) {
                        for (std::size_t ki = 0; ki < kn; ki++) {
                            acc[qi][ki] = xsimd::batch<T>(static_cast<T>(0));
                        }
                    }
                    for (std::size_t r = 0; r < R; r++) {
                        const std::ptrdiff_t h = static_cast<std::ptrdiff_t>(p * S_H + r * D_H)
                            - static_cast<std::ptrdiff_t>(P_H);
                        if (h < 0 || h >= static_cast<std::ptrdiff_t>(H)) continue;
                        const std::size_t h_in = static_cast<std::size_t>(h);
                        for (std::size_t s = 0; s < S; s++) {
                            // input receptive-field base per tile pixel (independent of k)
                            const T* input_base[Q_BLOCK];
                            for (std::size_t qi = 0; qi < qn; qi++) {
                                const std::ptrdiff_t w = static_cast<std::ptrdiff_t>((q0 + qi) * S_W + s * D_W)
                                    - static_cast<std::ptrdiff_t>(P_W);
                                input_base[qi] = (w < 0 || w >= static_cast<std::ptrdiff_t>(W))
                                    ? nullptr
                                    : input_ptr_n + h_in * W * C + static_cast<std::size_t>(w) * C;
                            }
                            const T* const kernel_rs = kernel_ptr + k0 * R * S * C_pad + r * S * C_pad + s * C_pad;
                            for (std::size_t c = 0; c < c_vec; c += inc) {
                                xsimd::batch<T> wvec[K_BLOCK];
                                for (std::size_t ki = 0; ki < kn; ki++) {
                                    wvec[ki] = xsimd::load_unaligned(kernel_rs + ki * R * S * C_pad + c);
                                }
                                for (std::size_t qi = 0; qi < qn; qi++) {
                                    if (input_base[qi] == nullptr) continue;
                                    const xsimd::batch<T> in_v = xsimd::load_unaligned(input_base[qi] + c);
                                    for (std::size_t ki = 0; ki < kn; ki++) {
                                        acc[qi][ki] = xsimd::fma(in_v, wvec[ki], acc[qi][ki]);
                                    }
                                }
                            }
                            if (c_vec != C) {
                                // Tail channel group [c_vec, C_pad): weight lanes past C
                                // are zero, input lanes past C are zero-filled.
                                xsimd::batch<T> wvec[K_BLOCK];
                                for (std::size_t ki = 0; ki < kn; ki++) {
                                    wvec[ki] = xsimd::load_unaligned(kernel_rs + ki * R * S * C_pad + c_vec);
                                }
                                for (std::size_t qi = 0; qi < qn; qi++) {
                                    if (input_base[qi] == nullptr) continue;
                                    const xsimd::batch<T> in_v = load_partial(input_base[qi] + c_vec, C - c_vec);
                                    for (std::size_t ki = 0; ki < kn; ki++) {
                                        acc[qi][ki] = xsimd::fma(in_v, wvec[ki], acc[qi][ki]);
                                    }
                                }
                            }
                        }
                    }
                    for (std::size_t qi = 0; qi < qn; qi++) {
                        T* const out_base = output_ptr_n + p * Q * K + (q0 + qi) * K + k0;
                        for (std::size_t ki = 0; ki < kn; ki++) {
                            out_base[ki] = xsimd::reduce_add(acc[qi][ki]);
                        }
                    }
                }
            }
        }
    };
    tf::Taskflow taskflow("conv2d_block_krsc");
    taskflow.for_each_index(
        static_cast<std::size_t>(0), N, static_cast<std::size_t>(1),
        simd_c
    );
    executor.run(taskflow).wait();
    return output;
}

template <FloatLike T>
Tensor<T, 4> conv2d_block_rsck(
    const Tensor<T, 4>& input,
    const Tensor<T, 4>& kernel,
    const std::array<std::size_t, 2>& stride = {1, 1},
    const Padding& padding = Padding(),
    const std::array<std::size_t, 2>& dilation = {1, 1}
) {
    // NHWC/RSCK format
    if (input.shape()[3] != kernel.shape()[2]) {
        throw std::invalid_argument("[conv.hpp][conv2d_block_rsck] Input and kernel channel dimensions do not match");
    }
    const std::array<std::size_t, 4> kernel_shape_crsk = {
        kernel.shape()[2], kernel.shape()[0], kernel.shape()[1], kernel.shape()[3]
    };
    const std::array<std::size_t, 2> pad = handle_padding(
        input.shape(), kernel_shape_crsk, stride, padding, dilation
    );
    const std::array<std::size_t, 4> output_shape = calculate_output_shape(
        input.shape(), kernel_shape_crsk,
        stride, pad, dilation
    );
    Tensor<T, 4> output(output_shape);
    const auto [N, H, W, C] = input.shape();
    const auto [R, S, C_, K] = kernel.shape();
    const auto [N_, P, Q, K_] = output_shape;
    const auto [S_H, S_W] = stride;
    const auto [P_H, P_W] = pad;
    const auto [D_H, D_W] = dilation;
    T const* const input_ptr = input.values().data();
    T* const output_ptr = output.values().data();
    // Vectorize over the output-channel axis K (broadcast one input value, FMA it
    // against a contiguous vector of K weights). Unlike the C-reduction kernels
    // there is no horizontal reduce_add: each SIMD lane owns a distinct output
    // channel, so accumulators map 1:1 onto contiguous output stores.
    //
    // Register tile: Q_BLOCK output pixels x KV_BLOCK channel-vectors. A weight
    // vector is reused across the Q_BLOCK pixels and a broadcast input value is
    // reused across the KV_BLOCK vectors, giving Q_BLOCK*KV_BLOCK independent FMA
    // chains to hide FMA latency while the r/s/c reduction is streamed.
    constexpr std::size_t inc = xsimd::batch<T>::size;
    constexpr std::size_t Q_BLOCK = 6;
    constexpr std::size_t KV_BLOCK = 2;
    // Zero-pad the weights' contiguous K axis to a full vector multiple
    // (XNNPACK-style) so every output-channel group runs full-width FMAs; the
    // padded lanes compute throwaway channels that are simply not stored.
    const std::size_t K_pad = round_up(K, inc);
    std::optional<Tensor<T, 4>> kernel_padded;
    if (K_pad != K) {
        kernel_padded = pad_last_axis(kernel, inc);
    }
    T const* const kernel_ptr = (kernel_padded ? *kernel_padded : kernel).values().data();
    auto simd_k = [&](std::size_t n) -> void {
        T* const output_ptr_n = output_ptr + n * P * Q * K;
        T const* const input_ptr_n = input_ptr + n * H * W * C;
        for (std::size_t p = 0; p < P; p++) {
            for (std::size_t q0 = 0; q0 < Q; q0 += Q_BLOCK) {
                const std::size_t qn = std::min(Q_BLOCK, Q - q0);
                // All of [0, K_pad) is vector-width aligned thanks to the padding.
                for (std::size_t k0 = 0; k0 < K_pad; k0 += KV_BLOCK * inc) {
                    const std::size_t kv_n = std::min(KV_BLOCK, (K_pad - k0) / inc);
                    xsimd::batch<T> acc[Q_BLOCK][KV_BLOCK];
                    for (std::size_t qi = 0; qi < qn; qi++) {
                        for (std::size_t kv = 0; kv < kv_n; kv++) {
                            acc[qi][kv] = xsimd::batch<T>(static_cast<T>(0));
                        }
                    }
                    for (std::size_t r = 0; r < R; r++) {
                        const std::ptrdiff_t h = static_cast<std::ptrdiff_t>(p * S_H + r * D_H)
                            - static_cast<std::ptrdiff_t>(P_H);
                        if (h < 0 || h >= static_cast<std::ptrdiff_t>(H)) continue;
                        const std::size_t h_in = static_cast<std::size_t>(h);
                        for (std::size_t s = 0; s < S; s++) {
                            // input receptive-field base per tile pixel (independent of k)
                            const T* input_base[Q_BLOCK];
                            for (std::size_t qi = 0; qi < qn; qi++) {
                                const std::ptrdiff_t w = static_cast<std::ptrdiff_t>((q0 + qi) * S_W + s * D_W)
                                    - static_cast<std::ptrdiff_t>(P_W);
                                input_base[qi] = (w < 0 || w >= static_cast<std::ptrdiff_t>(W))
                                    ? nullptr
                                    : input_ptr_n + h_in * W * C + static_cast<std::size_t>(w) * C;
                            }
                            const T* const kernel_rs = kernel_ptr + r * S * C * K_pad + s * C * K_pad;
                            for (std::size_t c = 0; c < C; c++) {
                                const T* const kernel_c = kernel_rs + c * K_pad + k0;
                                xsimd::batch<T> wvec[KV_BLOCK];
                                for (std::size_t kv = 0; kv < kv_n; kv++) {
                                    wvec[kv] = xsimd::load_unaligned(kernel_c + kv * inc);
                                }
                                for (std::size_t qi = 0; qi < qn; qi++) {
                                    if (input_base[qi] == nullptr) continue;
                                    const xsimd::batch<T> in_b(input_base[qi][c]);
                                    for (std::size_t kv = 0; kv < kv_n; kv++) {
                                        acc[qi][kv] = xsimd::fma(in_b, wvec[kv], acc[qi][kv]);
                                    }
                                }
                            }
                        }
                    }
                    for (std::size_t qi = 0; qi < qn; qi++) {
                        T* const out_base = output_ptr_n + p * Q * K + (q0 + qi) * K + k0;
                        for (std::size_t kv = 0; kv < kv_n; kv++) {
                            const std::size_t k = k0 + kv * inc;
                            if (k + inc <= K) {
                                acc[qi][kv].store_unaligned(out_base + kv * inc);
                            } else {
                                // Last vector may cover padded channels; store
                                // only the lanes that exist in the real output.
                                store_partial(acc[qi][kv], out_base + kv * inc, K - k);
                            }
                        }
                    }
                }
            }
        }
    };
    tf::Taskflow taskflow("conv2d_block_rsck");
    taskflow.for_each_index(
        static_cast<std::size_t>(0), N, static_cast<std::size_t>(1),
        simd_k
    );
    executor.run(taskflow).wait();
    return output;
}

template <FloatLike T>
Tensor<T, 4> conv2d_simd_rsck(
    const Tensor<T, 4>& input,
    const Tensor<T, 4>& kernel,
    const std::array<std::size_t, 2>& stride = {1, 1},
    const Padding& padding = Padding(),
    const std::array<std::size_t, 2>& dilation = {1, 1}
) {
    // NHWC/RSCK format
    if (input.shape()[3] != kernel.shape()[2]) {
        throw std::invalid_argument("[conv.hpp][conv2d_simd_rsck] Input and kernel channel dimensions do not match");
    }
    const std::array<std::size_t, 4> kernel_shape_crsk = {
        kernel.shape()[2], kernel.shape()[0], kernel.shape()[1], kernel.shape()[3]
    };
    const std::array<std::size_t, 2> pad = handle_padding(
        input.shape(), kernel_shape_crsk, stride, padding, dilation
    );
    const std::array<std::size_t, 4> output_shape = calculate_output_shape(
        input.shape(), kernel_shape_crsk,
        stride, pad, dilation
    );
    Tensor<T, 4> output(output_shape);
    const auto [N, H, W, C] = input.shape();
    const auto [R, S, C_, K] = kernel.shape();
    const auto [N_, P, Q, K_] = output_shape;
    const auto [S_H, S_W] = stride;
    const auto [P_H, P_W] = pad;
    const auto [D_H, D_W] = dilation;
    T const* const input_ptr = input.values().data();
    T* const output_ptr = output.values().data();
    // Vectorize over the output-channel axis K (contiguous in RSCK): broadcast one
    // input value and FMA it against a vector of K weights. Each SIMD lane owns a
    // distinct output channel, so accumulators store directly to contiguous output
    // with no horizontal reduce_add.
    constexpr std::size_t inc = xsimd::batch<T>::size;
    // Zero-pad the weights' contiguous K axis to a full vector multiple
    // (XNNPACK-style) so every output-channel group runs full-width FMAs; the
    // padded lanes compute throwaway channels that are simply not stored.
    const std::size_t K_pad = round_up(K, inc);
    std::optional<Tensor<T, 4>> kernel_padded;
    if (K_pad != K) {
        kernel_padded = pad_last_axis(kernel, inc);
    }
    T const* const kernel_ptr = (kernel_padded ? *kernel_padded : kernel).values().data();
    auto implicit_gemm = [&](std::size_t n) -> void {
        T* const output_ptr_n = output_ptr + n * P * Q * K;
        T const* const input_ptr_n = input_ptr + n * H * W * C;
        for (std::size_t p = 0; p < P; p++) {
            for (std::size_t q = 0; q < Q; q++) {
                for (std::size_t k = 0; k < K_pad; k += inc) {
                    xsimd::batch<T> acc = xsimd::batch<T>(static_cast<T>(0));
                    for (std::size_t r = 0; r < R; r++) {
                        const std::ptrdiff_t h = static_cast<std::ptrdiff_t>(p * S_H + r * D_H)
                            - static_cast<std::ptrdiff_t>(P_H);
                        if (h < 0 || h >= static_cast<std::ptrdiff_t>(H)) continue;
                        const std::size_t h_in = static_cast<std::size_t>(h);
                        for (std::size_t s = 0; s < S; s++) {
                            const std::ptrdiff_t w = static_cast<std::ptrdiff_t>(q * S_W + s * D_W)
                                - static_cast<std::ptrdiff_t>(P_W);
                            if (w < 0 || w >= static_cast<std::ptrdiff_t>(W)) continue;
                            const std::size_t w_in = static_cast<std::size_t>(w);
                            const T* const input_base = input_ptr_n + h_in * W * C + w_in * C;
                            const T* const kernel_base = kernel_ptr + r * S * C * K_pad + s * C * K_pad + k;
                            for (std::size_t c = 0; c < C; c++) {
                                const xsimd::batch<T> in_b(input_base[c]);
                                const xsimd::batch<T> kernel_batch = xsimd::load_unaligned(kernel_base + c * K_pad);
                                acc = xsimd::fma(in_b, kernel_batch, acc);
                            }
                        }
                    }
                    if (k + inc <= K) {
                        acc.store_unaligned(output_ptr_n + p * Q * K + q * K + k);
                    } else {
                        // Last vector may cover padded channels; store only the
                        // lanes that exist in the real output.
                        store_partial(acc, output_ptr_n + p * Q * K + q * K + k, K - k);
                    }
                }
            }
        }
    };
    tf::Taskflow taskflow("conv2d_simd_rsck");
    taskflow.for_each_index(
        static_cast<std::size_t>(0), N, static_cast<std::size_t>(1),
        implicit_gemm
    );
    executor.run(taskflow).wait();
    return output;
}

template <FloatLike T>
Tensor<T, 4> conv2d_simd_krsc(
    const Tensor<T, 4>& input,
    const Tensor<T, 4>& kernel,
    const std::array<std::size_t, 2>& stride = {1, 1},
    const Padding& padding = Padding(),
    const std::array<std::size_t, 2>& dilation = {1, 1}
) {
    // NHWC/KRSC format
    if (input.shape()[3] != kernel.shape()[3]) {
        throw std::invalid_argument("[conv.hpp][conv2d_simd_krsc] Input and kernel channel dimensions do not match");
    }
    const std::array<std::size_t, 4> kernel_shape_crsk = {
        kernel.shape()[3], kernel.shape()[1], kernel.shape()[2], kernel.shape()[0]
    };
    const std::array<std::size_t, 2> pad = handle_padding(
        input.shape(), kernel_shape_crsk, stride, padding, dilation
    );
    const std::array<std::size_t, 4> output_shape = calculate_output_shape(
        input.shape(), kernel_shape_crsk,
        stride, pad, dilation
    );
    Tensor<T, 4> output(output_shape);
    const auto [N, H, W, C] = input.shape();
    const auto [K, R, S, C_] = kernel.shape();
    const auto [N_, P, Q, K_] = output_shape;
    const auto [S_H, S_W] = stride;
    const auto [P_H, P_W] = pad;
    const auto [D_H, D_W] = dilation;
    T const* const input_ptr = input.values().data();
    T* const output_ptr = output.values().data();
    constexpr std::size_t inc = xsimd::batch<T>::size;
    // Zero-pad the weights' contiguous C axis to a full vector multiple
    // (XNNPACK-style) so the reduction runs entirely in full-width FMAs: the
    // padded weight lanes are zero, and the matching input tail lanes are
    // zero-filled via load_partial (the input itself cannot be over-read).
    const std::size_t C_pad = round_up(C, inc);
    std::optional<Tensor<T, 4>> kernel_padded;
    if (C_pad != C) {
        kernel_padded = pad_last_axis(kernel, inc);
    }
    T const* const kernel_ptr = (kernel_padded ? *kernel_padded : kernel).values().data();
    const std::size_t vec_size = C - (C % inc); // channels covered by full input vectors
    auto implicit_gemm = [&](std::size_t n) -> void {
        T* const output_ptr_n = output_ptr + n * P * Q * K;
        for (std::size_t p = 0; p < P; p++) {
            for (std::size_t q = 0; q < Q; q++) {
                for (std::size_t k = 0; k < K; k++) {
                    xsimd::batch<T> sum_batch = xsimd::batch<T>(static_cast<T>(0));
                    for (std::size_t r = 0; r < R; r++) {
                        for (std::size_t s = 0; s < S; s++) {
                            const std::ptrdiff_t h = static_cast<std::ptrdiff_t>(p * S_H + r * D_H)
                                - static_cast<std::ptrdiff_t>(P_H);
                            if (h < 0 || h >= static_cast<std::ptrdiff_t>(H)) continue;
                            const std::ptrdiff_t w = static_cast<std::ptrdiff_t>(q * S_W + s * D_W)
                                - static_cast<std::ptrdiff_t>(P_W);
                            if (w < 0 || w >= static_cast<std::ptrdiff_t>(W)) continue;
                            const std::size_t h_in = static_cast<std::size_t>(h);
                            const std::size_t w_in = static_cast<std::size_t>(w);
                            const T* const input_base = input_ptr + n * H * W * C + h_in * W * C + w_in * C;
                            const T* const kernel_base = kernel_ptr + k * R * S * C_pad + r * S * C_pad + s * C_pad;
                            for (std::size_t c = 0; c < vec_size; c += inc) {
                                xsimd::batch<T> input_batch = xsimd::load_unaligned(input_base + c);
                                xsimd::batch<T> kernel_batch = xsimd::load_unaligned(kernel_base + c);
                                sum_batch = xsimd::fma(input_batch, kernel_batch, sum_batch);
                            }
                            if (vec_size != C) {
                                // Tail channel group [vec_size, C_pad): weight lanes past
                                // C are zero, input lanes past C are zero-filled.
                                const xsimd::batch<T> input_batch = load_partial(input_base + vec_size, C - vec_size);
                                const xsimd::batch<T> kernel_batch = xsimd::load_unaligned(kernel_base + vec_size);
                                sum_batch = xsimd::fma(input_batch, kernel_batch, sum_batch);
                            }
                        }
                    }
                    output_ptr_n[p * Q * K + q * K + k] = xsimd::reduce_add(sum_batch);
                }
            }
        }
    };
    tf::Taskflow taskflow("conv2d_simd_krsc");
    taskflow.for_each_index(
        static_cast<std::size_t>(0), N, static_cast<std::size_t>(1),
        implicit_gemm
    );
    executor.run(taskflow).wait();
    return output;
}

template <FloatLike T>
Tensor<T, 4> conv2d_padding_separate_krsc(
    const Tensor<T, 4>& input,
    const Tensor<T, 4>& kernel,
    const std::array<std::size_t, 2>& stride = {1, 1},
    const Padding& padding = Padding(),
    const std::array<std::size_t, 2>& dilation = {1, 1}
) {
    // NHWC/KRSC format
    if (input.shape()[3] != kernel.shape()[3]) {
        throw std::invalid_argument("[conv.hpp][conv2d_padding_separate] Input and kernel channel dimensions do not match");
    }
    const std::array<std::size_t, 4> kernel_shape_crsk = {
        kernel.shape()[3], kernel.shape()[1], kernel.shape()[2], kernel.shape()[0]
    };
    const std::array<std::size_t, 2> pad = handle_padding(
        input.shape(), kernel_shape_crsk, stride, padding, dilation
    );
    const std::array<std::size_t, 4> output_shape = calculate_output_shape(
        input.shape(), kernel_shape_crsk,
        stride, pad, dilation
    );
    Tensor<T, 4> output(output_shape);
    const auto [N, H, W, C] = input.shape();
    const auto [K, R, S, C_] = kernel.shape();
    const auto [N_, P, Q, K_] = output_shape;
    const auto [S_H, S_W] = stride;
    const auto [P_H, P_W] = pad;
    const auto [D_H, D_W] = dilation;
    // Interior region [PP_start, PP) x [QQ_start, QQ): every kernel tap lands
    // inside the input, so no bounds checking is needed there.
    auto interior_start = [](std::size_t p_pad, std::size_t s) -> std::size_t {
        return (p_pad + s - 1) / s; // ceil(pad / stride)
    };
    auto interior_end = [](
        std::size_t dim, std::size_t p_pad, std::size_t s, std::size_t ext, std::size_t out
    ) -> std::size_t {
        // largest output index o with o*s + ext - pad <= dim - 1
        const std::ptrdiff_t num = static_cast<std::ptrdiff_t>(dim) - 1
            + static_cast<std::ptrdiff_t>(p_pad) - static_cast<std::ptrdiff_t>(ext);
        if (num < 0) return 0;
        const std::size_t e = static_cast<std::size_t>(num) / s + 1;
        return e > out ? out : e;
    };
    const std::size_t PP_start = std::min(interior_start(P_H, S_H), P); // unpadded output height start index
    const std::size_t QQ_start = std::min(interior_start(P_W, S_W), Q); // unpadded output width start index
    const std::size_t PP = std::max(interior_end(H, P_H, S_H, (R - 1) * D_H, P), PP_start); // unpadded output height
    const std::size_t QQ = std::max(interior_end(W, P_W, S_W, (S - 1) * D_W, Q), QQ_start); // unpadded output width
    T const* const input_ptr = input.values().data();
    T const* const kernel_ptr = kernel.values().data();
    T* const output_ptr = output.values().data();
    constexpr std::size_t inc = xsimd::batch<T>::size;
    const std::size_t vec_size = C - (C % inc);
    // solve for each output pixel (p, q) in the unpadded output
    auto unpadded = [&](std::size_t n) -> void {
        T* const output_ptr_n = output_ptr + n * P * Q * K;
        for (std::size_t p = PP_start; p < PP; p++) {
            for (std::size_t q = QQ_start; q < QQ; q++) {
                for (std::size_t k = 0; k < K; k++) {
                    T sum = static_cast<T>(0);
                    xsimd::batch<T> sum_batch(static_cast<T>(0));
                    for (std::size_t r = 0; r < R; r++) {
                        for (std::size_t s = 0; s < S; s++) {
                            const std::size_t h_in = p * S_H + r * D_H - P_H;
                            const std::size_t w_in = q * S_W + s * D_W - P_W;
                            const T* const input_base = input_ptr + n * H * W * C + h_in * W * C + w_in * C;
                            const T* const kernel_base = kernel_ptr + k * R * S * C + r * S * C + s * C;
                            for (std::size_t c = 0; c < vec_size; c += inc) {
                                auto input_vec = xsimd::load_unaligned(input_base + c);
                                auto kernel_vec = xsimd::load_unaligned(kernel_base + c);
                                sum_batch = xsimd::fma(input_vec, kernel_vec, sum_batch);
                            }
                            for (std::size_t c = vec_size; c < C; c++) {
                                sum += input_base[c] * kernel_base[c];
                            }
                        }
                    }
                    output_ptr_n[p * Q * K + q * K + k] = sum + xsimd::reduce_add(sum_batch);
                }
            }
        }
    };
    // solve a single boundary output pixel (p, q) with bounds checking
    auto padded_pixel = [&](std::size_t n, T* const output_ptr_n, std::size_t p, std::size_t q) -> void {
        for (std::size_t k = 0; k < K; k++) {
            T sum = static_cast<T>(0);
            xsimd::batch<T> sum_batch(static_cast<T>(0));
            for (std::size_t r = 0; r < R; r++) {
                const std::ptrdiff_t h = static_cast<std::ptrdiff_t>(p * S_H + r * D_H)
                    - static_cast<std::ptrdiff_t>(P_H);
                if (h < 0 || h >= static_cast<std::ptrdiff_t>(H)) continue;
                for (std::size_t s = 0; s < S; s++) {
                    const std::ptrdiff_t w = static_cast<std::ptrdiff_t>(q * S_W + s * D_W)
                        - static_cast<std::ptrdiff_t>(P_W);
                    if (w < 0 || w >= static_cast<std::ptrdiff_t>(W)) continue;
                    const std::size_t h_in = static_cast<std::size_t>(h);
                    const std::size_t w_in = static_cast<std::size_t>(w);
                    const T* const input_base = input_ptr + n * H * W * C + h_in * W * C + w_in * C;
                    const T* const kernel_base = kernel_ptr + k * R * S * C + r * S * C + s * C;
                    for (std::size_t c = 0; c < vec_size; c += inc) {
                        auto input_vec = xsimd::load_unaligned(input_base + c);
                        auto kernel_vec = xsimd::load_unaligned(kernel_base + c);
                        sum_batch = xsimd::fma(input_vec, kernel_vec, sum_batch);
                    }
                    for (std::size_t c = vec_size; c < C; c++) {
                        sum += input_base[c] * kernel_base[c];
                    }
                }
            }
            output_ptr_n[p * Q * K + q * K + k] = sum + xsimd::reduce_add(sum_batch);
        }
    };
    // solve for each output pixel (p, q) in the padded frame around the interior
    auto padded = [&](std::size_t n) -> void {
        T* const output_ptr_n = output_ptr + n * P * Q * K;
        // top band: rows above the interior
        for (std::size_t p = 0; p < PP_start; p++) {
            for (std::size_t q = 0; q < Q; q++) {
                padded_pixel(n, output_ptr_n, p, q);
            }
        }
        // bottom band: rows below the interior
        for (std::size_t p = PP; p < P; p++) {
            for (std::size_t q = 0; q < Q; q++) {
                padded_pixel(n, output_ptr_n, p, q);
            }
        }
        // left and right bands: columns beside the interior
        for (std::size_t p = PP_start; p < PP; p++) {
            for (std::size_t q = 0; q < QQ_start; q++) {
                padded_pixel(n, output_ptr_n, p, q);
            }
            for (std::size_t q = QQ; q < Q; q++) {
                padded_pixel(n, output_ptr_n, p, q);
            }
        }
    };
    tf::Taskflow taskflow("conv2d_padding_separate_krsc");
    taskflow.for_each_index(
        static_cast<std::size_t>(0), N, static_cast<std::size_t>(1),
        [&](std::size_t n) -> void {
            unpadded(n);
            padded(n);
        }
    );
    executor.run(taskflow).wait();
    return output;
}

template <FloatLike T>
Tensor<T, 4> conv2d_implicit_gemm_krsc(
    const Tensor<T, 4>& input,
    const Tensor<T, 4>& kernel,
    const std::array<std::size_t, 2>& stride = {1, 1},
    const Padding& padding = Padding(),
    const std::array<std::size_t, 2>& dilation = {1, 1}
) {
    // NHWC/KRSC format
    if (input.shape()[3] != kernel.shape()[3]) {
        throw std::invalid_argument("[conv.hpp][conv2d_implicit_gemm] Input and kernel channel dimensions do not match");
    }
    const std::array<std::size_t, 4> kernel_shape_crsk = {
        kernel.shape()[3], kernel.shape()[1], kernel.shape()[2], kernel.shape()[0]
    };
    const std::array<std::size_t, 2> pad = handle_padding(
        input.shape(), kernel_shape_crsk, stride, padding, dilation
    );
    const std::array<std::size_t, 4> output_shape = calculate_output_shape(
        input.shape(), kernel_shape_crsk,
        stride, pad, dilation
    );
    Tensor<T, 4> output(output_shape);
    const auto [N, H, W, C] = input.shape();
    const auto [K, R, S, C_] = kernel.shape();
    const auto [N_, P, Q, K_] = output_shape;
    const auto [S_H, S_W] = stride;
    const auto [P_H, P_W] = pad;
    const auto [D_H, D_W] = dilation;
    T const* const input_ptr = input.values().data();
    T const* const kernel_ptr = kernel.values().data();
    T* const output_ptr = output.values().data();
    auto implicit_gemm = [&](std::size_t n) -> void {
        T* const output_ptr_n = output_ptr + n * P * Q * K;
        for (std::size_t p = 0; p < P; p++) {
            for (std::size_t q = 0; q < Q; q++) {
                for (std::size_t k = 0; k < K; k++) {
                    T sum = static_cast<T>(0);
                    for (std::size_t r = 0; r < R; r++) {
                        for (std::size_t s = 0; s < S; s++) {
                            const std::ptrdiff_t h = static_cast<std::ptrdiff_t>(p * S_H + r * D_H)
                                - static_cast<std::ptrdiff_t>(P_H);
                            if (h < 0 || h >= static_cast<std::ptrdiff_t>(H)) continue;
                            const std::ptrdiff_t w = static_cast<std::ptrdiff_t>(q * S_W + s * D_W)
                                - static_cast<std::ptrdiff_t>(P_W);
                            if (w < 0 || w >= static_cast<std::ptrdiff_t>(W)) continue;
                            const std::size_t h_in = static_cast<std::size_t>(h);
                            const std::size_t w_in = static_cast<std::size_t>(w);
                            const T* const input_base = input_ptr + n * H * W * C + h_in * W * C + w_in * C;
                            const T* const kernel_base = kernel_ptr + k * R * S * C + r * S * C + s * C;
                            for (std::size_t c = 0; c < C; c++) {
                                sum += input_base[c] * kernel_base[c];
                            }
                        }
                    }
                    output_ptr_n[p * Q * K + q * K + k] = sum;
                }
            }
        }
    };
    tf::Taskflow taskflow("conv2d_implicit_gemm");
    taskflow.for_each_index(
        static_cast<std::size_t>(0), N, static_cast<std::size_t>(1),
        implicit_gemm
    );
    executor.run(taskflow).wait();
    return output;
}

template <FloatLike T>
Tensor<T, 4> conv2d_explicit_gemm_crsk(
    const Tensor<T, 4>& input,
    const Tensor<T, 4>& kernel,
    const std::array<std::size_t, 2>& stride = {1, 1},
    const Padding& padding = Padding(),
    const std::array<std::size_t, 2>& dilation = {1, 1}
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
    const Tensor<T, 4>& input,
    const Tensor<T, 4>& kernel,
    const std::array<std::size_t, 2>& stride = {1, 1},
    const Padding& padding = Padding(),
    const std::array<std::size_t, 2>& dilation = {1, 1}
) {
    // NHWC/CRSK format
    if (input.shape()[3] != kernel.shape()[0]) {
        throw std::invalid_argument("[conv.hpp][conv2d_pipeline] Input and kernel channel dimensions do not match");
    }
    std::array<std::size_t, 2> pad = handle_padding(input.shape(), kernel.shape(), stride, padding, dilation);
    const std::size_t N = input.shape()[0];
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
    for (std::size_t i = 0; i < num_lines; i++) {
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
    tf::Taskflow taskflow("conv2d_pipeline");
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
                im2col2d(
                    input, kernel, stride, pad, dilation, pf.token(), buffer[pf.line()].im2col_result
                );
            }
        },
        tf::Pipe{
            tf::PipeType::PARALLEL,
            [&](tf::Pipeflow& pf) -> void {
                matmul<T>(
                    buffer[pf.line()].im2col_result, kernel_matrix, buffer[pf.line()].matmul_result
                );
            }
        },
        tf::Pipe{
            tf::PipeType::PARALLEL,
            [&](tf::Pipeflow& pf) -> void {
                col2im2d(buffer[pf.line()].matmul_result, output_shape, pf.token(), output);
            }
        }
    );
    tf::Task pipeline_task = taskflow.composed_of(pipeline).name("pipeline_task");
    kernel_task.precede(pipeline_task);
    executor.run(taskflow).wait();
    return output;
}

// Owning wrapper around an XNNPACK f32 convolution operator. The expensive,
// input-shape-invariant work — operator creation (which packs the weights) and
// reshape (which builds the indirection buffer + sizes the workspace) — happens
// once in the constructor. Each run() only rebinds the input/output pointers
// (setup) and executes (run), so a built operator can be reused across many
// inputs of the same shape without re-packing. This lets callers (e.g. the
// benchmark) hoist the build out of the timed region and measure run cost only.
class Conv2dXnnpack {
public:
    // kernel is KRSC; input_shape is NHWC.
    Conv2dXnnpack(
        const std::array<std::size_t, 4>& input_shape,
        const Tensor<float, 4>& kernel,
        const std::array<std::size_t, 2>& stride = {1, 1},
        const Padding& padding = Padding(),
        const std::array<std::size_t, 2>& dilation = {1, 1}
    ) : input_shape_(input_shape) {
        if (input_shape[3] != kernel.shape()[3]) {
            throw std::invalid_argument("[conv.hpp][Conv2dXnnpack] Input and kernel channel dimensions do not match");
        }
        const std::array<std::size_t, 4> kernel_crsk = {
            kernel.shape()[3], kernel.shape()[1], kernel.shape()[2], kernel.shape()[0]
        };
        const std::array<std::size_t, 2> pad = handle_padding(
            input_shape, kernel_crsk, stride, padding, dilation
        );
        output_shape_ = calculate_output_shape(
            input_shape, kernel_crsk, stride, pad, dilation
        );
        const auto [N, H, W, C] = input_shape;
        const auto [K, R, S, C_] = kernel.shape();
        ensure_initialized();
        // create: packs the weights into XNNPACK's internal blocked layout
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
            &op_
        );
        if (status != xnn_status_success) {
            throw std::runtime_error("[conv.hpp][Conv2dXnnpack] Failed to create XNNPACK conv2d operator");
        }
        // reshape: builds the indirection buffer and sizes the workspace
        std::size_t workspace_size = 0;
        std::size_t output_height = 0, output_width = 0;
        status = xnn_reshape_convolution2d_nhwc_f32(
            op_, N, H, W,
            &workspace_size,
            &output_height, &output_width,
            threadpool()
        );
        if (status != xnn_status_success) {
            xnn_delete_operator(op_);
            throw std::runtime_error("[conv.hpp][Conv2dXnnpack] Failed to reshape XNNPACK conv2d operator");
        }
        if (workspace_size > 0) {
            workspace_ = std::malloc(workspace_size);
        }
    }

    Conv2dXnnpack(const Conv2dXnnpack&) = delete;
    Conv2dXnnpack& operator=(const Conv2dXnnpack&) = delete;

    ~Conv2dXnnpack() {
        if (op_ != nullptr) {
            xnn_delete_operator(op_);
        }
        std::free(workspace_);
    }

    // setup (rebind pointers) + run. No weight packing or reshape here.
    Tensor<float, 4> run(const Tensor<float, 4>& input) {
        Tensor<float, 4> output(output_shape_);
        xnn_status status = xnn_setup_convolution2d_nhwc_f32(
            op_, workspace_,
            input.values().data(), output.values().data()
        );
        if (status != xnn_status_success) {
            throw std::runtime_error("[conv.hpp][Conv2dXnnpack] Failed to setup XNNPACK conv2d operator");
        }
        status = xnn_run_operator(op_, threadpool());
        if (status != xnn_status_success) {
            throw std::runtime_error("[conv.hpp][Conv2dXnnpack] Failed to run XNNPACK conv2d operator");
        }
        return output;
    }

    const std::array<std::size_t, 4>& output_shape() const { return output_shape_; }

private:
    static pthreadpool_t threadpool() {
        static pthreadpool_t tp = pthreadpool_create(8);
        return tp;
    }
    static void ensure_initialized() {
        static std::once_flag xnn_init_flag;
        std::call_once(xnn_init_flag, []() -> void {
            if (xnn_initialize(nullptr) != xnn_status_success) {
                throw std::runtime_error("[conv.hpp][Conv2dXnnpack] Failed to initialize XNNPACK");
            }
        });
    }

    xnn_operator_t op_ = nullptr;
    void* workspace_ = nullptr;
    std::array<std::size_t, 4> input_shape_{};
    std::array<std::size_t, 4> output_shape_{};
};

template <FloatLike T>
Tensor<T, 4> conv2d_xnnpack(
    const Tensor<T, 4>& input,
    const Tensor<T, 4>& kernel,
    const std::array<std::size_t, 2>& stride = {1, 1},
    const Padding& padding = Padding(),
    const std::array<std::size_t, 2>& dilation = {1, 1}
) {
    // NHWC/KRSC format. Full lifecycle per call (build + run); the benchmark
    // instead reuses a Conv2dXnnpack directly to time run() in isolation.
    static_assert(std::is_same_v<T, float>, "XNNPACK conv2d only supports float");
    Conv2dXnnpack op(input.shape(), kernel, stride, padding, dilation);
    return op.run(input);
}

} // namespace sc

#endif // CONV_HPP