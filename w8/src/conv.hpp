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
#include <memory>
#include <string>
#include <print>

#include <xsimd.hpp>
#include <taskflow.hpp>
#include <algorithm/pipeline.hpp>
#include <algorithm/for_each.hpp>
#include <cblas.h>
#include <omp.h>
#include <xnnpack.h>
// Include only the pieces we use, not the tiny_dnn.h umbrella. The umbrella
// drags in image I/O, dataset parsers, cereal registration, and an xtensor-based
// experimental Tensor -- none of which the conv path needs, and several of which
// do not compile under g++-15/C++23.
#include <tiny_dnn/network.h>
#include <tiny_dnn/layers/convolutional_layer.h>

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
    const int openblas_threads = static_cast<int>(openblas_get_num_threads());
    openblas_set_num_threads(static_cast<int>(sc::num_threads()));
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
    openblas_set_num_threads(static_cast<int>(openblas_threads));
}

// Register-tile block sizes for the blocked conv kernels. These are compile-time
// constants overridable via preprocessor defines so a sweep can benchmark
// different tile shapes by recompiling. The #ifndef defaults preserve the
// original hard-coded values exactly.
#ifndef SCONE_BLOCK_RSCK_Q
#define SCONE_BLOCK_RSCK_Q 6
#endif
#ifndef SCONE_BLOCK_RSCK_KV
#define SCONE_BLOCK_RSCK_KV 2
#endif

// Cache-block size for the C (input-channel) reduction in conv2d_ind_rsck. The
// reduction is split into panels of this many channels; each weight panel is
// reused across every output pixel while resident in cache, at the cost of
// read-modify-writing partial sums to the output once per panel. A value >= the
// actual C disables blocking (single panel, no output RMW).
//
// Default DISABLES blocking: measured on this machine (large shared L2) the full
// weight buffer already fits L2, so blocking eliminates no re-streaming and only
// adds output-RMW traffic plus register-to-memory round trips -- it is a net loss
// at every panel size tried (C=128/256). Override to a smaller value on hardware
// with a smaller L2, where trading output traffic for weight-reuse locality can
// pay; a sweep can tune it by recompiling.
#ifndef SCONE_IND_RSCK_C_BLOCK
#define SCONE_IND_RSCK_C_BLOCK 1000000
#endif

namespace sc {

// Compiled block sizes, readable from benchmark code to record which tile
// configuration a binary was built with.
constexpr std::size_t block_rsck_q = SCONE_BLOCK_RSCK_Q;
constexpr std::size_t block_rsck_kv = SCONE_BLOCK_RSCK_KV;
constexpr std::size_t ind_rsck_c_block = SCONE_IND_RSCK_C_BLOCK;

// Program-lifetime thread pool. Intentionally leaked (heap-allocated, never
// deleted) so its destructor does NOT run during static destruction at exit:
// tf::Executor::~Executor() joins its worker threads, and that join can deadlock
// against a worker parked in a condition wait during C++ runtime teardown, hang-
// ing the process after all work is done. Leaking lets the OS reclaim the threads
// on exit instead. All call sites use `executor` as an lvalue, unchanged.
inline tf::Executor& executor = *(new tf::Executor(num_threads()));

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

// KCRS = [out_channel][in_channel][ky][kx], tiny-dnn's (OIHW) weight order. The
// result's flat layout matches tiny-dnn's weight vector exactly, so the wrapper
// can load it with a straight copy.
template <FloatLike T>
Tensor<T, 4> crsk_to_kcrs(const Tensor<T, 4>& kernel) {
    const auto [C, R, S, K] = kernel.shape();
    Tensor<T, 4> converted(K, C, R, S);
    for (std::size_t c = 0; c < C; c++) {
        for (std::size_t r = 0; r < R; r++) {
            for (std::size_t s = 0; s < S; s++) {
                for (std::size_t k = 0; k < K; k++) {
                    converted.values()[k * C * R * S + c * R * S + r * S + s] =
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

// Zero-pad a kernel's contiguous (last) axis up to a SIMD-width multiple: the
// one-time "weight pack" the SIMD/blocked convs perform internally so their FMA
// reductions run full width. For the RSCK kernels this pads the K axis, for the
// KRSC kernel the C axis -- either way the last, contiguous axis. Exposed so a
// caller that reuses one kernel across many inputs can pack once and feed the
// result to the matching *_prepacked conv, keeping the pack out of any timed
// region -- mirroring how Conv2dXnnpack packs its weights once in its
// constructor rather than on every run().
template <FloatLike T>
Tensor<T, 4> simd_pack_kernel(const Tensor<T, 4>& kernel) {
    constexpr std::size_t inc = xsimd::batch<T>::size;
    const std::size_t last = kernel.shape()[3];
    if (round_up(last, inc) == last) {
        return kernel;
    }
    return pad_last_axis(kernel, inc);
}

// Common interface every convolution backend implements. The constructor (in
// each derived class) performs all input-shape-invariant setup exactly once --
// kernel layout conversion, weight packing, indirection/offset tables, operator
// creation, and buffer allocation. run() executes the convolution for a concrete
// input and records the wall time of the "fair" region (the compute a caller
// would actually pay per invocation); get_time() returns that duration for the
// most recent run(). Every derived constructor takes the identical arguments:
// (input_shape [NHWC], kernel [canonical CRSK], stride, padding, dilation).
template <FloatLike T>
class Conv2d {
public:
    virtual ~Conv2d() = default;
    // Run the convolution on `input` and record the elapsed time of the timed
    // region in last_time_.
    virtual Tensor<T, 4> run(const Tensor<T, 4>& input) = 0;
    // Duration of the most recent run()'s timed region.
    std::chrono::nanoseconds get_time() const { return last_time_; }
    virtual const std::array<std::size_t, 4>& output_shape() const = 0;
    // Register-tile dims {block_q, block_k} for the blocked backends; nullopt for
    // every other backend (its CSV block columns are left blank).
    virtual std::optional<std::array<std::size_t, 2>> block_config() const {
        return std::nullopt;
    }

protected:
    // Wrap a callable, storing its wall time in last_time_. Derived run()s call
    // this around exactly the region that should be timed and forward its result.
    template <class F>
    auto timed(F&& f) -> decltype(f()) {
        const auto start = std::chrono::high_resolution_clock::now();
        auto result = f();
        last_time_ = std::chrono::high_resolution_clock::now() - start;
        return result;
    }

    std::chrono::nanoseconds last_time_{0};
};

// Blocked RSCK conv, K-vectorized with a Q_BLOCK x KV_BLOCK register tile.
// One-time setup in the constructor: convert the canonical CRSK kernel to RSCK
// and zero-pad its contiguous K axis to a SIMD-width multiple (XNNPACK-style),
// so run()'s FMA reductions run full width; the padded channels are computed but
// never stored. run() is pure compute over the pre-packed kernel.
template <FloatLike T>
class Conv2dBlockRsck : public Conv2d<T> {
public:
    Conv2dBlockRsck(
        const std::array<std::size_t, 4>& input_shape,
        const Tensor<T, 4>& kernel,
        const std::array<std::size_t, 2>& stride = {1, 1},
        const Padding& padding = Padding(),
        const std::array<std::size_t, 2>& dilation = {1, 1}
    ) : input_shape_(input_shape), stride_(stride), padding_(padding), dilation_(dilation) {
        if (input_shape[3] != kernel.shape()[0]) {
            throw std::invalid_argument("[conv.hpp][Conv2dBlockRsck] Input and kernel channel dimensions do not match");
        }
        // canonical CRSK -> RSCK, then zero-pad the contiguous K axis once.
        const Tensor<T, 4> rsck = crsk_to_rsck<T>(kernel);
        K_ = rsck.shape()[3];
        kernel_ = simd_pack_kernel<T>(rsck);
        const std::array<std::size_t, 2> pad = handle_padding(
            input_shape, kernel.shape(), stride, padding, dilation
        );
        output_shape_ = calculate_output_shape(input_shape, kernel.shape(), stride, pad, dilation);
    }

    const std::array<std::size_t, 4>& output_shape() const override { return output_shape_; }
    std::optional<std::array<std::size_t, 2>> block_config() const override {
        return std::array<std::size_t, 2>{block_rsck_q, block_rsck_kv};
    }

    Tensor<T, 4> run(const Tensor<T, 4>& input) override {
      if (input.shape() != input_shape_) {
          throw std::invalid_argument("[conv.hpp][Conv2dBlockRsck] Input shape does not match the built operator");
      }
      const Tensor<T, 4>& kernel = kernel_;
      const std::size_t K = K_;
      const std::array<std::size_t, 2>& stride = stride_;
      const Padding& padding = padding_;
      const std::array<std::size_t, 2>& dilation = dilation_;
      return this->timed([&]() -> Tensor<T, 4> {
    // The padded K axis is what lives in the kernel tensor; the true K drives
    // the padding math, output shape, and which lanes get stored.
    const std::size_t K_pad = kernel.shape()[3];
    const std::array<std::size_t, 4> kernel_shape_crsk = {
        kernel.shape()[2], kernel.shape()[0], kernel.shape()[1], K
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
    const auto [R, S, C_, K_pad_] = kernel.shape();
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
    constexpr std::size_t Q_BLOCK = SCONE_BLOCK_RSCK_Q;
    constexpr std::size_t KV_BLOCK = SCONE_BLOCK_RSCK_KV;
    // Weights arrive pre-padded on the contiguous K axis (XNNPACK-style) so every
    // output-channel group runs full-width FMAs; the padded lanes compute
    // throwaway channels that are simply not stored.
    T const* const kernel_ptr = kernel.values().data();
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
      });
    }

private:
    std::array<std::size_t, 4> input_shape_{};
    std::array<std::size_t, 4> output_shape_{};
    std::array<std::size_t, 2> stride_{1, 1};
    Padding padding_{};
    std::array<std::size_t, 2> dilation_{1, 1};
    Tensor<T, 4> kernel_;  // packed RSCK (K padded), set in constructor
    std::size_t K_ = 0;    // true (unpadded) K
};

// Owning wrapper around an indirection-buffer RSCK convolution. This mirrors
// Conv2dXnnpack's lifecycle so the two can be timed on the same footing: all
// input-shape-invariant work -- prepacking the weights and building the
// indirection offset table -- happens once in the constructor, and each run()
// only rebinds the indirection pointers for the concrete input (the
// shape-invariant analogue of xnn_setup) and streams the tiled FMA compute.
//
// Indirection buffer: for every (output pixel, kernel tap) pair we precompute,
// once, the element offset of the C-channel input pixel that tap reads -- or a
// sentinel meaning "this tap lands in the padding". At run() time each task
// resolves its batch slice of those offsets into concrete pointers, with padding
// taps pointed at a shared zero buffer. The hot loop then dereferences a pointer
// per (pixel, tap) with no address arithmetic and no bounds branch: padding taps
// read C zeros and contribute nothing to the reduction, exactly as XNNPACK's
// microkernels treat an indirection entry that points at its zero buffer.
//
// Weight prepacking: the raw RSCK weights are reordered once, in the constructor,
// into exactly the order the tiled compute consumes them -- grouped as
// [k-group][tap][c][KVW], where a k-group is the KV_BLOCK*inc-wide block of
// output channels the inner loop holds in registers. Because the compute walks
// k-groups then taps then c, this layout makes the inner-loop weight reads a
// single sequential forward stream over the whole packed buffer per output tile,
// instead of the K_pad-strided gather the plain RSCK layout forces (each c a
// cache line apart). Output channels past the true K are zero-filled in the pack
// (throwaway lanes, computed but never stored), so K is rounded up to a whole
// number of k-groups. This is the same idea as XNNPACK packing weights into its
// internal blocked layout in xnn_create.
//
// C-dimension cache blocking: the C (input-channel) reduction is split into
// panels of SCONE_IND_RSCK_C_BLOCK channels, with the panel loop wrapped OUTSIDE
// the output-pixel sweep. Each weight panel (RS*C_BLOCK*K_pack floats) is thus
// streamed once and reused across every output pixel while it stays cache-
// resident, instead of the whole weight buffer being re-streamed per output tile.
// The price is that partial sums are read-modify-written to the output once per
// panel (the first panel writes, later panels accumulate), so this trades a
// little output traffic for weight-reuse locality -- the classic GEMM "kc"
// blocking. A block size >= C collapses to a single panel with no output RMW.
//
// The compute is otherwise the same K-vectorized register tile as
// conv2d_block_rsck (Q_BLOCK output pixels x KV_BLOCK channel-vectors), so this
// backend isolates the combined effect of the indirection buffer, weight
// prepacking, and C-blocking against that kernel.
template <FloatLike T>
class Conv2dIndRsck : public Conv2d<T> {
public:
    // kernel is canonical CRSK; input_shape is NHWC. The CRSK->RSCK conversion,
    // the offset table, and the weight prepack all happen once here.
    Conv2dIndRsck(
        const std::array<std::size_t, 4>& input_shape,
        const Tensor<T, 4>& kernel_crsk,
        const std::array<std::size_t, 2>& stride = {1, 1},
        const Padding& padding = Padding(),
        const std::array<std::size_t, 2>& dilation = {1, 1}
    ) : input_shape_(input_shape) {
        // canonical CRSK -> RSCK (C on axis 2, K on axis 3); the rest of the
        // constructor consumes the RSCK form exactly as before.
        const Tensor<T, 4> kernel = crsk_to_rsck<T>(kernel_crsk);
        if (input_shape[3] != kernel.shape()[2]) {
            throw std::invalid_argument("[conv.hpp][Conv2dIndRsck] Input and kernel channel dimensions do not match");
        }
        K_ = kernel.shape()[3];
        const std::array<std::size_t, 4> kernel_shape_crsk = {
            kernel.shape()[2], kernel.shape()[0], kernel.shape()[1], K_
        };
        const std::array<std::size_t, 2> pad = handle_padding(
            input_shape, kernel_shape_crsk, stride, padding, dilation
        );
        output_shape_ = calculate_output_shape(
            input_shape, kernel_shape_crsk, stride, pad, dilation
        );
        std::tie(N_, H_, W_, C_) = std::tuple{input_shape[0], input_shape[1], input_shape[2], input_shape[3]};
        R_ = kernel.shape()[0];
        S_ = kernel.shape()[1];
        P_ = output_shape_[1];
        Q_ = output_shape_[2];
        const auto [S_H, S_W] = stride;
        const auto [P_H, P_W] = pad;
        const auto [D_H, D_W] = dilation;
        // Build the indirection offset table: one entry per (output pixel, tap).
        // A valid tap stores the element offset (channel 0) of the input pixel it
        // reads within a single image; an out-of-bounds tap stores PADDING. The
        // per-image base (n*H*W*C) is added at run() time, so this table is
        // input-data-invariant and built exactly once, like xnn_reshape's buffer.
        const std::size_t RS = R_ * S_;
        offsets_.resize(P_ * Q_ * RS);
        for (std::size_t p = 0; p < P_; p++) {
            for (std::size_t q = 0; q < Q_; q++) {
                const std::size_t pixel = p * Q_ + q;
                for (std::size_t r = 0; r < R_; r++) {
                    const std::ptrdiff_t h = static_cast<std::ptrdiff_t>(p * S_H + r * D_H)
                        - static_cast<std::ptrdiff_t>(P_H);
                    for (std::size_t s = 0; s < S_; s++) {
                        const std::ptrdiff_t w = static_cast<std::ptrdiff_t>(q * S_W + s * D_W)
                            - static_cast<std::ptrdiff_t>(P_W);
                        const std::size_t t = r * S_ + s;
                        const bool in_bounds = h >= 0 && h < static_cast<std::ptrdiff_t>(H_)
                            && w >= 0 && w < static_cast<std::ptrdiff_t>(W_);
                        offsets_[pixel * RS + t] = in_bounds
                            ? static_cast<std::ptrdiff_t>(static_cast<std::size_t>(h) * W_ * C_
                                + static_cast<std::size_t>(w) * C_)
                            : PADDING;
                    }
                }
            }
        }
        // Shared zero buffer that padding taps point at; sized to the C channels
        // the reduction reads per pixel.
        zeros_.assign(C_, static_cast<T>(0));
        // Per-run pointer buffer, reused across run() calls (shape is fixed).
        indirection_.resize(N_ * P_ * Q_ * RS);
        // Prepack weights into the tile's consumption order: [k-group][tap][c][KVW].
        // K is rounded up to a whole number of KVW-wide k-groups; channels past the
        // true K are zero (throwaway lanes, computed but never stored). As the
        // compute walks k-group -> tap -> c, this makes each output tile's weight
        // reads one contiguous forward stream over packed_, replacing the plain
        // RSCK layout's K_pad-strided (one-cache-line-per-c) gather.
        constexpr std::size_t inc = xsimd::batch<T>::size;
        constexpr std::size_t KVW = SCONE_BLOCK_RSCK_KV * inc;
        K_pack_ = round_up(K_, KVW);
        const std::size_t num_kgroups = K_pack_ / KVW;
        packed_.assign(num_kgroups * RS * C_ * KVW, static_cast<T>(0));
        T const* const kraw = kernel.values().data(); // RSCK: ((r*S+s)*C + c)*K + k
        for (std::size_t kg = 0; kg < num_kgroups; kg++) {
            for (std::size_t t = 0; t < RS; t++) {
                for (std::size_t c = 0; c < C_; c++) {
                    T* const dst = packed_.data() + ((kg * RS + t) * C_ + c) * KVW;
                    for (std::size_t lk = 0; lk < KVW; lk++) {
                        const std::size_t kchan = kg * KVW + lk;
                        dst[lk] = (kchan < K_) ? kraw[(t * C_ + c) * K_ + kchan] : static_cast<T>(0);
                    }
                }
            }
        }
    }

    // Rebind the indirection pointers for `input`, then run the tiled compute. No
    // weight packing or offset-table build happens here. The whole rebind+compute
    // is the timed region (matching how the benchmark timed op->run() wholesale).
    Tensor<T, 4> run(const Tensor<T, 4>& input) override {
        if (input.shape() != input_shape_) {
            throw std::invalid_argument("[conv.hpp][Conv2dIndRsck] Input shape does not match the built operator");
        }
        return this->timed([&]() -> Tensor<T, 4> {
        Tensor<T, 4> output(output_shape_);
        const std::size_t RS = R_ * S_;
        const std::size_t PQRS = P_ * Q_ * RS;
        T const* const input_ptr = input.values().data();
        T const* const packed_ptr = packed_.data();
        T const* const zero_ptr = zeros_.data();
        T* const output_ptr = output.values().data();
        const std::ptrdiff_t* const offsets = offsets_.data();
        T const** const indirection = indirection_.data();
        const std::size_t K = K_;
        const std::size_t K_pack = K_pack_;
        const std::size_t C = C_;
        const std::size_t R = R_;
        const std::size_t S = S_;
        const std::size_t P = P_;
        const std::size_t Q = Q_;
        constexpr std::size_t inc = xsimd::batch<T>::size;
        constexpr std::size_t Q_BLOCK = SCONE_BLOCK_RSCK_Q;
        constexpr std::size_t KV_BLOCK = SCONE_BLOCK_RSCK_KV;
        constexpr std::size_t KVW = KV_BLOCK * inc; // output channels per k-group
        constexpr std::size_t C_BLOCK = SCONE_IND_RSCK_C_BLOCK;
        auto compute = [&](std::size_t n) -> void {
            // setup: resolve this image's offset slice into concrete pointers,
            // padding taps -> the shared zero buffer. One branch per (pixel, tap)
            // here keeps every branch out of the O(K*C) compute loop below.
            T const* const image_base = input_ptr + n * H_ * W_ * C;
            T const** const ptrs = indirection + n * PQRS;
            for (std::size_t i = 0; i < PQRS; i++) {
                ptrs[i] = (offsets[i] == PADDING) ? zero_ptr : image_base + offsets[i];
            }
            T* const output_ptr_n = output_ptr + n * P * Q * K;
            // C cache-blocking: sweep all output pixels once per channel panel so
            // the panel's weights stay cache-resident across the sweep. The first
            // panel writes the output; later panels read-modify-write partial sums.
            for (std::size_t c0 = 0; c0 < C; c0 += C_BLOCK) {
                const std::size_t c1 = std::min(C, c0 + C_BLOCK);
                const bool first_panel = (c0 == 0);
                for (std::size_t p = 0; p < P; p++) {
                    for (std::size_t q0 = 0; q0 < Q; q0 += Q_BLOCK) {
                        const std::size_t qn = std::min(Q_BLOCK, Q - q0);
                        // Every k-group is a full KVW-wide block (K padded up to a KVW
                        // multiple in the pack), so kv_n is always KV_BLOCK here; padded
                        // channels past K are simply not stored below.
                        for (std::size_t k0 = 0; k0 < K_pack; k0 += KVW) {
                            const std::size_t kg = k0 / KVW;
                            xsimd::batch<T> acc[Q_BLOCK][KV_BLOCK];
                            // Seed accumulators: zero on the first panel, else reload
                            // this tile's partial sums written by earlier panels.
                            for (std::size_t qi = 0; qi < qn; qi++) {
                                T* const out_base = output_ptr_n + p * Q * K + (q0 + qi) * K + k0;
                                for (std::size_t kv = 0; kv < KV_BLOCK; kv++) {
                                    const std::size_t k = k0 + kv * inc;
                                    if (first_panel || k >= K) {
                                        acc[qi][kv] = xsimd::batch<T>(static_cast<T>(0));
                                    } else if (k + inc <= K) {
                                        acc[qi][kv] = xsimd::load_unaligned(out_base + kv * inc);
                                    } else {
                                        acc[qi][kv] = load_partial(out_base + kv * inc, K - k);
                                    }
                                }
                            }
                            for (std::size_t r = 0; r < R; r++) {
                                for (std::size_t s = 0; s < S; s++) {
                                    const std::size_t t = r * S + s;
                                    // Indirection: input receptive-field base per tile
                                    // pixel, already resolved (padding -> zeros). No
                                    // address math, no bounds branch in this loop.
                                    const T* input_base[Q_BLOCK];
                                    for (std::size_t qi = 0; qi < qn; qi++) {
                                        input_base[qi] = ptrs[(p * Q + q0 + qi) * RS + t];
                                    }
                                    // Prepacked weights: this (k-group, tap) block is a
                                    // contiguous [C][KVW] run, so the c-loop below walks
                                    // packed_ forward with unit stride (no K_pad gather).
                                    const T* const wbase_kt = packed_ptr + (kg * RS + t) * C * KVW;
                                    for (std::size_t c = c0; c < c1; c++) {
                                        const T* const wbase = wbase_kt + c * KVW;
                                        xsimd::batch<T> wvec[KV_BLOCK];
                                        for (std::size_t kv = 0; kv < KV_BLOCK; kv++) {
                                            wvec[kv] = xsimd::load_unaligned(wbase + kv * inc);
                                        }
                                        for (std::size_t qi = 0; qi < qn; qi++) {
                                            const xsimd::batch<T> in_b(input_base[qi][c]);
                                            for (std::size_t kv = 0; kv < KV_BLOCK; kv++) {
                                                acc[qi][kv] = xsimd::fma(in_b, wvec[kv], acc[qi][kv]);
                                            }
                                        }
                                    }
                                }
                            }
                            for (std::size_t qi = 0; qi < qn; qi++) {
                                T* const out_base = output_ptr_n + p * Q * K + (q0 + qi) * K + k0;
                                for (std::size_t kv = 0; kv < KV_BLOCK; kv++) {
                                    const std::size_t k = k0 + kv * inc;
                                    if (k + inc <= K) {
                                        acc[qi][kv].store_unaligned(out_base + kv * inc);
                                    } else if (k < K) {
                                        // Straddles the true-K boundary: store live lanes.
                                        store_partial(acc[qi][kv], out_base + kv * inc, K - k);
                                    }
                                    // else: k >= K, a fully padded k-group vector; skip.
                                }
                            }
                        }
                    }
                }
            }
        };
        tf::Taskflow taskflow("conv2d_ind_rsck");
        taskflow.for_each_index(
            static_cast<std::size_t>(0), N_, static_cast<std::size_t>(1),
            compute
        );
        executor.run(taskflow).wait();
        return output;
        });
    }

    const std::array<std::size_t, 4>& output_shape() const override { return output_shape_; }
    std::optional<std::array<std::size_t, 2>> block_config() const override {
        return std::array<std::size_t, 2>{block_rsck_q, block_rsck_kv};
    }

private:
    static constexpr std::ptrdiff_t PADDING = std::numeric_limits<std::ptrdiff_t>::min();

    std::array<std::size_t, 4> input_shape_{};
    std::array<std::size_t, 4> output_shape_{};
    std::size_t N_ = 0, H_ = 0, W_ = 0, C_ = 0;
    std::size_t R_ = 0, S_ = 0, K_ = 0, K_pack_ = 0, P_ = 0, Q_ = 0;
    std::vector<T> packed_;                 // prepacked weights: [k-group][tap][c][KVW]
    std::vector<std::ptrdiff_t> offsets_;   // P*Q*R*S element offsets (or PADDING)
    std::vector<T> zeros_;                  // shared zero buffer for padding taps
    std::vector<T const*> indirection_;     // N*P*Q*R*S resolved pointers, per run
};

// SIMD RSCK conv, K-vectorized. One-time setup in the constructor: convert the
// canonical CRSK kernel to RSCK and zero-pad its contiguous K axis to a SIMD-
// width multiple, so run()'s FMA reductions run full width; the padded channels
// are computed but never stored. run() is pure compute over the pre-packed kernel.
template <FloatLike T>
class Conv2dSimdRsck : public Conv2d<T> {
public:
    Conv2dSimdRsck(
        const std::array<std::size_t, 4>& input_shape,
        const Tensor<T, 4>& kernel,
        const std::array<std::size_t, 2>& stride = {1, 1},
        const Padding& padding = Padding(),
        const std::array<std::size_t, 2>& dilation = {1, 1}
    ) : input_shape_(input_shape), stride_(stride), padding_(padding), dilation_(dilation) {
        if (input_shape[3] != kernel.shape()[0]) {
            throw std::invalid_argument("[conv.hpp][Conv2dSimdRsck] Input and kernel channel dimensions do not match");
        }
        const Tensor<T, 4> rsck = crsk_to_rsck<T>(kernel);
        K_ = rsck.shape()[3];
        kernel_ = simd_pack_kernel<T>(rsck);
        const std::array<std::size_t, 2> pad = handle_padding(
            input_shape, kernel.shape(), stride, padding, dilation
        );
        output_shape_ = calculate_output_shape(input_shape, kernel.shape(), stride, pad, dilation);
    }

    const std::array<std::size_t, 4>& output_shape() const override { return output_shape_; }

    Tensor<T, 4> run(const Tensor<T, 4>& input) override {
      if (input.shape() != input_shape_) {
          throw std::invalid_argument("[conv.hpp][Conv2dSimdRsck] Input shape does not match the built operator");
      }
      const Tensor<T, 4>& kernel = kernel_;
      const std::size_t K = K_;
      const std::array<std::size_t, 2>& stride = stride_;
      const Padding& padding = padding_;
      const std::array<std::size_t, 2>& dilation = dilation_;
      return this->timed([&]() -> Tensor<T, 4> {
    // NHWC input; kernel is RSCK with K padded to a SIMD-width multiple.
    if (input.shape()[3] != kernel.shape()[2]) {
        throw std::invalid_argument("[conv.hpp][Conv2dSimdRsck] Input and kernel channel dimensions do not match");
    }
    // The padded K axis is what lives in the kernel tensor; the true K drives
    // the padding math, output shape, and which lanes get stored.
    const std::size_t K_pad = kernel.shape()[3];
    const std::array<std::size_t, 4> kernel_shape_crsk = {
        kernel.shape()[2], kernel.shape()[0], kernel.shape()[1], K
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
    const auto [R, S, C_, K_pad_] = kernel.shape();
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
    // Weights arrive pre-padded on the contiguous K axis (XNNPACK-style) so every
    // output-channel group runs full-width FMAs; the padded lanes compute
    // throwaway channels that are simply not stored.
    T const* const kernel_ptr = kernel.values().data();
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
      });
    }

private:
    std::array<std::size_t, 4> input_shape_{};
    std::array<std::size_t, 4> output_shape_{};
    std::array<std::size_t, 2> stride_{1, 1};
    Padding padding_{};
    std::array<std::size_t, 2> dilation_{1, 1};
    Tensor<T, 4> kernel_;  // packed RSCK (K padded), set in constructor
    std::size_t K_ = 0;    // true (unpadded) K
};

// SIMD KRSC conv, C-reduction vectorized. One-time setup in the constructor:
// convert the canonical CRSK kernel to KRSC and zero-pad its contiguous C axis
// to a SIMD-width multiple, so the reduction runs entirely in full-width FMAs;
// the padded weight lanes are zero and the matching input tail lanes are zero-
// filled. run() is pure compute over the pre-packed kernel.
template <FloatLike T>
class Conv2dSimdKrsc : public Conv2d<T> {
public:
    Conv2dSimdKrsc(
        const std::array<std::size_t, 4>& input_shape,
        const Tensor<T, 4>& kernel,
        const std::array<std::size_t, 2>& stride = {1, 1},
        const Padding& padding = Padding(),
        const std::array<std::size_t, 2>& dilation = {1, 1}
    ) : input_shape_(input_shape), stride_(stride), padding_(padding), dilation_(dilation) {
        if (input_shape[3] != kernel.shape()[0]) {
            throw std::invalid_argument("[conv.hpp][Conv2dSimdKrsc] Input and kernel channel dimensions do not match");
        }
        const Tensor<T, 4> krsc = crsk_to_krsc<T>(kernel);
        C_ = krsc.shape()[3];
        kernel_ = simd_pack_kernel<T>(krsc);
        const std::array<std::size_t, 2> pad = handle_padding(
            input_shape, kernel.shape(), stride, padding, dilation
        );
        output_shape_ = calculate_output_shape(input_shape, kernel.shape(), stride, pad, dilation);
    }

    const std::array<std::size_t, 4>& output_shape() const override { return output_shape_; }

    Tensor<T, 4> run(const Tensor<T, 4>& input) override {
      if (input.shape() != input_shape_) {
          throw std::invalid_argument("[conv.hpp][Conv2dSimdKrsc] Input shape does not match the built operator");
      }
      const Tensor<T, 4>& kernel = kernel_;
      const std::size_t C = C_;
      const std::array<std::size_t, 2>& stride = stride_;
      const Padding& padding = padding_;
      const std::array<std::size_t, 2>& dilation = dilation_;
      return this->timed([&]() -> Tensor<T, 4> {
    // NHWC input; kernel is KRSC with C padded to a SIMD-width multiple.
    if (input.shape()[3] != C) {
        throw std::invalid_argument("[conv.hpp][Conv2dSimdKrsc] Input and kernel channel dimensions do not match");
    }
    // The padded C axis is what lives in the kernel tensor; the true C drives the
    // input indexing, the reduction extent, and the padding math.
    const std::size_t C_pad = kernel.shape()[3];
    const std::array<std::size_t, 4> kernel_shape_crsk = {
        C, kernel.shape()[1], kernel.shape()[2], kernel.shape()[0]
    };
    const std::array<std::size_t, 2> pad = handle_padding(
        input.shape(), kernel_shape_crsk, stride, padding, dilation
    );
    const std::array<std::size_t, 4> output_shape = calculate_output_shape(
        input.shape(), kernel_shape_crsk,
        stride, pad, dilation
    );
    Tensor<T, 4> output(output_shape);
    const auto [N, H, W, C_in] = input.shape();
    const auto [K, R, S, C_pad_] = kernel.shape();
    const auto [N_, P, Q, K_] = output_shape;
    const auto [S_H, S_W] = stride;
    const auto [P_H, P_W] = pad;
    const auto [D_H, D_W] = dilation;
    T const* const input_ptr = input.values().data();
    T* const output_ptr = output.values().data();
    constexpr std::size_t inc = xsimd::batch<T>::size;
    // Weights arrive pre-padded on the contiguous C axis (XNNPACK-style) so the
    // reduction runs entirely in full-width FMAs: the padded weight lanes are
    // zero, and the matching input tail lanes are zero-filled via load_partial
    // (the input itself cannot be over-read).
    T const* const kernel_ptr = kernel.values().data();
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
      });
    }

private:
    std::array<std::size_t, 4> input_shape_{};
    std::array<std::size_t, 4> output_shape_{};
    std::array<std::size_t, 2> stride_{1, 1};
    Padding padding_{};
    std::array<std::size_t, 2> dilation_{1, 1};
    Tensor<T, 4> kernel_;  // packed KRSC (C padded), set in constructor
    std::size_t C_ = 0;    // true (unpadded) C
};

// Padding-separated KRSC conv: the interior (every tap in-bounds) runs a branch-
// free vectorized reduction, the boundary bands run a bounds-checked variant.
// One-time setup in the constructor: convert the canonical CRSK kernel to KRSC.
template <FloatLike T>
class Conv2dPaddingSeparateKrsc : public Conv2d<T> {
public:
    Conv2dPaddingSeparateKrsc(
        const std::array<std::size_t, 4>& input_shape,
        const Tensor<T, 4>& kernel,
        const std::array<std::size_t, 2>& stride = {1, 1},
        const Padding& padding = Padding(),
        const std::array<std::size_t, 2>& dilation = {1, 1}
    ) : input_shape_(input_shape), stride_(stride), padding_(padding), dilation_(dilation) {
        if (input_shape[3] != kernel.shape()[0]) {
            throw std::invalid_argument("[conv.hpp][Conv2dPaddingSeparateKrsc] Input and kernel channel dimensions do not match");
        }
        kernel_ = crsk_to_krsc<T>(kernel);
        const std::array<std::size_t, 2> pad = handle_padding(
            input_shape, kernel.shape(), stride, padding, dilation
        );
        output_shape_ = calculate_output_shape(input_shape, kernel.shape(), stride, pad, dilation);
    }

    const std::array<std::size_t, 4>& output_shape() const override { return output_shape_; }

    Tensor<T, 4> run(const Tensor<T, 4>& input) override {
      if (input.shape() != input_shape_) {
          throw std::invalid_argument("[conv.hpp][Conv2dPaddingSeparateKrsc] Input shape does not match the built operator");
      }
      const Tensor<T, 4>& kernel = kernel_;
      const std::array<std::size_t, 2>& stride = stride_;
      const Padding& padding = padding_;
      const std::array<std::size_t, 2>& dilation = dilation_;
      return this->timed([&]() -> Tensor<T, 4> {
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
      });
    }

private:
    std::array<std::size_t, 4> input_shape_{};
    std::array<std::size_t, 4> output_shape_{};
    std::array<std::size_t, 2> stride_{1, 1};
    Padding padding_{};
    std::array<std::size_t, 2> dilation_{1, 1};
    Tensor<T, 4> kernel_;  // KRSC, set in constructor
};

// Implicit-GEMM KRSC conv: the naive direct reduction, kept as a scalar baseline.
// One-time setup in the constructor: convert the canonical CRSK kernel to KRSC.
template <FloatLike T>
class Conv2dImplicitGemmKrsc : public Conv2d<T> {
public:
    Conv2dImplicitGemmKrsc(
        const std::array<std::size_t, 4>& input_shape,
        const Tensor<T, 4>& kernel,
        const std::array<std::size_t, 2>& stride = {1, 1},
        const Padding& padding = Padding(),
        const std::array<std::size_t, 2>& dilation = {1, 1}
    ) : input_shape_(input_shape), stride_(stride), padding_(padding), dilation_(dilation) {
        if (input_shape[3] != kernel.shape()[0]) {
            throw std::invalid_argument("[conv.hpp][Conv2dImplicitGemmKrsc] Input and kernel channel dimensions do not match");
        }
        kernel_ = crsk_to_krsc<T>(kernel);
        const std::array<std::size_t, 2> pad = handle_padding(
            input_shape, kernel.shape(), stride, padding, dilation
        );
        output_shape_ = calculate_output_shape(input_shape, kernel.shape(), stride, pad, dilation);
    }

    const std::array<std::size_t, 4>& output_shape() const override { return output_shape_; }

    Tensor<T, 4> run(const Tensor<T, 4>& input) override {
      if (input.shape() != input_shape_) {
          throw std::invalid_argument("[conv.hpp][Conv2dImplicitGemmKrsc] Input shape does not match the built operator");
      }
      const Tensor<T, 4>& kernel = kernel_;
      const std::array<std::size_t, 2>& stride = stride_;
      const Padding& padding = padding_;
      const std::array<std::size_t, 2>& dilation = dilation_;
      return this->timed([&]() -> Tensor<T, 4> {
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
      });
    }

private:
    std::array<std::size_t, 4> input_shape_{};
    std::array<std::size_t, 4> output_shape_{};
    std::array<std::size_t, 2> stride_{1, 1};
    Padding padding_{};
    std::array<std::size_t, 2> dilation_{1, 1};
    Tensor<T, 4> kernel_;  // KRSC, set in constructor
};

// Explicit-GEMM CRSK conv: per-sample im2col -> BLAS matmul -> col2im. One-time
// setup in the constructor: im2col the kernel into its GEMM matrix form (hoisted
// out of the per-call, per-sample work). Kernel stays in canonical CRSK.
template <FloatLike T>
class Conv2dExplicitGemmCrsk : public Conv2d<T> {
public:
    Conv2dExplicitGemmCrsk(
        const std::array<std::size_t, 4>& input_shape,
        const Tensor<T, 4>& kernel,
        const std::array<std::size_t, 2>& stride = {1, 1},
        const Padding& padding = Padding(),
        const std::array<std::size_t, 2>& dilation = {1, 1}
    ) : input_shape_(input_shape), stride_(stride), padding_(padding), dilation_(dilation), kernel_(kernel) {
        if (input_shape[3] != kernel.shape()[0]) {
            throw std::invalid_argument("[conv.hpp][Conv2dExplicitGemmCrsk] Input and kernel channel dimensions do not match");
        }
        const std::array<std::size_t, 2> pad = handle_padding(
            input_shape, kernel.shape(), stride, padding, dilation
        );
        output_shape_ = calculate_output_shape(input_shape, kernel.shape(), stride, pad, dilation);
        kernel_matrix_ = Tensor<T, 2>(kernel.size() / kernel.shape()[3], kernel.shape()[3]);
        im2col2d(kernel, kernel_matrix_);
    }

    const std::array<std::size_t, 4>& output_shape() const override { return output_shape_; }

    Tensor<T, 4> run(const Tensor<T, 4>& input) override {
      if (input.shape() != input_shape_) {
          throw std::invalid_argument("[conv.hpp][Conv2dExplicitGemmCrsk] Input shape does not match the built operator");
      }
      const Tensor<T, 4>& kernel = kernel_;
      const Tensor<T, 2>& kernel_matrix = kernel_matrix_;
      const std::array<std::size_t, 2>& stride = stride_;
      const Padding& padding = padding_;
      const std::array<std::size_t, 2>& dilation = dilation_;
      return this->timed([&]() -> Tensor<T, 4> {
    std::array<std::size_t, 2> pad = handle_padding(input.shape(), kernel.shape(), stride, padding, dilation);
    const std::array<std::size_t, 4> output_shape = calculate_output_shape(
        input.shape(), kernel.shape(),
        stride, pad, dilation
    );
    Tensor<T, 4> output(output_shape);
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
      });
    }

private:
    std::array<std::size_t, 4> input_shape_{};
    std::array<std::size_t, 4> output_shape_{};
    std::array<std::size_t, 2> stride_{1, 1};
    Padding padding_{};
    std::array<std::size_t, 2> dilation_{1, 1};
    Tensor<T, 4> kernel_;         // canonical CRSK
    Tensor<T, 2> kernel_matrix_;  // im2col of the kernel, built once
};

// Pipelined explicit-GEMM CRSK conv: a Taskflow pipeline overlaps im2col, matmul,
// and col2im across samples over a fixed pool of line buffers. One-time setup in
// the constructor: im2col the kernel and allocate the line buffers (both hoisted
// out of the per-call work; the old kernel_task dependency is gone since the
// kernel matrix is ready before run()). Kernel stays in canonical CRSK.
template <FloatLike T>
class Conv2dPipeline : public Conv2d<T> {
public:
    Conv2dPipeline(
        const std::array<std::size_t, 4>& input_shape,
        const Tensor<T, 4>& kernel,
        const std::array<std::size_t, 2>& stride = {1, 1},
        const Padding& padding = Padding(),
        const std::array<std::size_t, 2>& dilation = {1, 1}
    ) : input_shape_(input_shape), stride_(stride), padding_(padding), dilation_(dilation), kernel_(kernel) {
        if (input_shape[3] != kernel.shape()[0]) {
            throw std::invalid_argument("[conv.hpp][Conv2dPipeline] Input and kernel channel dimensions do not match");
        }
        const std::array<std::size_t, 2> pad = handle_padding(
            input_shape, kernel.shape(), stride, padding, dilation
        );
        output_shape_ = calculate_output_shape(input_shape, kernel.shape(), stride, pad, dilation);
        for (std::size_t i = 0; i < num_lines; i++) {
            buffer_[i].im2col_result = Tensor<T, 2>(
                output_shape_[1] * output_shape_[2], kernel.size() / kernel.shape()[3]
            );
            buffer_[i].matmul_result = Tensor<T, 2>(
                output_shape_[1] * output_shape_[2], kernel.shape()[3]
            );
            buffer_[i].im2col_result.values().resize(buffer_[i].im2col_result.size());
            buffer_[i].matmul_result.values().resize(buffer_[i].matmul_result.size());
        }
        kernel_matrix_ = Tensor<T, 2>(kernel.size() / kernel.shape()[3], kernel.shape()[3]);
        im2col2d(kernel, kernel_matrix_);
    }

    const std::array<std::size_t, 4>& output_shape() const override { return output_shape_; }

    Tensor<T, 4> run(const Tensor<T, 4>& input) override {
      if (input.shape() != input_shape_) {
          throw std::invalid_argument("[conv.hpp][Conv2dPipeline] Input shape does not match the built operator");
      }
      const Tensor<T, 4>& kernel = kernel_;
      const std::array<std::size_t, 2>& stride = stride_;
      const Padding& padding = padding_;
      const std::array<std::size_t, 2>& dilation = dilation_;
      std::array<PipeData, num_lines>& buffer = buffer_;
      Tensor<T, 2>& kernel_matrix = kernel_matrix_;
      return this->timed([&]() -> Tensor<T, 4> {
    std::array<std::size_t, 2> pad = handle_padding(input.shape(), kernel.shape(), stride, padding, dilation);
    const std::size_t N = input.shape()[0];
    const std::array<std::size_t, 4> output_shape = calculate_output_shape(
        input.shape(), kernel.shape(),
        stride, pad, dilation
    );
    Tensor<T, 4> output = Tensor<T, 4>(output_shape);
    tf::Taskflow taskflow("conv2d_pipeline");
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
    executor.run(taskflow).wait();
    return output;
      });
    }

private:
    static constexpr std::size_t num_lines = 4;
    struct PipeData {
        Tensor<T, 2> im2col_result;
        Tensor<T, 2> matmul_result;
    };
    std::array<std::size_t, 4> input_shape_{};
    std::array<std::size_t, 4> output_shape_{};
    std::array<std::size_t, 2> stride_{1, 1};
    Padding padding_{};
    std::array<std::size_t, 2> dilation_{1, 1};
    Tensor<T, 4> kernel_;                       // canonical CRSK
    std::array<PipeData, num_lines> buffer_{};  // reused pipeline line buffers
    Tensor<T, 2> kernel_matrix_;                // im2col of the kernel, built once
};

// Owning wrapper around an XNNPACK f32 convolution operator. The expensive,
// input-shape-invariant work — operator creation (which packs the weights) and
// reshape (which builds the indirection buffer + sizes the workspace) — happens
// once in the constructor. Each run() only rebinds the input/output pointers
// (setup) and executes (run), so a built operator can be reused across many
// inputs of the same shape without re-packing. This lets callers (e.g. the
// benchmark) hoist the build out of the timed region and measure run cost only.
class Conv2dXnnpack : public Conv2d<float> {
public:
    // kernel is canonical CRSK; input_shape is NHWC. The CRSK->KRSC conversion
    // and XNNPACK's own weight pack both happen once here.
    Conv2dXnnpack(
        const std::array<std::size_t, 4>& input_shape,
        const Tensor<float, 4>& crsk,
        const std::array<std::size_t, 2>& stride = {1, 1},
        const Padding& padding = Padding(),
        const std::array<std::size_t, 2>& dilation = {1, 1}
    ) : input_shape_(input_shape) {
        // canonical CRSK -> KRSC (XNNPACK's OHWI weight order); the rest of the
        // constructor consumes the KRSC form exactly as before.
        const Tensor<float, 4> kernel = crsk_to_krsc<float>(crsk);
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

    // setup (rebind pointers) + run, both inside the timed region (matching how
    // the benchmark timed op->run() wholesale). No weight packing or reshape here.
    Tensor<float, 4> run(const Tensor<float, 4>& input) override {
        return this->timed([&]() -> Tensor<float, 4> {
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
        });
    }

    const std::array<std::size_t, 4>& output_shape() const override { return output_shape_; }

private:
    static pthreadpool_t threadpool() {
        static pthreadpool_t tp = pthreadpool_create(num_threads());
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

// Owning wrapper around a tiny-dnn f32 convolution. Like Conv2dXnnpack, the
// build-once work (constructing the convolutional_layer, allocating and loading
// the weights) happens in the constructor, so run() only does the input/output
// layout conversion and the forward pass. tiny-dnn's convolutional_layer supports
// only valid/same padding, so arbitrary symmetric padding is emulated by
// pre-padding the input and running a valid conv; dilation and stride are handled
// natively by the layer. The kernel arrives already in tiny-dnn's KCRS
// ([out][in][ky][kx]) weight order via crsk_to_kcrs, so weights load by a copy.
class Conv2dTinydnn : public Conv2d<float> {
public:
    // kernel is canonical CRSK; input_shape is NHWC. The CRSK->KCRS conversion
    // and the layer build + weight load all happen once here.
    Conv2dTinydnn(
        const std::array<std::size_t, 4>& input_shape,
        const Tensor<float, 4>& crsk,
        const std::array<std::size_t, 2>& stride = {1, 1},
        const Padding& padding = Padding(),
        const std::array<std::size_t, 2>& dilation = {1, 1}
    ) : input_shape_(input_shape) {
        // canonical CRSK -> KCRS ([K][C][R][S]), tiny-dnn's weight order; the rest
        // of the constructor consumes the KCRS form exactly as before.
        const Tensor<float, 4> kernel = crsk_to_kcrs<float>(crsk);
        const auto [K, C, R, S] = kernel.shape();
        if (input_shape[3] != C) {
            throw std::invalid_argument("[conv.hpp][Conv2dTinydnn] Input and kernel channel dimensions do not match");
        }
        const auto [N, H, W, C_in] = input_shape;
        // handle_padding / calculate_output_shape want the kernel in CRSK layout
        // (spatial dims at [1],[2]; out channels at [3]); reconstruct it from the
        // KCRS shape the kernel now arrives in.
        const std::array<std::size_t, 4> kernel_crsk = {C, R, S, K};
        pad_ = handle_padding(input_shape, kernel_crsk, stride, padding, dilation);
        output_shape_ = calculate_output_shape(input_shape, kernel_crsk, stride, pad_, dilation);
        padded_h_ = H + 2 * pad_[0];
        padded_w_ = W + 2 * pad_[1];
        channels_ = C;

        // Build the layer over the (already padded) input dims using valid
        // padding; stride and dilation are applied natively by tiny-dnn.
        // has_bias must stay true: Conv2dOp::compute unconditionally reads the
        // bias input edge, so a bias-less layer dereferences a missing edge and
        // crashes. We keep the bias and zero it below.
        net_ << tiny_dnn::convolutional_layer(
            padded_w_, padded_h_,
            S, R,
            C, K,
            tiny_dnn::padding::valid,
            /*has_bias=*/true,
            stride[1], stride[0],
            dilation[1], dilation[0]
        );
        net_.init_weight();

        // Zero the bias so the layer computes a pure convolution.
        std::fill(net_[0]->weights()[1]->begin(), net_[0]->weights()[1]->end(), 0.0f);

        // KCRS flat layout matches tiny-dnn's weight vector exactly, so copy.
        tiny_dnn::vec_t& weights = *net_[0]->weights()[0];
        const std::vector<float>& kv = kernel.values();
        std::copy(kv.begin(), kv.end(), weights.begin());
    }

    Conv2dTinydnn(const Conv2dTinydnn&) = delete;
    Conv2dTinydnn& operator=(const Conv2dTinydnn&) = delete;

    // NHWC input -> tiny-dnn's batched planar CHW (zero-padded). tiny-dnn batches
    // as one tensor_t per sample, each holding a single input-edge vec_t. Split
    // out from the forward pass so the benchmark can hoist this layout transpose
    // out of the timed region: it is not convolution work, and unlike the
    // NHWC-native backends tiny-dnn would otherwise be charged for adapting to a
    // format it does not use.
    std::vector<tiny_dnn::tensor_t> to_batch(const Tensor<float, 4>& input) const {
        const auto [N, H, W, C] = input_shape_;
        const std::size_t plane = padded_h_ * padded_w_;
        std::vector<tiny_dnn::tensor_t> batch(
            N, tiny_dnn::tensor_t{tiny_dnn::vec_t(channels_ * plane, 0.0f)});
        const std::vector<float>& iv = input.values();
        for (std::size_t n = 0; n < N; n++) {
            tiny_dnn::vec_t& dst = batch[n][0];
            for (std::size_t h = 0; h < H; h++) {
                for (std::size_t w = 0; w < W; w++) {
                    for (std::size_t c = 0; c < C; c++) {
                        const std::size_t si = ((n * H + h) * W + w) * C + c;
                        const std::size_t di = (c * padded_h_ + (h + pad_[0])) * padded_w_ + (w + pad_[1]);
                        dst[di] = iv[si];
                    }
                }
            }
        }
        return batch;
    }

    // The convolution forward pass on an already-converted batch. This is the
    // fair unit to time against the other backends. tiny-dnn parallelizes over
    // the batch with `#pragma omp parallel for`, whose width defaults to
    // omp_get_max_threads() (all cores) and otherwise ignores our thread setting.
    // Pin it to num_threads() so the sweep's SCONE_NUM_THREADS takes effect,
    // matching how XNNPACK sizes its pthreadpool.
    std::vector<tiny_dnn::tensor_t> forward(const std::vector<tiny_dnn::tensor_t>& batch) {
        omp_set_num_threads(static_cast<int>(num_threads()));
        return net_.predict(batch);
    }

    // tiny-dnn planar CHW output -> NHWC. results[n][0] is sample n's output edge.
    Tensor<float, 4> from_output(const std::vector<tiny_dnn::tensor_t>& results) const {
        const auto [ON, P, Q, KO] = output_shape_;
        Tensor<float, 4> output(output_shape_);
        std::vector<float>& ov = output.values();
        for (std::size_t n = 0; n < ON; n++) {
            const tiny_dnn::vec_t& src = results[n][0];
            for (std::size_t k = 0; k < KO; k++) {
                for (std::size_t p = 0; p < P; p++) {
                    for (std::size_t q = 0; q < Q; q++) {
                        const std::size_t si = (k * P + p) * Q + q;
                        const std::size_t di = ((n * P + p) * Q + q) * KO + k;
                        ov[di] = src[si];
                    }
                }
            }
        }
        return output;
    }

    // Full lifecycle: convert in, convert out around the forward pass. Only the
    // forward pass is timed -- the NHWC<->CHW transpose is not convolution work,
    // so it stays out of the timed region (preserving the benchmark's fair
    // comparison against the NHWC-native backends).
    Tensor<float, 4> run(const Tensor<float, 4>& input) override {
        const std::vector<tiny_dnn::tensor_t> batch = to_batch(input);
        const std::vector<tiny_dnn::tensor_t> results =
            this->timed([&]() -> std::vector<tiny_dnn::tensor_t> { return forward(batch); });
        return from_output(results);
    }

    const std::array<std::size_t, 4>& output_shape() const override { return output_shape_; }

private:
    tiny_dnn::network<tiny_dnn::sequential> net_;
    std::array<std::size_t, 4> input_shape_{};
    std::array<std::size_t, 4> output_shape_{};
    std::array<std::size_t, 2> pad_{};
    std::size_t padded_h_ = 0;
    std::size_t padded_w_ = 0;
    std::size_t channels_ = 0;
};

// Factory: build a convolution backend by name. The kernel is canonical CRSK
// (shape [C, R, S, K]); each backend converts it to its own layout and does all
// other one-time setup in its constructor. XNNPACK and tiny-dnn are float-only,
// so they are only constructible when T is float; requesting them for another T,
// or passing an unknown name, throws std::invalid_argument.
template <FloatLike T>
std::unique_ptr<Conv2d<T>> make_conv2d(
    const std::string& name,
    const std::array<std::size_t, 4>& input_shape,
    const Tensor<T, 4>& kernel,
    const std::array<std::size_t, 2>& stride = {1, 1},
    const Padding& padding = Padding(),
    const std::array<std::size_t, 2>& dilation = {1, 1}
) {
    if (name == "conv2d_implicit_gemm_krsc")
        return std::make_unique<Conv2dImplicitGemmKrsc<T>>(input_shape, kernel, stride, padding, dilation);
    if (name == "conv2d_explicit_gemm_crsk")
        return std::make_unique<Conv2dExplicitGemmCrsk<T>>(input_shape, kernel, stride, padding, dilation);
    if (name == "conv2d_pipeline")
        return std::make_unique<Conv2dPipeline<T>>(input_shape, kernel, stride, padding, dilation);
    if (name == "conv2d_padding_separate_krsc")
        return std::make_unique<Conv2dPaddingSeparateKrsc<T>>(input_shape, kernel, stride, padding, dilation);
    if (name == "conv2d_simd_rsck")
        return std::make_unique<Conv2dSimdRsck<T>>(input_shape, kernel, stride, padding, dilation);
    if (name == "conv2d_simd_krsc")
        return std::make_unique<Conv2dSimdKrsc<T>>(input_shape, kernel, stride, padding, dilation);
    if (name == "conv2d_block_rsck")
        return std::make_unique<Conv2dBlockRsck<T>>(input_shape, kernel, stride, padding, dilation);
    if (name == "conv2d_ind_rsck")
        return std::make_unique<Conv2dIndRsck<T>>(input_shape, kernel, stride, padding, dilation);
    if constexpr (std::is_same_v<T, float>) {
        if (name == "conv2d_xnnpack")
            return std::make_unique<Conv2dXnnpack>(input_shape, kernel, stride, padding, dilation);
        if (name == "conv2d_tinydnn")
            return std::make_unique<Conv2dTinydnn>(input_shape, kernel, stride, padding, dilation);
    }
    throw std::invalid_argument("[conv.hpp][make_conv2d] Unknown or unsupported backend: " + name);
}

} // namespace sc

#endif // CONV_HPP