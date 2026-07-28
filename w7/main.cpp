#include <print>
#include <string>
#include <chrono>
#include <vector>
#include <array>
#include <fstream>
#include <functional>
#include <thread>
#include <memory>
#include <optional>

#include <taskflow.hpp>

#include <scone.hpp>

using ConvFn = std::function<sc::Tensor<float, 4>(
    const sc::Tensor<float, 4>&,
    const sc::Tensor<float, 4>&,
    const std::array<std::size_t, 2>&,
    const sc::Padding&,
    const std::array<std::size_t, 2>&
)>;

// A runner binds a fixed conv config (kernel + params) and computes an output
// for a given input. All per-config build work is done before it is returned.
using RunFn = std::function<sc::Tensor<float, 4>(const sc::Tensor<float, 4>&)>;

// Builds a runner for a fixed config, performing any expensive one-time setup
// (e.g. XNNPACK operator creation + weight packing) here so it stays OUT of the
// timed region and only the returned closure's cost is measured.
using PrepareFn = std::function<RunFn(
    const sc::Tensor<float, 4>&,        // kernel (already in backend format)
    const std::array<std::size_t, 4>&,  // input shape (NHWC)
    const std::array<std::size_t, 2>&,  // stride
    const sc::Padding&,                 // padding
    const std::array<std::size_t, 2>&   // dilation
)>;

struct Backend {
    std::string name;
    ConvFn fn;
    // Given (C, R, S, K), adjust the kernel shape for this backend's format
    std::function<
        sc::Tensor<float, 4>(const sc::Tensor<float, 4>&)
    > crsk_to_fn = [](const sc::Tensor<float, 4>& tensor) -> sc::Tensor<float, 4> {
        return tensor;
    };
    // Optional: when set, used instead of `fn` to isolate run cost from build
    // cost. When null, the timed runner just calls `fn` (which for the SIMD/GEMM
    // backends is pure compute, since their repack is already done in crsk_to_fn).
    PrepareFn prepare = nullptr;
    // Active register-tile dims {block_q, block_k} for the blocked backends;
    // empty for every non-block backend.
    std::optional<std::array<std::size_t, 2>> block = std::nullopt;
};

// Prebuild the XNNPACK operator (create + reshape + weight pack) outside the
// timed region; the returned runner only does setup (pointer rebind) + run.
RunFn xnnpack_prepare(
    const sc::Tensor<float, 4>& kernel,
    const std::array<std::size_t, 4>& input_shape,
    const std::array<std::size_t, 2>& stride,
    const sc::Padding& padding,
    const std::array<std::size_t, 2>& dilation
) {
    auto op = std::make_shared<sc::Conv2dXnnpack>(input_shape, kernel, stride, padding, dilation);
    return [op](const sc::Tensor<float, 4>& input) -> sc::Tensor<float, 4> {
        return op->run(input);
    };
}

// Prebuild the indirection-buffer operator (weight pack + offset table) outside
// the timed region; the returned runner only rebinds the indirection pointers
// and computes, so its timing excludes the build just as XNNPACK's does. `kernel`
// arrives already in RSCK layout (via crsk_to_rsck).
RunFn ind_rsck_prepare(
    const sc::Tensor<float, 4>& kernel,
    const std::array<std::size_t, 4>& input_shape,
    const std::array<std::size_t, 2>& stride,
    const sc::Padding& padding,
    const std::array<std::size_t, 2>& dilation
) {
    auto op = std::make_shared<sc::Conv2dIndRsck<float>>(input_shape, kernel, stride, padding, dilation);
    return [op](const sc::Tensor<float, 4>& input) -> sc::Tensor<float, 4> {
        return op->run(input);
    };
}

// Prebuild the SIMD/blocked weight pack (contiguous-axis zero-pad) outside the
// timed region; the returned runner runs pure compute on the pre-packed kernel,
// so its timing excludes the pack just as XNNPACK's excludes weight packing.
// `kernel` arrives already in the backend's layout (via its crsk_to_fn), and its
// last axis is the true (unpadded) count the *_prepacked core needs. `Prepacked`
// is the matching conv2d_*_prepacked function.
template <auto Prepacked>
RunFn simd_pack_prepare(
    const sc::Tensor<float, 4>& kernel,
    const std::array<std::size_t, 4>& /*input_shape*/,
    const std::array<std::size_t, 2>& stride,
    const sc::Padding& padding,
    const std::array<std::size_t, 2>& dilation
) {
    const std::size_t last = kernel.shape()[3];
    auto packed = std::make_shared<sc::Tensor<float, 4>>(sc::simd_pack_kernel(kernel));
    return [packed, last, stride, padding, dilation](const sc::Tensor<float, 4>& input) -> sc::Tensor<float, 4> {
        return Prepacked(input, *packed, last, stride, padding, dilation);
    };
}

std::vector<Backend> BACKENDS = {
    {"conv2d_implicit_gemm_krsc", sc::conv2d_implicit_gemm_krsc<float>, sc::crsk_to_krsc<float>},
    {"conv2d_explicit_gemm_crsk", sc::conv2d_explicit_gemm_crsk<float>},
    {"conv2d_pipeline", sc::conv2d_pipeline<float>},
    {"conv2d_padding_separate_krsc", sc::conv2d_padding_separate_krsc<float>, sc::crsk_to_krsc<float>},
    {"conv2d_simd_rsck", sc::conv2d_simd_rsck<float>, sc::crsk_to_rsck<float>,
        simd_pack_prepare<sc::conv2d_simd_rsck_prepacked<float>>},
    {"conv2d_simd_krsc", sc::conv2d_simd_krsc<float>, sc::crsk_to_krsc<float>,
        simd_pack_prepare<sc::conv2d_simd_krsc_prepacked<float>>},
    {"conv2d_block_rsck", sc::conv2d_block_rsck<float>, sc::crsk_to_rsck<float>,
        simd_pack_prepare<sc::conv2d_block_rsck_prepacked<float>>,
        std::array<std::size_t, 2>{sc::block_rsck_q, sc::block_rsck_kv}},
    {"conv2d_ind_rsck", sc::conv2d_ind_rsck<float>, sc::crsk_to_rsck<float>, ind_rsck_prepare,
        std::array<std::size_t, 2>{sc::block_rsck_q, sc::block_rsck_kv}},
    {"conv2d_xnnpack", sc::conv2d_xnnpack<float>, sc::crsk_to_krsc<float>, xnnpack_prepare}
};

std::chrono::microseconds warmup(const std::string& backend_name, std::size_t iters = 10) {
    auto backend_it = std::find_if(BACKENDS.begin(), BACKENDS.end(),
        [&](const Backend& backend) -> bool {
            return backend.name == backend_name;
        }
    );
    if (backend_it == BACKENDS.end()) {
        throw std::invalid_argument("[main.cpp][warmup] Backend not found: " + backend_name);
    }
    const Backend& backend = *backend_it;
    auto kernel_shape = std::array<std::size_t, 4>{64, 3, 3, 64};
    std::chrono::microseconds total{0};
    for (std::size_t i = 0; i < iters; i++) {
        sc::Tensor<float, 4> input(std::array<std::size_t, 4>{32, 128, 128, 64});
        for (std::size_t j = 0; j < input.size(); j++) {
            input.values()[j] = static_cast<float>(j % 127);
        }
        sc::Tensor<float, 4> kernel = backend.crsk_to_fn(sc::Tensor<float, 4>::ones(kernel_shape));
        auto start = std::chrono::high_resolution_clock::now();
        backend.fn(input, kernel, {1, 1}, sc::Padding(sc::PaddingEnum::SAME), {1, 1});
        auto end = std::chrono::high_resolution_clock::now();
        total += std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    }
    return total;
}

int main() {
    warmup("conv2d_xnnpack", 4);
    return 0;
}
