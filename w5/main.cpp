#include <print>
#include <string>
#include <chrono>
#include <vector>
#include <array>
#include <fstream>
#include <functional>
#include <thread>

#include <taskflow.hpp>

#include <scone.hpp>

using ConvFn = std::function<sc::Tensor<float, 4>(
    const sc::Tensor<float, 4>&,
    const sc::Tensor<float, 4>&,
    const std::array<std::size_t, 2>&,
    const sc::Padding&,
    const std::array<std::size_t, 2>&
)>;

struct Backend {
    std::string name;
    ConvFn fn;
    // Given (C_in, R, S, K_out), return the kernel shape for this backend's format
    std::function<
        sc::Tensor<float, 4>(const sc::Tensor<float, 4>&)
    > crsk_to_fn = [](const sc::Tensor<float, 4>& tensor) -> sc::Tensor<float, 4> {
        return tensor;
    };
};

std::vector<Backend> BACKENDS = {
    {"conv2d_implicit_gemm_krsc", sc::conv2d_implicit_gemm_krsc<float>, sc::crsk_to_krsc<float>},
    {"conv2d_explicit_gemm", sc::conv2d_explicit_gemm<float>},
    {"conv2d_pipeline", sc::conv2d_pipeline<float>},
    {"conv2d_padding_separate_krsc", sc::conv2d_padding_separate_krsc<float>, sc::crsk_to_krsc<float>},
    {"conv2d_simd_rsck", sc::conv2d_simd_rsck<float>, sc::crsk_to_rsck<float>},
    {"conv2d_xnnpack", sc::conv2d_xnnpack<float>, sc::crsk_to_krsc<float>}
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
        sc::Tensor<float, 4> input(std::array<std::size_t, 4>{32, 64, 64, 64});
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
    warmup("conv2d_simd_rskc", 4);
    return 0;
}
