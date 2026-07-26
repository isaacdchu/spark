#include <print>
#include <string>
#include <chrono>
#include <vector>
#include <array>
#include <fstream>
#include <thread>
#include <memory>

#include <taskflow.hpp>

#include <scone.hpp>

// Warm up a backend by building it once and running it `iters` times. The kernel
// is canonical CRSK; make_conv2d converts it to the backend's layout in the
// constructor. Only run() is timed (build stays out of the timed region).
std::chrono::microseconds warmup(const std::string& backend_name, std::size_t iters = 10) {
    const std::array<std::size_t, 4> kernel_shape{64, 3, 3, 64};
    const std::array<std::size_t, 4> input_shape{32, 64, 64, 64};
    std::chrono::microseconds total{0};
    for (std::size_t i = 0; i < iters; i++) {
        sc::Tensor<float, 4> input(input_shape);
        for (std::size_t j = 0; j < input.size(); j++) {
            input.values()[j] = static_cast<float>(j % 127);
        }
        const sc::Tensor<float, 4> kernel = sc::Tensor<float, 4>::ones(kernel_shape);
        auto op = sc::make_conv2d<float>(
            backend_name, input.shape(), kernel,
            {1, 1}, sc::Padding(sc::PaddingEnum::SAME), {1, 1}
        );
        auto start = std::chrono::high_resolution_clock::now();
        op->run(input);
        auto end = std::chrono::high_resolution_clock::now();
        total += std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    }
    return total;
}

int main() {
    warmup("conv2d_simd_rsck", 4);
    return 0;
}
