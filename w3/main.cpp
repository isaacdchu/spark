#include <print>
#include <string>
#include <chrono>
#include <vector>
#include <array>
#include <set>
#include <fstream>

#include <taskflow.hpp>

#include <scone.hpp>

struct BenchmarkResult {
    std::size_t input_size;
    std::size_t kernel_size;
    double total_ms;
};
std::vector<BenchmarkResult> results;

void save_csv(const std::string& path, const std::vector<BenchmarkResult>& results) {
    // Saves results to a CSV file
    // Overwrite the existing file
    std::ofstream f(path, std::ios::trunc);
    if (!f) {
        throw std::runtime_error("Failed to open file for writing");
    }
    f <<
        "backend," \
        "input_size,kernel_size," \
        "total_ms\n";
    for (const auto& r: results) {
        f << std::format(
            "TASKFLOW," \
            "{},{}," \
            "{}\n",
            r.input_size, r.kernel_size,
            r.total_ms);
    }
}

void warmup() {
    for (std::size_t i = 0; i < 10; i++) {
        sc::Tensor<float, 4> input = sc::Tensor<float, 4>(16, 64, 64, 64);
        for (std::size_t i = 0; i < input.size(); i++) {
            input.values()[i] = static_cast<float>(i);
        }
        sc::Tensor<float, 4> kernel = sc::Tensor<float, 4>::ones(64, 3, 3, 64);
        sc::Tensor<float, 4> output = sc::conv2d(input, kernel);
    }
}

auto benchmark(
    const std::array<std::size_t, 4>& input_shape,
    const std::array<std::size_t, 4>& kernel_shape,
    const std::array<std::size_t, 2>& stride = {1, 1},
    std::variant<sc::padding_t, sc::Padding> padding = sc::Padding::VALID,
    const std::array<std::size_t, 2>& dilation = {1, 1}
) {
    std::chrono::microseconds mean_duration = std::chrono::microseconds(0);
    for (std::size_t i = 0; i < 10; i++) {
        sc::Tensor<float, 4> input = sc::Tensor<float, 4>(input_shape);
        for (std::size_t i = 0; i < input.size(); i++) {
            input.values()[i] = static_cast<float>(i);
        }
        sc::Tensor<float, 4> kernel = sc::Tensor<float, 4>::ones(kernel_shape);
        std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
        sc::Tensor<float, 4> output = sc::conv2d(input, kernel, stride, padding, dilation);
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        mean_duration += std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    }
    mean_duration /= 10;
    results.push_back(BenchmarkResult{
        .input_size = input_shape[1],
        .kernel_size = kernel_shape[1],
        .total_ms = static_cast<double>(mean_duration.count()) / 1000,
    });
    return mean_duration;
}

void bench_input_kernel_size() {
    auto warmup_start = std::chrono::high_resolution_clock::now();
    warmup();
    auto warmup_end = std::chrono::high_resolution_clock::now();
    std::vector<std::size_t> input_sizes = {16, 32, 64, 128};
    for (const auto& input_size : input_sizes) {
        std::set<std::size_t> kernel_sizes = {1, 3, 5, input_size / 4, input_size / 2};
        for (const auto& kernel_size : kernel_sizes) {
            std::array<std::size_t, 4> input_shape = {16, input_size, input_size, 3};
            std::array<std::size_t, 4> kernel_shape = {3, kernel_size, kernel_size, 3};
            auto mean_duration = benchmark(input_shape, kernel_shape);
            std::println("Input: {} {} {} {}, Kernel: {} {} {} {}, Conv2d time: {:.3f} ms",
                input_shape[0], input_shape[1], input_shape[2], input_shape[3],
                kernel_shape[0], kernel_shape[1], kernel_shape[2], kernel_shape[3],
                mean_duration.count() / 1000.0
            );
        }
    }
    save_csv("input_kernel_benchmark_results.csv", results);
    // std::println("Output:\n{}", output.to_string());
    auto warmup_duration = std::chrono::duration_cast<std::chrono::microseconds>(warmup_end - warmup_start);
    std::println("Warmup time: {:.3f} ms", warmup_duration.count() / 1000.0);
}

void bench_channel_stride_size() {
    auto warmup_start = std::chrono::high_resolution_clock::now();
    warmup();
    auto warmup_end = std::chrono::high_resolution_clock::now();
    std::vector<std::size_t> channel_sizes = {4, 8, 16, 32, 64, 128, 256};
    std::vector<std::size_t> stride_sizes = {1, 2, 4};
    for (const auto& channel_size : channel_sizes) {
        std::array<std::size_t, 4> input_shape = {16, 64, 64, channel_size};
        std::array<std::size_t, 4> kernel_shape = {channel_size, 3, 3, channel_size};
        for (const auto& stride_size : stride_sizes) {
            auto mean_duration = benchmark(input_shape, kernel_shape, {stride_size, stride_size});
            std::println("Input: {} {} {} {}, Kernel: {} {} {} {}, Stride: {}, Conv2d time: {:.3f} ms",
                input_shape[0], input_shape[1], input_shape[2], input_shape[3],
                kernel_shape[0], kernel_shape[1], kernel_shape[2], kernel_shape[3],
                stride_size,
                mean_duration.count() / 1000.0
            );
        }
    }
    save_csv("channel_stride_benchmark_results.csv", results);
    // std::println("Output:\n{}", output.to_string());
    auto warmup_duration = std::chrono::duration_cast<std::chrono::microseconds>(warmup_end - warmup_start);
    std::println("Warmup time: {:.3f} ms", warmup_duration.count() / 1000.0);
}

int main() {
    // warmup();
    bench_input_kernel_size();
    bench_channel_stride_size();
    return 0;
}