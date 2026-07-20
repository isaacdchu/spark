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

struct BenchmarkResult {
    std::string backend;
    std::size_t input_size;
    std::size_t kernel_size;
    std::size_t channels;
    std::size_t stride;
    double total_ms;
};

using ConvFn = std::function<sc::Tensor<float, 4>(
    const sc::Tensor<float, 4>&,
    const sc::Tensor<float, 4>&,
    const std::array<std::size_t, 2>&,
    std::variant<sc::padding_t, sc::Padding>,
    const std::array<std::size_t, 2>&
)>;

struct Backend {
    std::string name;
    ConvFn fn;
    // Given (C_in, R, S, K_out), return the kernel shape for this backend's format
    std::function<std::array<std::size_t, 4>(
        std::size_t, std::size_t, std::size_t, std::size_t
    )> make_kernel_shape;
};

void save_csv(const std::string& path, const std::vector<BenchmarkResult>& results) {
    std::ofstream f(path, std::ios::trunc);
    if (!f) {
        throw std::runtime_error("Failed to open file for writing");
    }
    f << "backend,input_size,kernel_size,channels,stride,total_ms\n";
    for (const auto& r : results) {
        f << std::format("{},{},{},{},{},{:.4f}\n",
            r.backend, r.input_size, r.kernel_size, r.channels, r.stride, r.total_ms);
    }
}

BenchmarkResult run_benchmark(
    const Backend& backend,
    std::size_t input_size,
    std::size_t kernel_size,
    std::size_t channels,
    std::size_t stride_val,
    int iters = 10
) {
    std::array<std::size_t, 4> input_shape = {16, input_size, input_size, channels};
    std::array<std::size_t, 4> kernel_shape = backend.make_kernel_shape(channels, kernel_size, kernel_size, channels);
    std::array<std::size_t, 2> stride = {stride_val, stride_val};

    std::chrono::microseconds total{0};
    for (int i = 0; i < iters; i++) {
        sc::Tensor<float, 4> input(input_shape);
        for (std::size_t j = 0; j < input.size(); j++) {
            input.values()[j] = static_cast<float>(j % 127);
        }
        sc::Tensor<float, 4> kernel = sc::Tensor<float, 4>::ones(kernel_shape);

        auto start = std::chrono::high_resolution_clock::now();
        sc::Tensor<float, 4> output = backend.fn(input, kernel, stride, sc::Padding::VALID, {1, 1});
        auto end = std::chrono::high_resolution_clock::now();
        total += std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    }
    total /= iters;

    return BenchmarkResult{
        .backend = backend.name,
        .input_size = input_size,
        .kernel_size = kernel_size,
        .channels = channels,
        .stride = stride_val,
        .total_ms = static_cast<double>(total.count()) / 1000.0,
    };
}

std::chrono::microseconds warmup(const Backend& backend, std::size_t iters = 5) {
    auto kernel_shape = backend.make_kernel_shape(64, 3, 3, 64);
    std::chrono::microseconds total{0};
    for (std::size_t i = 0; i < iters; i++) {
        sc::Tensor<float, 4> input(std::array<std::size_t, 4>{16, 64, 64, 64});
        for (std::size_t j = 0; j < input.size(); j++) {
            input.values()[j] = static_cast<float>(j % 127);
        }
        sc::Tensor<float, 4> kernel = sc::Tensor<float, 4>::ones(kernel_shape);
        auto start = std::chrono::high_resolution_clock::now();
        backend.fn(input, kernel, {1, 1}, sc::Padding::VALID, {1, 1});
        auto end = std::chrono::high_resolution_clock::now();
        total += std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    }
    return total;
}

std::vector<Backend> BACKENDS = {
    {
        "conv2d_implicit_gemm",
        [](const sc::Tensor<float, 4>& in, const sc::Tensor<float, 4>& k,
            const std::array<std::size_t, 2>& s, std::variant<sc::padding_t, sc::Padding> p,
            const std::array<std::size_t, 2>& d) {
            return sc::conv2d_implicit_gemm(in, k, s, p, d);
        },
        [](std::size_t c, std::size_t r, std::size_t s, std::size_t k) -> std::array<std::size_t, 4> {
            return {c, r, s, k}; // CRSK
        }
    },
    {
        "conv2d_explicit_gemm",
        [](const sc::Tensor<float, 4>& in, const sc::Tensor<float, 4>& k,
            const std::array<std::size_t, 2>& s, std::variant<sc::padding_t, sc::Padding> p,
            const std::array<std::size_t, 2>& d) {
            return sc::conv2d_explicit_gemm(in, k, s, p, d);
        },
        [](std::size_t c, std::size_t r, std::size_t s, std::size_t k) -> std::array<std::size_t, 4> {
            return {c, r, s, k}; // CRSK
        }
    },
    {
        "conv2d_krsc",
        [](const sc::Tensor<float, 4>& in, const sc::Tensor<float, 4>& k,
            const std::array<std::size_t, 2>& s, std::variant<sc::padding_t, sc::Padding> p,
            const std::array<std::size_t, 2>& d) {
            return sc::conv2d_krsc(in, k, s, p, d);
        },
        [](std::size_t c, std::size_t r, std::size_t s, std::size_t k) -> std::array<std::size_t, 4> {
            return {k, r, s, c}; // KRSC
        }
    },
    {
        "conv2d_rsck",
        [](const sc::Tensor<float, 4>& in, const sc::Tensor<float, 4>& k,
            const std::array<std::size_t, 2>& s, std::variant<sc::padding_t, sc::Padding> p,
            const std::array<std::size_t, 2>& d) {
            return sc::conv2d_rsck(in, k, s, p, d);
        },
        [](std::size_t c, std::size_t r, std::size_t s, std::size_t k) -> std::array<std::size_t, 4> {
            return {r, s, c, k}; // RSCK
        }
    },
    {
        "conv2d_pipeline",
        [](const sc::Tensor<float, 4>& in, const sc::Tensor<float, 4>& k,
            const std::array<std::size_t, 2>& s, std::variant<sc::padding_t, sc::Padding> p,
            const std::array<std::size_t, 2>& d) {
            return sc::conv2d_pipeline(in, k, s, p, d);
        },
        [](std::size_t c, std::size_t r, std::size_t s, std::size_t k) -> std::array<std::size_t, 4> {
            return {c, r, s, k}; // CRSK
        }
    },
    {
        "conv2d_xnnpack",
        [](const sc::Tensor<float, 4>& in, const sc::Tensor<float, 4>& k,
            const std::array<std::size_t, 2>& s, std::variant<sc::padding_t, sc::Padding> p,
            const std::array<std::size_t, 2>& d) {
            return sc::conv2d_xnnpack(in, k, s, p, d);
        },
        [](std::size_t c, std::size_t r, std::size_t s, std::size_t k) -> std::array<std::size_t, 4> {
            return {k, r, s, c}; // KRSC
        }
    },
};

void bench(
    const std::vector<std::string>& use = BACKENDS
        | std::views::transform([](const Backend& b) { return b.name; })
        | std::ranges::to<std::vector<std::string>>()
) {
    std::vector<Backend> backends;
    for (const auto& name : use) {
        auto it = std::find_if(BACKENDS.begin(), BACKENDS.end(),
            [&](const Backend& b) { return b.name == name; });
        if (it != BACKENDS.end()) {
            backends.push_back(*it);
        }
    }
    // Benchmark 1: varying input size × kernel size (channels=3, stride=1)
    {
        std::println("\n=== Input Size x Kernel Size (channels=3, stride=1, N=16) ===");
        std::vector<BenchmarkResult> results;
        std::vector<std::size_t> input_sizes = {16, 32, 64, 128};
        for (const auto& backend : backends) {
            std::println("Warming up {}...", backend.name);
            warmup(backend);
            for (auto input_sz : input_sizes) {
                std::vector<std::size_t> kernel_sizes = {1, 3, 5, input_sz / 2};
                for (auto kernel_sz : kernel_sizes) {
                    auto r = run_benchmark(backend, input_sz, kernel_sz, 3, 1);
                    std::println("  {:15s}  input={:3d}x{:3d}  kernel={:d}x{:d}  {:.3f} ms",
                        r.backend, r.input_size, r.input_size, r.kernel_size, r.kernel_size, r.total_ms);
                    results.push_back(r);
                }
            }
        }
        save_csv("input_kernel_benchmark_results.csv", results);
        std::println("Saved to input_kernel_benchmark_results.csv");
    }

    // Benchmark 2: varying channels × stride (input=64x64, kernel=3x3)
    {
        std::println("\n=== Channels x Stride (input=64x64, kernel=3x3, N=16) ===");
        std::vector<BenchmarkResult> results;
        std::vector<std::size_t> channel_sizes = {1, 3, 16, 64, 128};
        std::vector<std::size_t> stride_sizes = {1, 2, 4};
        for (const auto& backend : backends) {
            std::println("Warming up {}...", backend.name);
            warmup(backend);
            for (auto ch : channel_sizes) {
                for (auto st : stride_sizes) {
                    auto r = run_benchmark(backend, 64, 3, ch, st);
                    std::println("  {:15s}  channels={:3d}  stride={:d}  {:.3f} ms",
                        r.backend, r.channels, r.stride, r.total_ms);
                    results.push_back(r);
                }
            }
        }
        save_csv("channel_stride_benchmark_results.csv", results);
        std::println("Saved to channel_stride_benchmark_results.csv");
    }
}

int main() {
    // bench({"conv2d_implicit_gemm", "conv2d_explicit_gemm"});
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    warmup(BACKENDS[1]);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    return 0;
}
