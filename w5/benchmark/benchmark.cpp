#include <print>
#include <string>
#include <chrono>
#include <vector>
#include <array>
#include <ranges>
#include <fstream>
#include <functional>
#include <thread>
#include <cstdio>
#include <memory>

#include <taskflow.hpp>

#include <scone.hpp>

struct BenchmarkResult {
    std::string backend;
    std::size_t batch_size;
    std::size_t input_size;
    std::size_t kernel_size;
    std::size_t input_channels;
    std::size_t output_channels;
    std::size_t stride;
    std::size_t padding;
    std::size_t dilation;
    long long int total_us;
};

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

std::vector<Backend> BACKENDS = {
    {"conv2d_implicit_gemm_krsc", sc::conv2d_implicit_gemm_krsc<float>, sc::crsk_to_krsc<float>},
    {"conv2d_explicit_gemm_crsk", sc::conv2d_explicit_gemm_crsk<float>},
    {"conv2d_pipeline", sc::conv2d_pipeline<float>},
    {"conv2d_padding_separate_krsc", sc::conv2d_padding_separate_krsc<float>, sc::crsk_to_krsc<float>},
    {"conv2d_simd_rsck", sc::conv2d_simd_rsck<float>, sc::crsk_to_rsck<float>},
    {"conv2d_simd_krsc", sc::conv2d_simd_krsc<float>, sc::crsk_to_krsc<float>},
    {"conv2d_block_rsck", sc::conv2d_block_rsck<float>, sc::crsk_to_rsck<float>},
    {"conv2d_block_krsc", sc::conv2d_block_krsc<float>, sc::crsk_to_krsc<float>},
    {"conv2d_xnnpack", sc::conv2d_xnnpack<float>, sc::crsk_to_krsc<float>, xnnpack_prepare}
};

void save_csv(const std::string& path, const std::vector<BenchmarkResult>& results) {
    std::ofstream f(path, std::ios::trunc);
    if (!f) {
        throw std::runtime_error("[benchmark.cpp][save_csv] Failed to open file for writing");
    }
    f << "backend,batch_size,input_size,kernel_size,input_channels,output_channels,stride,padding,dilation,total_us\n";
    for (const auto& r : results) {
        f << std::format("{},{},{},{},{},{},{},{},{},{}\n",
            r.backend,
            r.batch_size,
            r.input_size,
            r.kernel_size,
            r.input_channels, r.output_channels,
            r.stride, r.padding, r.dilation,
            r.total_us
        );
    }
}

void print_progress_bar(std::size_t current, std::size_t total, const std::string& label) {
    constexpr int bar_width = 40;
    double progress = static_cast<double>(current) / static_cast<double>(total);
    int filled = static_cast<int>(progress * bar_width);
    std::string title = std::format("{}", label);
    title += std::string(title.size() < 25 ? 25 - title.size() : 0, ' ');
    std::print("\r{}[", title);
    for (int i = 0; i < bar_width; i++) {
        std::print("{}", i < filled ? '#' : '-');
    }
    std::print("] {:3}% ({}/{})", static_cast<int>(progress * 100), current, total);
    std::fflush(stdout);
}

std::chrono::microseconds warmup(const Backend& backend, std::size_t iters = 5) {
    auto kernel_shape = std::array<std::size_t, 4>{64, 3, 3, 64};
    std::chrono::microseconds total{0};
    for (std::size_t i = 0; i < iters; i++) {
        sc::Tensor<float, 4> input(std::array<std::size_t, 4>{16, 64, 64, 64});
        for (std::size_t j = 0; j < input.size(); j++) {
            input.values()[j] = static_cast<float>(j % 127);
        }
        sc::Tensor<float, 4> kernel = backend.crsk_to_fn(sc::Tensor<float, 4>::ones(kernel_shape));
        auto start = std::chrono::high_resolution_clock::now();
        backend.fn(input, kernel, {1, 1}, sc::Padding(), {1, 1});
        auto end = std::chrono::high_resolution_clock::now();
        total += std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    }
    return total;
}

BenchmarkResult run_benchmark(
    const Backend& backend,
    const std::array<std::size_t, 4>& input_shape,
    const std::array<std::size_t, 4>& kernel_shape,
    std::size_t stride = 1,
    std::size_t padding = 0,
    std::size_t dilation = 1,
    int iters = 10
) {
    // NHWC/CRSK format. Build the kernel (and its layout repack) once, outside
    // the timed region, matching how a caller would reuse a prepared operator.
    const sc::Tensor<float, 4> kernel = backend.crsk_to_fn(sc::Tensor<float, 4>::ones(kernel_shape));
    const sc::Padding pad(padding, padding);
    // Build the runner once, outside the timer. For backends with a `prepare`
    // hook (XNNPACK) this performs create + reshape + weight packing here; for
    // the rest the runner is a thin wrapper over the pure-compute `fn`.
    RunFn run = backend.prepare
        ? backend.prepare(kernel, input_shape, {stride, stride}, pad, {dilation, dilation})
        : [&](const sc::Tensor<float, 4>& input) -> sc::Tensor<float, 4> {
              return backend.fn(input, kernel, {stride, stride}, pad, {dilation, dilation});
          };
    auto make_input = [&]() -> sc::Tensor<float, 4> {
        sc::Tensor<float, 4> input(input_shape);
        for (std::size_t j = 0; j < input.size(); j++) {
            input.values()[j] = static_cast<float>(j % 127);
        }
        return input;
    };
    // Priming run (untimed) to warm instruction/data caches uniformly.
    { const sc::Tensor<float, 4> input = make_input(); run(input); }
    std::chrono::microseconds total{0};
    for (int i = 0; i < iters; i++) {
        const sc::Tensor<float, 4> input = make_input();
        auto start = std::chrono::high_resolution_clock::now();
        sc::Tensor<float, 4> output = run(input);
        auto end = std::chrono::high_resolution_clock::now();
        total += std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    }
    total /= iters;

    const auto [N, H, W, C] = input_shape;
    const auto [C_, R, S, K] = kernel_shape;

    return BenchmarkResult{
        .backend = backend.name,
        .batch_size = N,
        .input_size = H,
        .kernel_size = R,
        .input_channels = C,
        .output_channels = K,
        .stride = stride,
        .padding = padding,
        .dilation = dilation,
        .total_us = total.count(),
    };
}

void benchmark(
    const std::vector<std::string>& use = BACKENDS
        | std::views::transform([](const Backend& b) { return b.name; })
        | std::ranges::to<std::vector<std::string>>()
) {
    std::vector<Backend> backends;
    for (const auto& name : use) {
        auto it = std::find_if(
            BACKENDS.begin(), BACKENDS.end(),
            [&](const Backend& b) -> bool {
                return b.name == name;
            }
        );
        if (it != BACKENDS.end()) {
            backends.push_back(*it);
        }
    }
    if (backends.size() != use.size()) {
        throw std::invalid_argument("[benchmark.cpp][benchmark] Some backends in `use` are not recognized");
    }
    std::vector<BenchmarkResult> results;
    const std::vector<std::size_t> batch_sizes = {16, 32};
    const std::vector<std::size_t> input_sizes = {32, 64, 128};
    const std::vector<std::size_t> kernel_sizes = {1, 3, 5};
    const std::vector<std::size_t> input_channels = {1, 16, 64};
    const std::vector<std::size_t> output_channels = {1, 16, 64};
    const std::vector<std::size_t> stride_sizes = {1, 2, 4};
    const std::vector<std::size_t> padding_sizes = {0, 1, 2};
    const std::size_t n_runs = (
        batch_sizes.size() *
        input_sizes.size() *
        kernel_sizes.size() *
        input_channels.size() *
        output_channels.size() *
        stride_sizes.size() *
        padding_sizes.size()
    );
    for (const auto& backend : backends) {
        warmup(backend);
        std::size_t completed = 0;
        print_progress_bar(completed, n_runs, backend.name);
        for (
            auto&& [
                batch_size, input_size, kernel_size, in_ch, out_ch, stride, padding
            ] : std::views::cartesian_product(
                batch_sizes, input_sizes, kernel_sizes, input_channels, output_channels, stride_sizes, padding_sizes
            )
        ) {
            if (kernel_size == 1 && padding > 0) {
                // Skip invalid cases where padding is applied to a 1x1 kernel
                print_progress_bar(++completed, n_runs, backend.name);
                continue;
            }
            std::array<std::size_t, 4> input_shape = {batch_size, input_size, input_size, in_ch};
            std::array<std::size_t, 4> kernel_shape = {in_ch, kernel_size, kernel_size, out_ch};
            auto r = run_benchmark(backend, input_shape, kernel_shape, stride, padding, 1);
            results.push_back(r);
            print_progress_bar(++completed, n_runs, backend.name);
        }
        std::println();
    }
    save_csv("benchmark/benchmark_results.csv", results);
}

int main() {
    const std::vector<std::string> use = {
        "conv2d_implicit_gemm_krsc",
        "conv2d_explicit_gemm_crsk",
        // "conv2d_block_rsck",
        // "conv2d_block_krsc",
        "conv2d_simd_rsck",
        "conv2d_simd_krsc",
        "conv2d_xnnpack",
    };
    std::println("Running benchmarks for backends: {}", use);
    benchmark(use);
    std::println("Benchmark results saved to benchmark/benchmark_results.csv");
    return 0;
}