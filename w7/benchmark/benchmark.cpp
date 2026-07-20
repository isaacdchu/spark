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
#include <cstdlib>
#include <cctype>
#include <memory>
#include <optional>
#include <stdexcept>

#include <taskflow.hpp>

#include <scone.hpp>

// Split a comma-separated string into trimmed, non-empty tokens.
std::vector<std::string> split_csv_tokens(const std::string& value) {
    std::vector<std::string> tokens;
    std::size_t start = 0;
    while (start <= value.size()) {
        std::size_t comma = value.find(',', start);
        std::size_t end = (comma == std::string::npos) ? value.size() : comma;
        std::size_t a = start;
        std::size_t b = end;
        while (a < b && std::isspace(static_cast<unsigned char>(value[a]))) a++;
        while (b > a && std::isspace(static_cast<unsigned char>(value[b - 1]))) b--;
        if (b > a) {
            tokens.push_back(value.substr(a, b - a));
        }
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return tokens;
}

// Read a comma-separated list of unsigned ints from env var `name`. Returns
// `fallback` when the var is unset or empty. Throws std::invalid_argument
// (naming the var) on a non-numeric token.
std::vector<std::size_t> parse_env_sizes(const char* name, std::vector<std::size_t> fallback) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') {
        return fallback;
    }
    const std::vector<std::string> tokens = split_csv_tokens(raw);
    if (tokens.empty()) {
        return fallback;
    }
    std::vector<std::size_t> out;
    for (const auto& tok : tokens) {
        try {
            std::size_t pos = 0;
            unsigned long long v = std::stoull(tok, &pos);
            if (pos != tok.size()) {
                throw std::invalid_argument("trailing characters");
            }
            out.push_back(static_cast<std::size_t>(v));
        } catch (const std::exception&) {
            throw std::invalid_argument(
                std::format("[benchmark.cpp][parse_env_sizes] {}: invalid unsigned int token \"{}\"", name, tok)
            );
        }
    }
    return out;
}

// Read a comma-separated list of backend names from env var `name`. Returns
// `fallback` when the var is unset or empty.
std::vector<std::string> parse_env_names(const char* name, std::vector<std::string> fallback) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') {
        return fallback;
    }
    std::vector<std::string> tokens = split_csv_tokens(raw);
    if (tokens.empty()) {
        return fallback;
    }
    return tokens;
}

struct BenchmarkResult {
    std::string backend;
    std::size_t num_threads;
    // Active register-tile dims for block backends; empty for all others.
    std::optional<std::size_t> block_q;
    std::optional<std::size_t> block_k;
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
    // Active register-tile dims {block_q, block_k} for the blocked backends;
    // empty for every non-block backend (its CSV block columns are left blank).
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

void save_csv(const std::string& path, const std::vector<BenchmarkResult>& results, bool append) {
    // In append mode, open at end and skip the header (assumed already present);
    // otherwise truncate and write a fresh header.
    std::ofstream f(path, append ? std::ios::app : std::ios::trunc);
    if (!f) {
        throw std::runtime_error("[benchmark.cpp][save_csv] Failed to open file for writing");
    }
    if (!append) {
        f << 
            "backend,num_threads,block_q,block_k,batch_size,"
            "input_size,kernel_size,input_channels,output_channels,"
            "stride,padding,dilation,total_us\n";
    }
    for (const auto& r : results) {
        const std::string block_q = r.block_q ? std::format("{}", *r.block_q) : std::string();
        const std::string block_k = r.block_k ? std::format("{}", *r.block_k) : std::string();
        f << std::format("{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
            r.backend,
            r.num_threads,
            block_q, block_k,
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
    {
        const sc::Tensor<float, 4> input = make_input();
        run(input);
    }
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
        .num_threads = sc::num_threads(),
        .block_q = backend.block ? std::optional<std::size_t>((*backend.block)[0]) : std::nullopt,
        .block_k = backend.block ? std::optional<std::size_t>((*backend.block)[1]) : std::nullopt,
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
    // Sweep dimensions default to the historical hardcoded values but can be
    // overridden by the dashboard via comma-separated env vars.
    const std::vector<std::size_t> batch_sizes = parse_env_sizes("SCONE_BENCH_BATCH_SIZES", {16, 32});
    const std::vector<std::size_t> input_sizes = parse_env_sizes("SCONE_BENCH_INPUT_SIZES", {32, 64});
    const std::vector<std::size_t> kernel_sizes = parse_env_sizes("SCONE_BENCH_KERNEL_SIZES", {3, 5});
    const std::vector<std::size_t> input_channels = parse_env_sizes("SCONE_BENCH_INPUT_CHANNELS", {1, 16, 32, 64});
    const std::vector<std::size_t> output_channels = parse_env_sizes("SCONE_BENCH_OUTPUT_CHANNELS", {1, 16, 32, 64});
    const std::vector<std::size_t> stride_sizes = parse_env_sizes("SCONE_BENCH_STRIDES", {1});
    const std::vector<std::size_t> padding_sizes = parse_env_sizes("SCONE_BENCH_PADDINGS", {0});
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
    // Output path and append behavior are configurable so a launcher can run this
    // binary many times (once per block-config / thread-count) and aggregate every
    // run into one master CSV.
    const char* out_env = std::getenv("SCONE_BENCH_OUTPUT");
    const std::string output_path = out_env ? out_env : "benchmark/benchmark_results.csv";
    const char* append_env = std::getenv("SCONE_BENCH_APPEND");
    const bool append = append_env && std::string(append_env) == "1";
    save_csv(output_path, results, append);
}

int main() {
    std::println("Running convolution benchmarks on {} threads", sc::num_threads());
    // Default backend list; overridable via SCONE_BENCH_BACKENDS (comma-separated).
    // Unknown names still error inside benchmark() as before.
    const std::vector<std::string> use = parse_env_names("SCONE_BENCH_BACKENDS", {
        "conv2d_implicit_gemm_krsc",
        "conv2d_explicit_gemm_crsk",
        "conv2d_block_rsck",
        "conv2d_ind_rsck",
        "conv2d_simd_rsck",
        // "conv2d_simd_krsc",
        "conv2d_xnnpack",
    });
    std::println("Running benchmarks for backends: {}", use);
    benchmark(use);
    const char* out_env = std::getenv("SCONE_BENCH_OUTPUT");
    std::println("Benchmark results saved to {}", out_env ? out_env : "benchmark/benchmark_results.csv");
    return 0;
}