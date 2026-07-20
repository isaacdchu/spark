#include <print>
#include <string>
#include <chrono>
#include <vector>
#include <array>
#include <set>
#include <fstream>

#include <omp.h>

#include <scone.hpp>

struct BenchmarkResult {
    std::size_t input_size;
    std::size_t kernel_size;
    std::string backend;
    double total_ms;
    double im2col_ms;
    double matmul_ms;
    double col2im_ms;
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
        "total_ms,im2col_ms,matmul_ms,col2im_ms\n";
    for (const auto& r: results) {
        f << std::format(
            "{}," \
            "{},{}," \
            "{},{},{},{}\n",
            r.backend,
            r.input_size, r.kernel_size,
            r.total_ms, r.im2col_ms, r.matmul_ms, r.col2im_ms);
    }
}

void warmup() {
    auto input = sc::Tensor<float, 4>(16, 3, 256, 256);
    for (std::size_t i = 0; i < input.size(); i++) {
        input.values()[i] = static_cast<float>(i);
    }
    auto kernel = sc::Tensor<float, 4>::ones(3, 3, 5, 5);
    for (int i = 0; i < 5; i++) {
        sc::IM2COL_BACKEND = sc::Im2ColBackend::STANDARD;
        auto output = sc::conv2d(input, kernel, sc::Padding::NONE);
        sc::IM2COL_BACKEND = sc::Im2ColBackend::TASKFLOW;
        output = sc::conv2d(input, kernel, sc::Padding::NONE);
        sc::IM2COL_BACKEND = sc::Im2ColBackend::OPENMP;
        output = sc::conv2d(input, kernel, sc::Padding::NONE);
    }
}

void benchmark_one(sc::Tensor<float, 4>& input, sc::Tensor<float, 4>& kernel, int num_iterations, sc::Im2ColBackend backend) {
    sc::IM2COL_BACKEND = backend;
    sc::total_times = {};
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        auto output = sc::conv2d(input, kernel, sc::Padding::NONE);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    results.push_back(BenchmarkResult{
        .input_size = input.shape()[2],
        .kernel_size = kernel.shape()[2],
        .backend = im2col_backend_to_string(backend),
        .total_ms = static_cast<double>(duration.count()) / num_iterations,
        .im2col_ms = static_cast<double>(sc::total_times.im2col.count()) / 1000 / num_iterations,
        .matmul_ms = static_cast<double>(sc::total_times.matmul.count()) / 1000 / num_iterations,
        .col2im_ms = static_cast<double>(sc::total_times.col2im.count()) / 1000 / num_iterations
    });
}

void benchmark_conv2d(
    const std::array<std::size_t, 4>& input_shape,
    const std::array<std::size_t, 4>& kernel_shape,
    int num_iterations,
    sc::Im2ColBackend backend
) {
    auto input = sc::Tensor<float, 4>(input_shape);
    for (std::size_t i = 0; i < input.size(); i++) {
        input.values()[i] = static_cast<float>(i);
    }
    auto kernel = sc::Tensor<float, 4>::ones(kernel_shape);
    benchmark_one(input, kernel, num_iterations, backend);
}

void benchmark_all() {
    // Print number of workers for the Taskflow executor
    std::println("Number of Taskflow workers: {}", sc::executor.num_workers());
    // Print number of threads for OpenMP
    #pragma omp parallel
    {
        #pragma omp single
        std::println("Number of OpenMP threads: {}", omp_get_num_threads());
    }
    std::println("Warmup...");
    warmup();
    std::println("Starting benchmark...");
    std::vector<std::size_t> input_sizes = {16, 32, 64, 128};
    std::vector<sc::Im2ColBackend> backends = {sc::Im2ColBackend::TASKFLOW};
    for (auto backend : backends) {
        for (std::size_t input_size : input_sizes) {
            std::set<std::size_t> kernel_sizes = {1, 3, 5, input_size / 4, input_size / 2};
            for (std::size_t kernel_size : kernel_sizes) {
                std::println("Testing with input size: {} and kernel size: {}", input_size, kernel_size);
                benchmark_conv2d(
                    {16, 3, input_size, input_size},
                    {3, 3, kernel_size, kernel_size},
                    32,
                    backend
                );
            }
        }
    }
    std::println("Saving results...");
    save_csv("benchmark_results.csv", results);
    std::println("Benchmark complete.");
}

void large(sc::Im2ColBackend backend) {
    auto input = sc::Tensor<float, 4>(16, 3, 1920, 1080);
    for (std::size_t i = 0; i < input.size(); i++) {
        input.values()[i] = static_cast<float>(i);
    }
    auto kernel = sc::Tensor<float, 4>::ones(3, 3, 5, 5);
    sc::IM2COL_BACKEND = backend;
    sc::total_times = {};
    auto start = std::chrono::high_resolution_clock::now();
    auto output = sc::conv2d(input, kernel, 1, sc::Padding::VALID, 1);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::println("Time taken ({}): {} ms", sc::im2col_backend_to_string(sc::IM2COL_BACKEND), duration.count());
    std::println(
        "Time breakdown:\n\tIM2COL: {} ms\n\tMATMUL: {} ms\n\tCOL2IM: {} ms",
        sc::total_times.im2col.count() / 1000,
        sc::total_times.matmul.count() / 1000,
        sc::total_times.col2im.count() / 1000
    );
}

void profile_taskflow() {
    auto input = sc::Tensor<float, 4>(16, 3, 1920, 1080);
    for (std::size_t i = 0; i < input.size(); i++)
    {
        input.values()[i] = static_cast<float>(i);
    }
    auto kernel = sc::Tensor<float, 4>::ones(3, 3, 5, 5);
    sc::IM2COL_BACKEND = sc::Im2ColBackend::TASKFLOW;
    sc::total_times = {};
    auto start = std::chrono::high_resolution_clock::now();
    auto output = sc::conv2d(input, kernel, 1, sc::Padding::SAME, 1);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::println("Time taken (Taskflow): {} ms", duration.count());
    std::println(
        "Time breakdown:\n\tIM2COL: {} ms\n\tMATMUL: {} ms\n\tCOL2IM: {} ms",
        sc::total_times.im2col.count() / 1000,
        sc::total_times.matmul.count() / 1000,
        sc::total_times.col2im.count() / 1000
    );
}

int main() {
    benchmark_all();
    // warmup();
    // large(sc::Im2ColBackend::STANDARD);
    // large(sc::Im2ColBackend::TASKFLOW);
    // large(sc::Im2ColBackend::OPENMP);
    // profile_taskflow();
    return 0;
}