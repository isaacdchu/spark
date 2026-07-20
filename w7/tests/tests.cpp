#include <array>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <memory>
#include <cmath>

#include <scone.hpp>

class TestCase {
public:
    std::size_t idx;
    std::array<std::size_t, 4> input_shape;
    std::vector<float> input_values;
    std::array<std::size_t, 4> kernel_shape;
    std::vector<float> kernel_values;
    std::array<std::size_t, 2> stride;
    std::array<std::size_t, 2> padding;
    std::array<std::size_t, 2> dilation;
    TestCase(
        const std::array<std::size_t, 4>& input_shape,
        std::vector<float>&& input_values,
        const std::array<std::size_t, 4>& kernel_shape,
        std::vector<float>&& kernel_values,
        const std::array<std::size_t, 2>& stride,
        const std::array<std::size_t, 2>& padding,
        const std::array<std::size_t, 2>& dilation
    ) :
        input_shape(input_shape), input_values(input_values),
        kernel_shape(kernel_shape), kernel_values(kernel_values),
        stride(stride), padding(padding), dilation(dilation)
    {
        if (input_values.size() != input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]) {
            throw std::invalid_argument(
                "[tests.cpp][TestCase::TestCase] Input values size does not match input shape"
            );
        }
        if (kernel_values.size() != kernel_shape[0] * kernel_shape[1] * kernel_shape[2] * kernel_shape[3]) {
            throw std::invalid_argument(
                "[tests.cpp][TestCase::TestCase] Kernel values size does not match kernel shape"
            );
        }
        static std::size_t idx_ = 0;
        idx = ++idx_;
    }
    std::string to_string() const {
        std::ostringstream oss;
        oss << "Test " << idx << '\n';
        oss << "Input: ";
        for (const auto dim : input_shape) {
            oss << dim << ' ';
        }
        oss << "\nKernel: ";
        for (const auto dim : kernel_shape) {
            oss << dim << ' ';
        }
        oss << "\nStride: " << stride[0] << 'x' << stride[1];
        oss << "\nPadding: " << padding[0] << 'x' << padding[1];
        oss << "\nDilation: " << dilation[0] << 'x' << dilation[1];
        return oss.str();
    }
};

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

static std::string run_case(const TestCase& conv_case, const Backend& backend) {
    auto input = sc::Tensor<float, 4>(conv_case.input_shape);
    input.values() = conv_case.input_values;
    auto kernel = sc::Tensor<float, 4>(conv_case.kernel_shape);
    kernel.values() = conv_case.kernel_values;
    const auto output = backend.fn(
        input, backend.crsk_to_fn(kernel),
        conv_case.stride, sc::Padding(conv_case.padding), conv_case.dilation
    );
    std::ostringstream oss;
    oss << conv_case.to_string();
    oss << "\nOutput:\n" << output.to_string();
    oss << "\nInput:\n" << input.to_string();
    oss << "\nKernel:\n" << kernel.to_string();
    return oss.str();
}

static std::size_t product(const std::array<std::size_t, 4>& shape) {
    std::size_t result = 1;
    for (const auto dim : shape) {
        result *= dim;
    }
    return result;
}

TestCase make_test_case(
    const std::array<std::size_t, 4>& input_shape,
    float input_seed,
    const std::array<std::size_t, 4>& kernel_shape,
    float kernel_seed,
    const std::array<std::size_t, 2>& stride,
    const std::array<std::size_t, 2>& padding,
    const std::array<std::size_t, 2>& dilation
) {
    std::vector<float> input_values = std::vector<float>();
    for (std::size_t i = 0; i < product(input_shape); i++) {
        input_values.push_back(static_cast<float>(std::sin(static_cast<double>(input_seed) + static_cast<double>(i)) * 10.0));
    }
    std::vector<float> kernel_values = std::vector<float>();
    for (std::size_t i = 0; i < product(kernel_shape); i++) {
        kernel_values.push_back(static_cast<float>(std::cos(static_cast<double>(kernel_seed) + static_cast<double>(i)) * 10.0));
    }
    return TestCase(
        input_shape, std::move(input_values),
        kernel_shape, std::move(kernel_values),
        stride, padding, dilation
    );
}

std::vector<TestCase> tests = {
    make_test_case({1, 4, 4, 1}, 0.0f, {1, 3, 3, 2}, 0.0f, {1, 1}, {0, 0}, {1, 1}),
    make_test_case({1, 8, 8, 1}, 1.0f, {1, 3, 3, 2}, 3.0f, {1, 1}, {0, 0}, {1, 1}),
    make_test_case({4, 8, 8, 1}, 2.0f, {1, 3, 3, 2}, 6.0f, {2, 2}, {1, 1}, {1, 1}),
    make_test_case({1, 8, 8, 2}, 2.0f, {2, 3, 3, 2}, 6.0f, {1, 1}, {1, 1}, {1, 1}),
    make_test_case({4, 16, 16, 3}, 3.0f, {3, 3, 3, 2}, 9.0f, {1, 2}, {1, 2}, {1, 1}),
    make_test_case({4, 16, 16, 4}, 3.0f, {4, 3, 3, 8}, 9.0f, {1, 1}, {1, 1}, {2, 2}),
    make_test_case({4, 16, 16, 4}, 4.0f, {4, 3, 3, 8}, 10.0f, {1, 1}, {1, 1}, {2, 1}),
    make_test_case({4, 16, 16, 4}, 5.0f, {4, 1, 2, 8}, 11.0f, {1, 1}, {2, 1}, {1, 2}),
    make_test_case({4, 16, 16, 4}, 5.0f, {4, 2, 3, 8}, 11.0f, {1, 1}, {2, 1}, {1, 2}),
    make_test_case({1, 5, 7, 4}, 6.0f, {4, 1, 1, 6}, 12.0f, {1, 1}, {0, 0}, {1, 1}),
    make_test_case({2, 10, 6, 2}, 7.0f, {2, 5, 1, 3}, 13.0f, {1, 1}, {2, 0}, {1, 1}),
    make_test_case({2, 6, 10, 2}, 8.0f, {2, 1, 5, 3}, 14.0f, {1, 1}, {0, 2}, {1, 1}),
    make_test_case({1, 15, 15, 3}, 9.0f, {3, 3, 3, 4}, 15.0f, {3, 3}, {1, 1}, {1, 1}),
    make_test_case({1, 12, 12, 2}, 10.0f, {2, 3, 3, 4}, 16.0f, {1, 1}, {0, 0}, {3, 3}),
    make_test_case({2, 9, 9, 3}, 11.0f, {3, 5, 5, 4}, 17.0f, {1, 1}, {2, 2}, {1, 1}),
    make_test_case({2, 14, 10, 3}, 12.0f, {3, 3, 2, 5}, 18.0f, {2, 3}, {1, 0}, {2, 1}),
    make_test_case({1, 8, 8, 8}, 13.0f, {8, 3, 3, 8}, 19.0f, {1, 1}, {1, 1}, {1, 1}),
    make_test_case({2, 7, 7, 5}, 14.0f, {5, 2, 2, 3}, 20.0f, {2, 2}, {0, 0}, {1, 1}),
    make_test_case({1, 3, 3, 2}, 15.0f, {2, 3, 3, 4}, 21.0f, {1, 1}, {2, 2}, {1, 1}),
    make_test_case({1, 1, 1, 3}, 16.0f, {3, 1, 1, 4}, 22.0f, {1, 1}, {0, 0}, {1, 1}),
    make_test_case({1, 20, 20, 3}, 17.0f, {3, 7, 7, 4}, 23.0f, {2, 2}, {3, 3}, {2, 2}),
    make_test_case({8, 6, 6, 2}, 18.0f, {2, 3, 3, 4}, 24.0f, {1, 1}, {1, 1}, {1, 1})
};

std::vector<Backend> backends = {
    {"conv2d_implicit_gemm_krsc", sc::conv2d_implicit_gemm_krsc<float>, sc::crsk_to_krsc<float>},
    {"conv2d_explicit_gemm_crsk", sc::conv2d_explicit_gemm_crsk<float>},
    {"conv2d_pipeline", sc::conv2d_pipeline<float>},
    {"conv2d_padding_separate_krsc", sc::conv2d_padding_separate_krsc<float>, sc::crsk_to_krsc<float>},
    {"conv2d_simd_rskc", sc::conv2d_simd_rsck<float>, sc::crsk_to_rsck<float>},
    {"conv2d_simd_krsc", sc::conv2d_simd_krsc<float>, sc::crsk_to_krsc<float>},
    {"conv2d_block_rsck", sc::conv2d_block_rsck<float>, sc::crsk_to_rsck<float>},
    {"conv2d_ind_rsck", sc::conv2d_ind_rsck<float>, sc::crsk_to_rsck<float>},
    {"conv2d_xnnpack", sc::conv2d_xnnpack<float>, sc::crsk_to_krsc<float>}
};

int main(int argc, char* argv[]) {
    if (argc != 2) {
        throw std::invalid_argument("Invalid number of arguments");
    }
    const std::filesystem::path output_file_name = argv[1];
    std::ofstream file(output_file_name);
    for (const auto& backend : backends) {
        file << "Backend: " << backend.name << "\n";
        for (const auto& test : tests) {
            file << run_case(test, backend);
            file << "\n";
        }
    }
    return 0;
}