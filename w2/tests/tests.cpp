#include <vector>
#include <array>
#include <string>
#include <functional>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include <omp.h>

#include <scone.hpp>
#include <sstream>
#include <iomanip>

static const std::vector<std::function<std::string()>> tests = {
    []() -> std::string {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(6);
        const int N = 1, C = 1, H = 4, W = 4;
        auto input = sc::Tensor<float, 4>(N, C, H, W);
        for (std::size_t i = 0; i < input.size(); i++) {
            input.values()[i] = static_cast<float>(i + 1);
        }
        auto kernel = sc::Tensor<float, 4>(1, 1, 1, 1);
        kernel.values()[0] = 1.0f;
        auto output = sc::conv2d(input, kernel, sc::Padding::NONE);

        oss << "INPUT_SHAPE: " << N << "," << C << "," << H << "," << W << "\n";
        oss << "INPUT_VALUES: ";
        for (std::size_t i = 0; i < input.size(); i++) {
            if (i) oss << ",";
            oss << input.values()[i];
        }
        oss << "\n";

        oss << "KERNEL_SHAPE: 1,1,1,1\n";
        oss << "KERNEL_VALUES: ";
        for (std::size_t i = 0; i < kernel.size(); i++) {
            if (i) oss << ",";
            oss << kernel.values()[i];
        }
        oss << "\n";

        oss << "PADDING: 0\n";
        oss << "STRIDE: 1,1\n";
        oss << "DILATION: 1,1\n";
        auto s = output.shape();
        oss << "OUTPUT_SHAPE: " << s[0] << "," << s[1] << "," << s[2] << "," << s[3] << "\n";
        oss << "OUTPUT_VALUES: ";
        for (std::size_t i = 0; i < output.size(); i++) {
            if (i) oss << ",";
            oss << output.values()[i];
        }
        return oss.str();
    },
    []() -> std::string {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(6);
        const int N = 1, C = 1, H = 5, W = 5;
        auto input = sc::Tensor<float, 4>(N, C, H, W);
        for (std::size_t i = 0; i < input.size(); i++) {
            input.values()[i] = static_cast<float>(i + 1);
        }
        const int kH = 3, kW = 3;
        auto kernel = sc::Tensor<float, 4>(1, 1, kH, kW);
        for (std::size_t i = 0; i < kernel.size(); i++) {
            kernel.values()[i] = static_cast<float>(i + 1);
        }
        auto output = sc::conv2d(input, kernel, sc::Padding::NONE);

        oss << "INPUT_SHAPE: " << N << "," << C << "," << H << "," << W << "\n";
        oss << "INPUT_VALUES: ";
        for (std::size_t i = 0; i < input.size(); i++) {
            if (i) oss << ",";
            oss << input.values()[i];
        }
        oss << "\n";

        oss << "KERNEL_SHAPE: " << 1 << "," << 1 << "," << kH << "," << kW << "\n";
        oss << "KERNEL_VALUES: ";
        for (std::size_t i = 0; i < kernel.size(); i++) {
            if (i) oss << ",";
            oss << kernel.values()[i];
        }
        oss << "\n";

        oss << "PADDING: 0\n";
        oss << "STRIDE: 1,1\n";
        oss << "DILATION: 1,1\n";
        auto s = output.shape();
        oss << "OUTPUT_SHAPE: " << s[0] << "," << s[1] << "," << s[2] << "," << s[3] << "\n";
        oss << "OUTPUT_VALUES: ";
        for (std::size_t i = 0; i < output.size(); i++) {
            if (i) oss << ",";
            oss << output.values()[i];
        }
        return oss.str();
    },
    []() -> std::string {
        // Stride 2, VALID padding
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(6);
        const int N = 1, C = 1, H = 6, W = 6;
        auto input = sc::Tensor<float, 4>(N, C, H, W);
        for (std::size_t i = 0; i < input.size(); i++) input.values()[i] = static_cast<float>(i + 1);
        const int kH = 3, kW = 3;
        auto kernel = sc::Tensor<float, 4>(1, 1, kH, kW);
        for (std::size_t i = 0; i < kernel.size(); i++) kernel.values()[i] = static_cast<float>(i + 1);
        auto output = sc::conv2d(input, kernel, sc::Padding::NONE, {2,2}, {1,1});

        oss << "INPUT_SHAPE: " << N << "," << C << "," << H << "," << W << "\n";
        oss << "INPUT_VALUES: ";
        for (std::size_t i = 0; i < input.size(); i++) { if (i) oss << ","; oss << input.values()[i]; }
        oss << "\n";
        oss << "KERNEL_SHAPE: " << 1 << "," << 1 << "," << kH << "," << kW << "\n";
        oss << "KERNEL_VALUES: ";
        for (std::size_t i = 0; i < kernel.size(); i++) { if (i) oss << ","; oss << kernel.values()[i]; }
        oss << "\n";
        oss << "PADDING: 0\n";
        oss << "STRIDE: 2,2\n";
        oss << "DILATION: 1,1\n";
        auto s = output.shape();
        oss << "OUTPUT_SHAPE: " << s[0] << "," << s[1] << "," << s[2] << "," << s[3] << "\n";
        oss << "OUTPUT_VALUES: ";
        for (std::size_t i = 0; i < output.size(); i++) { if (i) oss << ","; oss << output.values()[i]; }
        return oss.str();
    },
    []() -> std::string {
        // SAME padding, stride 1
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(6);
        const int N = 1, C = 1, H = 5, W = 5;
        auto input = sc::Tensor<float, 4>(N, C, H, W);
        for (std::size_t i = 0; i < input.size(); i++) input.values()[i] = static_cast<float>(i + 1);
        const int kH = 3, kW = 3;
        auto kernel = sc::Tensor<float, 4>(1, 1, kH, kW);
        for (std::size_t i = 0; i < kernel.size(); i++) kernel.values()[i] = static_cast<float>(i + 1);
        auto output = sc::conv2d(input, kernel, sc::Padding::SAME, {1,1}, {1,1});

        oss << "INPUT_SHAPE: " << N << "," << C << "," << H << "," << W << "\n";
        oss << "INPUT_VALUES: ";
        for (std::size_t i = 0; i < input.size(); i++) { if (i) oss << ","; oss << input.values()[i]; }
        oss << "\n";
        oss << "KERNEL_SHAPE: " << 1 << "," << 1 << "," << kH << "," << kW << "\n";
        oss << "KERNEL_VALUES: ";
        for (std::size_t i = 0; i < kernel.size(); i++) { if (i) oss << ","; oss << kernel.values()[i]; }
        oss << "\n";
        oss << "PADDING: SAME\n";
        oss << "STRIDE: 1,1\n";
        oss << "DILATION: 1,1\n";
        auto s = output.shape();
        oss << "OUTPUT_SHAPE: " << s[0] << "," << s[1] << "," << s[2] << "," << s[3] << "\n";
        oss << "OUTPUT_VALUES: ";
        for (std::size_t i = 0; i < output.size(); i++) { if (i) oss << ","; oss << output.values()[i]; }
        return oss.str();
    },
    []() -> std::string {
        // Dilation 2
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(6);
        const int N = 1, C = 1, H = 7, W = 7;
        auto input = sc::Tensor<float, 4>(N, C, H, W);
        for (std::size_t i = 0; i < input.size(); i++) input.values()[i] = static_cast<float>(i + 1);
        const int kH = 3, kW = 3;
        auto kernel = sc::Tensor<float, 4>(1, 1, kH, kW);
        for (std::size_t i = 0; i < kernel.size(); i++) kernel.values()[i] = static_cast<float>(i + 1);
        auto output = sc::conv2d(input, kernel, sc::Padding::NONE, {1,1}, {2,2});

        oss << "INPUT_SHAPE: " << N << "," << C << "," << H << "," << W << "\n";
        oss << "INPUT_VALUES: ";
        for (std::size_t i = 0; i < input.size(); i++) { if (i) oss << ","; oss << input.values()[i]; }
        oss << "\n";
        oss << "KERNEL_SHAPE: " << 1 << "," << 1 << "," << kH << "," << kW << "\n";
        oss << "KERNEL_VALUES: ";
        for (std::size_t i = 0; i < kernel.size(); i++) { if (i) oss << ","; oss << kernel.values()[i]; }
        oss << "\n";
        oss << "PADDING: 0\n";
        oss << "STRIDE: 1,1\n";
        oss << "DILATION: 2,2\n";
        auto s = output.shape();
        oss << "OUTPUT_SHAPE: " << s[0] << "," << s[1] << "," << s[2] << "," << s[3] << "\n";
        oss << "OUTPUT_VALUES: ";
        for (std::size_t i = 0; i < output.size(); i++) { if (i) oss << ","; oss << output.values()[i]; }
        return oss.str();
    },
    []() -> std::string {
        // Multi-channel in/out
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(6);
        const int N = 1, C = 2, H = 4, W = 4;
        const int outC = 2, inC = 2, kH = 3, kW = 3;
        auto input = sc::Tensor<float, 4>(N, C, H, W);
        for (std::size_t i = 0; i < input.size(); i++) input.values()[i] = static_cast<float>(i + 1);
        auto kernel = sc::Tensor<float, 4>(outC, inC, kH, kW);
        for (std::size_t i = 0; i < kernel.size(); i++) kernel.values()[i] = static_cast<float>((i % 9) + 1);
        auto output = sc::conv2d(input, kernel, sc::Padding::NONE, {1,1}, {1,1});

        oss << "INPUT_SHAPE: " << N << "," << C << "," << H << "," << W << "\n";
        oss << "INPUT_VALUES: ";
        for (std::size_t i = 0; i < input.size(); i++) { if (i) oss << ","; oss << input.values()[i]; }
        oss << "\n";
        oss << "KERNEL_SHAPE: " << outC << "," << inC << "," << kH << "," << kW << "\n";
        oss << "KERNEL_VALUES: ";
        for (std::size_t i = 0; i < kernel.size(); i++) { if (i) oss << ","; oss << kernel.values()[i]; }
        oss << "\n";
        oss << "PADDING: 0\n";
        oss << "STRIDE: 1,1\n";
        oss << "DILATION: 1,1\n";
        auto s = output.shape();
        oss << "OUTPUT_SHAPE: " << s[0] << "," << s[1] << "," << s[2] << "," << s[3] << "\n";
        oss << "OUTPUT_VALUES: ";
        for (std::size_t i = 0; i < output.size(); i++) { if (i) oss << ","; oss << output.values()[i]; }
        return oss.str();
    }
};

int main(int argc, char* argv[]) {
    if (argc != 2) {
        throw std::invalid_argument("Invalid number of arguments");
    }
    sc::IM2COL_BACKEND = sc::Im2ColBackend::TASKFLOW;
    std::string output_file_name = argv[1];
    auto output_file = std::filesystem::path(output_file_name);
    std::ofstream file(output_file);
    for (auto test : tests) {
        file << test() << std::endl;
    }
    return 0;
}