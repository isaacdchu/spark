#ifndef UTIL_HPP
#define UTIL_HPP

#include <array>
#include <stdexcept>
#include <limits>
#include <iostream>
#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdlib>
#include <string_view>

namespace sc {

// Number of threads used by the conv2d backends. Read once from the
// SCONE_NUM_THREADS environment variable; falls back to 8 when unset,
// empty, or not a positive integer.
inline std::size_t num_threads() {
    static const std::size_t value = []() -> std::size_t {
        constexpr std::size_t default_threads = 8;
        const char* env = std::getenv("SCONE_NUM_THREADS");
        if (env == nullptr) {
            return default_threads;
        }
        std::string_view sv(env);
        std::size_t parsed = 0;
        auto [ptr, ec] = std::from_chars(sv.data(), sv.data() + sv.size(), parsed);
        if (ec != std::errc{} || ptr != sv.data() + sv.size() || parsed == 0) {
            std::cerr << "[util.hpp][num_threads] Ignoring invalid SCONE_NUM_THREADS=\""
                      << sv << "\", using " << default_threads << "\n";
            return default_threads;
        }
        return parsed;
    }();
    return value;
}

enum class PaddingEnum {
    NONE,
    VALID,
    SAME
};

struct padding_t {
    std::size_t PADH;
    std::size_t PADW;
    operator std::array<std::size_t, 2>() const {
        return {PADH, PADW};
    }
    padding_t(const std::array<std::size_t, 2>& pad) : PADH(pad[0]), PADW(pad[1]) {}
};

class Padding {
private:
    const PaddingEnum enum_value;
    const std::array<std::size_t, 2> array_value;
public:
    const bool is_enum;
    Padding() : enum_value(PaddingEnum::NONE), array_value({0, 0}), is_enum(true) {}
    Padding(PaddingEnum value) : enum_value(value), array_value({0, 0}), is_enum(true) {}
    Padding(const std::array<std::size_t, 2>& value) : enum_value(PaddingEnum::NONE), array_value(value), is_enum(false) {}
    Padding(std::size_t P_H, std::size_t P_W) : enum_value(PaddingEnum::NONE), array_value({P_H, P_W}), is_enum(false) {}
    operator PaddingEnum() const {
        if (!is_enum) {
            throw std::runtime_error("[util.hpp][Padding] is not an enum");
        }
        return enum_value;
    }
    operator std::array<std::size_t, 2>() const {
        if (is_enum) {
            throw std::runtime_error("[util.hpp][Padding] is not an array");
        }
        return array_value;
    }
};

std::array<std::size_t, 2> calculate_padding(
    const std::array<std::size_t, 4>& input_shape,
    const std::array<std::size_t, 4>& kernel_shape,
    const std::array<std::size_t, 2>& stride,
    const std::array<std::size_t, 2>& dilation
) {
    // xHWx/xRSx
    std::array<std::size_t, 2> padding;
    const std::ptrdiff_t in_h = static_cast<std::ptrdiff_t>(input_shape[1]);
    const std::ptrdiff_t in_w = static_cast<std::ptrdiff_t>(input_shape[2]);
    const std::ptrdiff_t s_h = static_cast<std::ptrdiff_t>(stride[0]);
    const std::ptrdiff_t s_w = static_cast<std::ptrdiff_t>(stride[1]);
    const std::ptrdiff_t k_h = static_cast<std::ptrdiff_t>(kernel_shape[1]);
    const std::ptrdiff_t k_w = static_cast<std::ptrdiff_t>(kernel_shape[2]);
    const std::ptrdiff_t d_h = static_cast<std::ptrdiff_t>(dilation[0]);
    const std::ptrdiff_t d_w = static_cast<std::ptrdiff_t>(dilation[1]);

    std::ptrdiff_t pad_total_h = (in_h - 1) * s_h + d_h * (k_h - 1) + 1 - in_h;
    if (pad_total_h < 0) pad_total_h = 0;
    if (pad_total_h % 2 == 1) {
        throw std::invalid_argument("[util.hpp][calculate_padding] Padding height is odd");
    }
    padding[0] = static_cast<std::size_t>(pad_total_h / 2);

    std::ptrdiff_t pad_total_w = (in_w - 1) * s_w + d_w * (k_w - 1) + 1 - in_w;
    if (pad_total_w < 0) pad_total_w = 0;
    if (pad_total_w % 2 == 1) {
        throw std::invalid_argument("[util.hpp][calculate_padding] Padding width is odd");
    }
    padding[1] = static_cast<std::size_t>(pad_total_w / 2);
    return padding;
}

std::array<std::size_t, 2> handle_padding(
    const std::array<std::size_t, 4>& input_shape,
    const std::array<std::size_t, 4>& kernel_shape,
    const std::array<std::size_t, 2>& stride,
    sc::Padding padding,
    const std::array<std::size_t, 2>& dilation
) {
    std::array<std::size_t, 2> pad = {0, 0};
    if (padding.is_enum) {
        switch (static_cast<PaddingEnum>(padding)) {
            case PaddingEnum::NONE:
            case PaddingEnum::VALID:
                pad = {0, 0};
                break;
            case PaddingEnum::SAME:
                pad = calculate_padding(input_shape, kernel_shape, stride, dilation);
                break;
            default:
                throw std::invalid_argument("[util.hpp][handle_padding] Unsupported padding enum value");
        }
    } else {
        pad = static_cast<std::array<std::size_t, 2>>(padding);
    }
    return pad;
}

std::array<std::size_t, 4> calculate_output_shape(
    const std::array<std::size_t, 4>& input_shape,
    const std::array<std::size_t, 4>& kernel_shape,
    const std::array<std::size_t, 2>& stride,
    const std::array<std::size_t, 2>& padding,
    const std::array<std::size_t, 2>& dilation
) {
    // NPQK
    return {
        input_shape[0],
        (input_shape[1] + 2 * padding[0] - dilation[0] * (kernel_shape[1] - 1) - 1 + stride[0]) / stride[0],
        (input_shape[2] + 2 * padding[1] - dilation[1] * (kernel_shape[2] - 1) - 1 + stride[1]) / stride[1],
        kernel_shape[3]
    };
}

} // namespace sc

#endif // UTIL_HPP