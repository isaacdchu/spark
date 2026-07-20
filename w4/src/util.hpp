#ifndef UTIL_HPP
#define UTIL_HPP

#include <array>
#include <stdexcept>
#include <limits>
#include <iostream>
#include <algorithm>
#include <mutex>

namespace sc {

enum class Padding {
    VALID,
    NONE,
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
        throw std::invalid_argument("Padding height is odd");
    }
    padding[0] = static_cast<std::size_t>(pad_total_h / 2);

    std::ptrdiff_t pad_total_w = (in_w - 1) * s_w + d_w * (k_w - 1) + 1 - in_w;
    if (pad_total_w < 0) pad_total_w = 0;
    if (pad_total_w % 2 == 1) {
        throw std::invalid_argument("Padding width is odd");
    }
    padding[1] = static_cast<std::size_t>(pad_total_w / 2);
    return padding;
}

}

#endif // UTIL_HPP