#ifndef UTIL_HPP
#define UTIL_HPP

#include <array>

namespace sc {

std::array<std::size_t, 4> calculate_output_shape(
    const std::array<std::size_t, 4>& input_shape,
    const std::array<std::size_t, 4>& kernel_shape,
    const std::array<std::size_t, 2>& stride,
    const std::array<std::size_t, 2>& padding,
    const std::array<std::size_t, 2>& dilation
) {
    return {
        input_shape[0],
        kernel_shape[0],
        (input_shape[2] + 2 * padding[0] - dilation[0] * (kernel_shape[2] - 1) - 1 + stride[0]) / stride[0],
        (input_shape[3] + 2 * padding[1] - dilation[1] * (kernel_shape[3] - 1) - 1 + stride[1]) / stride[1]
    };
}

}

#endif // UTIL_HPP