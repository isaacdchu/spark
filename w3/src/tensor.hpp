#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <array>
#include <concepts>
#include <functional>
#include <algorithm>
#include <string>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <functional>

#include "concepts.hpp"

namespace sc {

template <FloatLike T, SizeLike auto N> requires (N > 0)
class Tensor {
private:
    std::array<std::size_t, N> shape_;
    std::array<std::size_t, N> strides_;
    std::size_t size_;
    std::vector<T> values_;

public:
    Tensor(const std::array<std::size_t, N>& shape) :
        shape_(shape),
        strides_{},
        size_(std::reduce(shape_.begin(), shape_.end(), static_cast<std::size_t>(1), std::multiplies<std::size_t>()))
    {
        for (const auto& dim : shape_) {
            if (dim == 0) {
                throw std::invalid_argument("Tensor dimensions must be greater than zero.");
            }
        }
        // Compute strides for row-major indexing
        strides_[N - 1] = 1;
        for (int i = static_cast<int>(N) - 2; i >= 0; --i) {
            strides_[i] = strides_[i + 1] * shape_[i + 1];
        }
        // Initialize with zeros
        values_ = std::vector<T>(size_, static_cast<T>(0));
    }

    template <SizeLike... Dims> requires (sizeof...(Dims) == N)
    Tensor(Dims... dims) {
        *this = Tensor<T, N>(std::array<std::size_t, N>{static_cast<std::size_t>(dims)...});
    }

    Tensor() :
        shape_({}),
        size_(1),
        values_(1, static_cast<T>(0))
    {
        shape_.fill(1);
    }

    const std::array<std::size_t, N>& shape() const {
        return shape_;
    }

    const std::array<std::size_t, N>& strides() const {
        return strides_;
    }

    std::size_t stride(std::size_t dim) const {
        return strides_[dim];
    }

    std::size_t size() const {
        return size_;
    }

    std::vector<T>& values() {
        return values_;
    }

    const std::vector<T>& values() const {
        return values_;
    }

    std::string to_string() const {
        std::string result = "Tensor(\n\tshape=[";
        for (size_t i = 0; i < shape_.size(); ++i) {
            result += std::to_string(shape_[i]);
            if (i < shape_.size() - 1) {
                result += ", ";
            }
        }
        result += "],\n\tvalues=";
        // Compute strides for row-major indexing
        std::array<std::size_t, N> strides{};
        if (N > 0) {
            strides[N - 1] = 1;
            for (int i = static_cast<int>(N) - 2; i >= 0; --i) {
                strides[i] = strides[i + 1] * shape_[i + 1];
            }
        }
        // Recursive formatter: at dimension `dim`, starting from `offset`.
        std::function<std::string(std::size_t, std::size_t)> fmt =
            [&](std::size_t dim, std::size_t offset) -> std::string {
                std::string s = "\n";
                for (std::size_t i = 0; i < dim; ++i) {
                    s += "\t";
                }
                s += "[";
                if (dim + 1 == N) {
                    for (std::size_t i = 0; i < shape_[dim]; ++i) {
                        if (i) s += ",";
                        s += std::to_string(values_[offset + i]);
                    }
                } else {
                    for (std::size_t i = 0; i < shape_[dim]; ++i) {
                        if (i) s += ",";
                        s += fmt(dim + 1, offset + i * strides[dim]);
                    }
                }
                if (dim + 1 != N) {
                    s += "\n";
                    for (std::size_t i = 0; i < dim; ++i) {
                        s += "\t";
                    }
                }
                s += "]";
                return s;
            };
        if (size_ == 0) {
            result += "[]";
        } else {
            result += fmt(0, 0);
        }
        result += "\n)";
        return result;
    }

    template <SizeLike... Dims>
    static Tensor<T, N> zeros(Dims... dims) {
        return Tensor<T, N>(dims...);
    }

    static Tensor<T, N> zeros(const std::array<std::size_t, N>& shape) {
        return Tensor<T, N>(shape);
    }

    template <SizeLike... Dims>
    static Tensor<T, N> ones(Dims... dims) {
        // Create a tensor filled with ones
        Tensor<T, N> tensor(dims...);
        std::fill(tensor.values_.begin(), tensor.values_.end(), static_cast<T>(1));
        return tensor;
    }

    static Tensor<T, N> ones(const std::array<std::size_t, N>& shape) {
        // Create a tensor filled with ones
        Tensor<T, N> tensor(shape);
        std::fill(tensor.values_.begin(), tensor.values_.end(), static_cast<T>(1));
        return tensor;
    }
};

} // namespace sc

#endif // TENSOR_HPP