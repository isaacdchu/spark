#ifndef CONCEPTS_HPP
#define CONCEPTS_HPP

#include <concepts>

// floating point types
template <typename T>
concept FloatLike = std::floating_point<T>;

// dimension types
template <typename T>
concept SizeLike = std::integral<T>;

#endif // CONCEPTS_HPP