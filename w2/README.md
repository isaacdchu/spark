For C++26 support, change line 46-48 of `utility/small_vector.hpp` from 
```cpp
template <typename T>
struct IsPod : std::integral_constant<bool, std::is_standard_layout<T>::value &&
    std::is_trivial<T>::value> {};
```
to
```cpp
template <typename T>
struct IsPod : std::integral_constant<bool, std::is_standard_layout<T>::value &&
    std::is_trivially_default_constructible<T>::value && std::is_trivially_copyable<T>::value> {};
```
