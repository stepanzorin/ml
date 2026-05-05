// ML (https://github.com/stepanzorin/ml)
// Copyright Text: 2026 Stepan Zorin <stz.hom@gmail.com>

#pragma once

#include <concepts>

namespace ml::common {

namespace detail {

template<typename T>
concept symbol = std::same_as<T, signed char> || std::same_as<T, unsigned char> || std::same_as<T, wchar_t>;

} // namespace detail

template<typename T>
concept feature = !detail::symbol<T> && (std::integral<T> || std::floating_point<T>);

template<typename T>
concept dataset_sample = requires(const T sample) {
    { sample.features };
    { sample.target };
};

} // namespace ml::common