// ML (https://github.com/stepanzorin/ml)
// Copyright Text: 2026 Stepan Zorin <stz.hom@gmail.com>

#pragma once

#include <concepts>

namespace ml::common {

template<typename T>
concept cvs_column = requires(const T sample) { std::default_initializable<T>; };

template<typename T>
concept dataset_sample = requires(const T sample) {
    { sample.features };
    { sample.target };
};

} // namespace ml::common