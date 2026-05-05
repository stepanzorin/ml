// ML (https://github.com/stepanzorin/ml)
// Copyright Text: 2026 Stepan Zorin <stz.hom@gmail.com>

#pragma once

#include <concepts>
#include <vector>

#include "metrics/regression/mean_squared_error.hpp"

namespace ml::metrics::regression {

template<std::convertible_to<double> T>
[[nodiscard]] double root_mean_squared_error(const std::vector<T> &targets, const std::vector<T> &predictions) {
    return std::sqrt(mean_squared_error(targets, predictions));
}

} // namespace ml::metrics::regression