// ML (https://github.com/stepanzorin/ml)
// Copyright Text: 2026 Stepan Zorin <stz.hom@gmail.com>

#pragma once

#include <cmath>
#include <concepts>
#include <ranges>
#include <stdexcept>
#include <vector>

namespace ml::metrics::regression {

template<std::convertible_to<double> T>
[[nodiscard]] double mean_absolute_error(const std::vector<T> &targets, const std::vector<T> &predictions) {
    if (targets.size() != predictions.size()) {
        throw std::runtime_error{"Targets and predictions size mismatch"};
    }

    if (targets.empty()) {
        throw std::runtime_error{"Targets must not be empty"};
    }

    auto error_sum = 0.0;

    for (const auto &&[target, prediction] : std::views::zip(targets, predictions)) {
        error_sum += std::abs(prediction - target);
    }

    return error_sum / static_cast<double>(targets.size());
}

} // namespace ml::metrics::regression